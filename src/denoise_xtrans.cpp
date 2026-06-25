// SPDX-License-Identifier: AGPL-3.0-or-later
//
// X-TRANS CFA WAVELET DENOISE.
//
// Ported from darktable src/iop/rawdenoise.c function wavelet_denoise_xtrans
// (Copyright (C) 2011-2026 darktable developers; GNU GPL v3). Original licensed
// under GPL-3.0-or-later; AGPL-3.0-or-later is one-way compatible per FSF.
//
// The canonical darktable source lives at .reference/darktable/rawdenoise.c.
//
// Algorithmic structure preserved 1:1:
//   - allocate width*(height+2) scratch; fimg points one row in so that
//     fimg[-width] (row above) is always a valid write target for edge splats;
//   - for each color c in {R,G,B}: seed top/bottom rows to 0.5, splat every
//     sensel of color c (sqrt-variance-stabilized) onto fimg and its neighbors
//     (green -> right+down; red/blue -> all 8 neighbors), fill any residual
//     edge gaps from the nearest same-color neighbor, call dwt_denoise, then
//     square and write back only at color-c CFA sites;
//   - out is written channel-by-channel, so `in` and `out` must not alias.
//
// Deviations from darktable (documented):
//   1. Extraction is single-threaded. darktable parallelizes it with a manual
//      chunk + boundary-restore scheme; we drop that (the splat is cheap vs the
//      wavelet transform, which stays parallel inside dwt_denoise). The
//      distribute-back loop and dwt_denoise remain OpenMP-parallel.
//   2. compute_channel_noise collapses to noise[i] = noise_all[i] * threshold
//      under darktable's default (neutral) per-channel force curves. We bake
//      that in; per-channel sliders are out of scope.
//   3. Domain: our caller runs this PRE-white-balance, via LibRaw's
//      pre_scalecolors_cb hook (black-subtracted raw CFA) — the same domain
//      darktable's rawdenoise operates in. The sqrt stabilizer and noise floor
//      are therefore correctly calibrated; no domain compensation is needed.

#include "denoise_xtrans.h"
#include "aligned_allocator.h"
#include "cfa_lookup.h"
#include "dwt_denoise.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

#ifdef RA_USE_OPENMP
#include <omp.h>
#endif

namespace rawalchemy {
namespace {

// Per-scale noise floor (darktable compute_channel_noise, neutral curves).
// First DWT_DENOISE_BANDS entries of {0.8002, 0.2735, 0.1202, 0.0585, 0.0291, ...}.
inline constexpr float NOISE_ALL[DWT_DENOISE_BANDS] = {
    0.8002f, 0.2735f, 0.1202f, 0.0585f, 0.0291f
};

// Variance-stabilizing transform: sqrt, clamped at 0. Matches darktable vstransform.
inline float vstransform(float v) { return std::sqrt(std::max(0.0f, v)); }

// Splat sensels of color `c` from `in` into the dense buffer `fimg`. Single
// row-major pass; the pre-row (fimg[-width..]) and post-row make neighbor
// writes at the top/bottom edges safe. Port of the per-chunk extraction body in
// darktable wavelet_denoise_xtrans, de-parallelized.
void extractChannel(const float* in, float* fimg, int w, int h,
                    const char xt[6][6], int c) {
    // Seed the top and bottom rows so splats that miss them have a defined value.
    for (int col = 0; col < w; ++col) {
        fimg[col] = 0.5f;
        fimg[static_cast<size_t>(h - 1) * w + col] = 0.5f;
    }

    for (int row = 0; row < h; ++row) {
        const float* inp = in + static_cast<size_t>(row) * w;
        float* fimgp = fimg + static_cast<size_t>(row) * w;

        // First column: a non-green sensel of color c splats up/right.
        if (c != 1 && xtransColor(row, 0, xt) == c) {
            const float d = vstransform(inp[0]);
            fimgp[0] = fimgp[-w] = fimgp[-w + 1] = d;
        }
        for (int col = (c != 1); col < w - 1; ++col) {
            if (xtransColor(row, col, xt) != c) continue;
            const float d = vstransform(inp[col]);
            fimgp[col] = d;
            if (c == 1) {
                // Green: copy right and down. X-Trans greens tile so this fills
                // all positions except the leftmost/rightmost columns.
                fimgp[col + 1] = fimgp[col + w] = d;
            } else {
                // Red/blue: copy to all 8 neighbors. Two greens can lie between
                // same-color sensels, so each splat covers one of them.
                fimgp[col - w - 1] = fimgp[col - w] = fimgp[col - w + 1] = d;
                fimgp[col - 1] = fimgp[col + 1] = d;
                if (row < h - 1)
                    fimgp[col + w - 1] = fimgp[col + w] = fimgp[col + w + 1] = d;
            }
        }
        // Leftmost column: if it is NOT color c, borrow from the nearest same-color neighbor.
        if (xtransColor(row, 0, xt) != c) {
            int src = 0;  // fallback: current sensel even if wrong color
            if (row > 1 && xtransColor(row - 1, 0, xt) == c) src = -w;
            else if (xtransColor(row, 1, xt) == c) src = 1;
            else if (row > 1 && xtransColor(row - 1, 1, xt) == c) src = -w + 1;
            fimgp[0] = vstransform(inp[src]);
        }
        // Rightmost column.
        if (c != 1 && xtransColor(row, w - 1, xt) == c) {
            const float d = vstransform(inp[w - 1]);
            fimgp[w - 2] = fimgp[w - 1] = fimgp[-1] = d;
        } else if (xtransColor(row, w - 1, xt) != c) {
            int src = w - 1;
            if (xtransColor(row, w - 2, xt) == c) src = w - 2;
            else if (row > 1 && xtransColor(row - 1, w - 1, xt) == c) src = -1;
            else if (row > 1 && xtransColor(row - 1, w - 2, xt) == c) src = -2;
            fimgp[w - 1] = vstransform(inp[src]);
        }
    }
}

} // namespace

void denoise_xtrans(const float* in, float* out, int w, int h,
                    const char xt[6][6], float threshold) {
    const size_t n = static_cast<size_t>(w) * h;
    if (n == 0) return;
    if (threshold <= 0.0f || in == out) {
        // Passthrough on zero threshold; never alias in/out across channels.
        if (in != out) std::copy(in, in + n, out);
        return;
    }

    // Per-scale thresholds under neutral force curves: noise[i] = noise_all[i] * threshold.
    float noise[DWT_DENOISE_BANDS];
    for (int i = 0; i < DWT_DENOISE_BANDS; ++i) noise[i] = NOISE_ALL[i] * threshold;

    // width*(height+2): one pad row above and below so neighbor splats at the
    // top/bottom edges are always in-bounds. fimg points one row in.
    // (AlignedVector throws std::bad_alloc on OOM like the rest of the codebase;
    // darktable returns NULL and passes through, but we propagate to the FFI
    // boundary consistently with the demosaic port.)
    AlignedVector<float, 64> buf(static_cast<size_t>(w) * (h + 2), 0.0f);
    float* fimg = buf.data() + w;

    for (int c = 0; c < 3; ++c) {
        extractChannel(in, fimg, w, h, xt, c);
        dwt_denoise(fimg, w, h, DWT_DENOISE_BANDS, noise);

        // Distribute the denoised channel back to its CFA sites (squaring undoes
        // the sqrt variance-stabilizing transform). Parallel over rows.
#ifdef RA_USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int row = 0; row < h; ++row) {
            const float* fimgp = fimg + static_cast<size_t>(row) * w;
            float* outp = out + static_cast<size_t>(row) * w;
            for (int col = 0; col < w; ++col)
                if (xtransColor(row, col, xt) == c) {
                    const float d = fimgp[col];
                    outp[col] = d * d;
                }
        }
    }
}

float computeXtransDenoiseThreshold(float iso, float manualThreshold) {
    if (manualThreshold >= 0.0f) return manualThreshold;  // 0 = off, >0 = literal
    // Auto: flat 0.01 across all ISO. denoise_xtrans applies a sqrt variance-
    // stabilizing transform (Anscombe), which makes the shot-noise threshold
    // ISO-independent — so, like darktable's rawdenoise (which has no ISO logic
    // and defaults to 0.01), we use a single fixed value. An ISO ramp would
    // re-introduce the coupling the VST is there to eliminate.
    (void)iso;
    return 0.01f;
}

} // namespace rawalchemy
