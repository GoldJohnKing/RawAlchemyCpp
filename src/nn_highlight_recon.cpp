// SPDX-License-Identifier: AGPL-3.0-or-later
// Inpaint-opposed highlight reconstruction — see nn_highlight_recon.h.
//
// Deliberate deviations from the darktable reference (each required by this pipeline):
//   1. clips[c] = clipFactor UNIFORMLY. darktable (opposed.c:97) scales clips by the WB
//      coefficients icoeffs[c] because it runs opposed POST-WB in its default pipeline.
//      We run PRE-WB (design §2.3 step 4 → step 5), which is darktable's wbon=false path
//      where icoeffs={1,1,1} — so uniform clips is the faithful equivalent here, not a
//      lossy approximation. (Earlier revisions of this comment mis-attributed the uniform
//      clips to a nominal_white collapse; the actual cause is the pipeline position.)
//   2. correction = {1,1,1} (pre-WB). Equivalent to darktable's late=FALSE path.
//   3. clipFactor = 0.93 (design §2.3 step 4), not darktable's 0.987 default.
//   4. Single-threaded (no OpenMP). One-time per image; the tile-inference loop
//      dominates latency and the !anyClipped fast path keeps well-exposed images cheap.
//
// Everything else — the 3×3 refavg window, the cube-root opposing-mean formula, the
// two-ring dilation offsets, the lo=0.2*clips chroma bound, the 100-sample minimum, and
// the step-D max(inval, ref+chroma) reconstruction — is copied 1:1 from opposed.c.
// Port only; do not "improve" the bounds without re-deriving against upstream.

#include "nn_highlight_recon.h"

#include "nn_logging.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace rawalchemy {
namespace {

// lo_clips = 0.2 * clips  (opposed.c:314): lower bound for chroma sampling.
constexpr float kLoClipFrac = 0.2f;
// Minimum halo samples for a channel's chrominance offset (opposed.c:335, CFA path).
// darktable's sRAW/linear variant uses 30; the CFA path uses 100.
constexpr long kMinChromaSamples = 100;
// Dilation safe margin (opposed.c:307): the hollow dist-2..3 ring reaches ±3, so the
// dilated mask is only computed where col/row have ≥3 margin on each side.
constexpr int kMaskBorder = 3;

inline float fcube(float a) { return a * a * a; }

// Opposed estimate (linear space) for a sensel of `color`: mean of the cube roots of
// the two opposing channels' 3×3-neighborhood means, cubed back. Ported 1:1 from
// _calc_refavg (segbased.c:186-222) with linear=TRUE and correction={1,1,1}.
inline float calcRefavg(const float* cfa, int W, int H, int row, int col,
                        int color, const uint8_t* fc) {
    float sum[3] = {0.0f, 0.0f, 0.0f};
    int cnt[3] = {0, 0, 0};
    const int y0 = std::max(0, row - 1);
    const int x0 = std::max(0, col - 1);
    const int y1 = std::min(H - 1, row + 1) + 1;  // exclusive upper bound
    const int x1 = std::min(W - 1, col + 1) + 1;
    for (int dy = y0; dy < y1; ++dy) {
        const size_t rowBase = static_cast<size_t>(dy) * W;
        for (int dx = x0; dx < x1; ++dx) {
            const int c = fc[rowBase + dx];
            sum[c] += std::max(0.0f, cfa[rowBase + dx]);
            ++cnt[c];
        }
    }
    float cbrtMean[3];
    for (int c = 0; c < 3; ++c) {
        cbrtMean[c] = (cnt[c] > 0) ? std::cbrt(sum[c] / static_cast<float>(cnt[c])) : 0.0f;
    }
    // opposed pairs: R<-(G,B), G<-(R,B), B<-(R,G)  (opposed.c:52-54 / segbased.c:217-220)
    float crootRef;
    switch (color) {
        case 0:  crootRef = 0.5f * (cbrtMean[1] + cbrtMean[2]); break;  // R
        case 1:  crootRef = 0.5f * (cbrtMean[0] + cbrtMean[2]); break;  // G
        default: crootRef = 0.5f * (cbrtMean[0] + cbrtMean[1]); break;  // B
    }
    return fcube(crootRef);
}

// Morphological dilation of one mask plane: set if the center, any neighbor in the
// 3×3 ring, or any cell in the hollow dist-2..3 ring is set. The hollow ring is what
// selects the near-clipped halo carrying local chrominance. Ported 1:1 from
// _mask_dilated (opposed.c:62-79); `in` points at the center, w = mask stride.
// Caller guarantees a ≥3px safe margin on all sides.
inline bool maskDilated(const uint8_t* in, int w) {
    if (in[0]) return true;
    const int w2 = 2 * w;
    const int w3 = 3 * w;
    // 3×3 ring (radius 1)
    if (in[-w - 1] | in[-w] | in[-w + 1] |
        in[-1] |              in[1]  |
        in[w - 1]  | in[w]  | in[w + 1]) {
        return true;
    }
    // hollow ring (radius 2..3)
    return (in[-w3 - 2] | in[-w3 - 1] | in[-w3]    | in[-w3 + 1] | in[-w3 + 2] |
            in[-w2 - 3] | in[-w2 - 2] | in[-w2 - 1] | in[-w2]    | in[-w2 + 1] | in[-w2 + 2] | in[-w2 + 3] |
            in[-w - 3]  | in[-w - 2]  |               in[-w + 2] | in[-w + 3] |
            in[-3]      | in[-2]      |               in[2]      | in[3] |
            in[w - 3]   | in[w - 2]   |               in[w + 2]  | in[w + 3] |
            in[w2 - 3]  | in[w2 - 2]  | in[w2 - 1]   | in[w2]    | in[w2 + 1] | in[w2 + 2] | in[w2 + 3] |
            in[w3 - 2]  | in[w3 - 1]  | in[w3]       | in[w3 + 1] | in[w3 + 2]) != 0;
}

} // namespace

void reconstructHighlightsOpposed(float* cfa, int W, int H,
                                 const CfaPhase& phase, float clipFactor) {
    const int mW = W / 3;
    const int mH = H / 3;
    // The hollow dist-2..3 dilation ring needs ≥3 superpixels of margin on each side.
    if (mW < 2 * kMaskBorder + 1 || mH < 2 * kMaskBorder + 1) return;

    // Per-sensel color (0=R,1=G,2=B). Recon runs BEFORE mirror-pad, so the sensor color
    // at original (y,x) is the canonical color at the phase-aligned position (y+dy, x+dx)
    // — same lookup the WB pass uses. Precomputed once: calcRefavg's 3×3 window would
    // otherwise re-resolve canonicalCfaColor (modulo + branch) ~27× per pixel.
    std::vector<uint8_t> fc(static_cast<size_t>(W) * H);
    for (int y = 0; y < H; ++y) {
        const size_t rowBase = static_cast<size_t>(y) * W;
        for (int x = 0; x < W; ++x) {
            fc[rowBase + x] = static_cast<uint8_t>(
                canonicalCfaColor(y + phase.dy, x + phase.dx, phase));
        }
    }

    const float clips = clipFactor;          // uniform — see file-header deviation #1
    const float loClip = kLoClipFrac * clips;

    // 6 mask planes on the 1/3 superpixel grid: [0..2]=clipped vote, [3..5]=dilated.
    const size_t mSize = static_cast<size_t>(mW) * mH;
    std::vector<uint8_t> mask(6 * mSize, 0);
    uint8_t* clipped[3] = { mask.data(), mask.data() + mSize, mask.data() + 2 * mSize };
    uint8_t* dilated[3] = { mask.data() + 3 * mSize, mask.data() + 4 * mSize, mask.data() + 5 * mSize };

    // --- Step A: clipped-superpixel mask. Per 3×3 block, per-channel vote. (opposed.c:267-289)
    bool anyClipped = false;
    for (int my = 0; my < mH - 1; ++my) {
        for (int mx = 0; mx < mW - 1; ++mx) {
            int present[3] = {0, 0, 0};
            for (int y = 0; y < 3; ++y) {
                const int r = 3 * my + y;
                const size_t rowBase = static_cast<size_t>(r) * W;
                for (int x = 0; x < 3; ++x) {
                    const int c = 3 * mx + x;
                    if (cfa[rowBase + c] >= clips) ++present[fc[rowBase + c]];
                }
            }
            const size_t mdx = static_cast<size_t>(my) * mW + mx;
            for (int ch = 0; ch < 3; ++ch) {
                if (present[ch]) {
                    clipped[ch][mdx] = 1;
                    anyClipped = true;
                }
            }
        }
    }

    // Fast path: well-exposed image — nothing to reconstruct.
    if (!anyClipped) {
        nnlog::info("[NN] highlight recon: no sensels >= %.2f, skipping", clipFactor);
        return;
    }

    // --- Step B: dilate each plane. Safe-margin guard mirrors opposed.c:307-310.
    for (int row = 0; row < mH; ++row) {
        const bool safeRow = row >= kMaskBorder && row < mH - kMaskBorder - 1;
        for (int col = 0; col < mW; ++col) {
            const bool safe = safeRow && col >= kMaskBorder && col < mW - kMaskBorder - 1;
            const size_t mx = static_cast<size_t>(row) * mW + col;
            for (int ch = 0; ch < 3; ++ch) {
                dilated[ch][mx] = safe ? maskDilated(&clipped[ch][mx], mW) : clipped[ch][mx];
            }
        }
    }

    // --- Step C: global per-channel chrominance offset from the near-clipped halo.
    // (opposed.c:314-335). double accumulator keeps the reduction deterministic under
    // the per-file -fno-fast-math exception so cross-validate outputs are reproducible.
    double sum[3] = {0.0, 0.0, 0.0};
    long cnt[3] = {0, 0, 0};
    for (int y = 0; y < H; ++y) {
        const size_t rowBase = static_cast<size_t>(y) * W;
        const size_t mrowBase = static_cast<size_t>(y / 3) * mW;
        for (int x = 0; x < W; ++x) {
            const size_t i = rowBase + x;
            const int c = fc[i];
            const float v = cfa[i];
            if (v < clips && v > loClip && dilated[c][mrowBase + (x / 3)]) {
                sum[c] += static_cast<double>(v - calcRefavg(cfa, W, H, y, x, c, fc.data()));
                ++cnt[c];
            }
        }
    }
    float chroma[3];
    for (int c = 0; c < 3; ++c) {
        chroma[c] = (cnt[c] > kMinChromaSamples) ? static_cast<float>(sum[c] / cnt[c]) : 0.0f;
    }

    // Diagnostic: confirms the recon is running and shows what it found. If this
    // reports "no sensels" the clip threshold is too high for the image's normalization;
    // if chroma is ~0 the scene is neutrally white-balanced; large |chroma| flags a cast.
    nnlog::info("[NN] highlight recon: clip=%.2f chroma R=%+.4f G=%+.4f B=%+.4f "
                "(halo samples R=%ld G=%ld B=%ld)",
                clipFactor, chroma[0], chroma[1], chroma[2], cnt[0], cnt[1], cnt[2]);

    // --- Step D: reconstruct clipped sensels in place. (opposed.c:361-377)
    // max() is physical: a clipped sensel sits at sensor saturation, so the opposed
    // estimate may only raise it (the cube-root mean underestimates fully-saturated reads).
    for (int y = 0; y < H; ++y) {
        const size_t rowBase = static_cast<size_t>(y) * W;
        for (int x = 0; x < W; ++x) {
            const size_t i = rowBase + x;
            const float v = cfa[i];
            if (v >= clips) {
                const int c = fc[i];
                const float ref = calcRefavg(cfa, W, H, y, x, c, fc.data());
                cfa[i] = std::max(v, ref + chroma[c]);
            }
        }
    }
}

} // namespace rawalchemy
