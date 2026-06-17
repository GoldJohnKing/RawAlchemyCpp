// CameraFTP - A Cross-platform FTP companion for camera photo transfer
// Copyright (C) 2026 GoldJohnKing <GoldJohnKing@Live.cn>
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file raw_preprocess.cpp
 * @brief Phase 1 RAW preprocessing — black-level subtraction + hot-pixel fix.
 *
 * - subtractBlackLevel(): port of Python `subtract_black_level` (core.py:20-31).
 * - fixHotPixels(): C++ port of darktable's `src/iop/hotpixels.c` (GPL-3.0,
 *   (C) 2011-2026 darktable developers) — `_process_bayer` + `_process_xtrans`
 *   local 4 same-color neighbor test + MAX replacement. AGPL-3.0 compatible.
 *   Replaces the prior global-σ + median algorithm (under-detected real hot
 *   pixels and false-hit edges).
 */

#include "raw_preprocess.h"

#include <algorithm>
#include <vector>

#ifdef RA_USE_OPENMP
#include <omp.h>
#endif

namespace rawalchemy {

// ---- subtractBlackLevel ----
// Port of Python `subtract_black_level` (core.py:20-31):
//   pat_size = cfa_pattern.shape[0]
//   for r in range(pat_size):
//     for c in range(pat_size):
//       color = cfa_pattern[r, c]
//       bl_c  = float(bl[min(color, len(bl) - 1)])
//       result[r::pat_size, c::pat_size] =
//           np.maximum(sensor_raw[r::pat_size, c::pat_size] - bl_c, 0) /
//           (wl - bl_c)
void subtractBlackLevel(RawMosaic& m) {
    const int patSize = (m.filters == 9) ? 6 : 2;
    const int W = m.width;
    const int H = m.height;
    const float wl = m.maximum;

    // cblack holds 4 collapsed per-channel values. Clamp the color index to
    // 3 (== len(bl) - 1 in the Python oracle where bl has 4 entries).
    for (int r = 0; r < patSize; ++r) {
        for (int c = 0; c < patSize; ++c) {
            const int color = cfaColor(m, r, c);
            const int idx = std::min(color, 3);
            const float bl_c = m.cblack[idx];
            const float denom = wl - bl_c;

            // Guard against degenerate white==black.
            if (denom <= 0.0f) {
                continue;  // leave values untouched; nothing to normalize.
            }

            // result[r::patSize, c::patSize] = max(val - bl_c, 0) / denom
            for (int y = r; y < H; y += patSize) {
                float* row = m.data.data() + static_cast<size_t>(y) * W;
                for (int x = c; x < W; x += patSize) {
                    const float v = row[x] - bl_c;
                    row[x] = (v > 0.0f ? v : 0.0f) / denom;
                }
            }
        }
    }
}

// ---- fixHotPixels (port of darktable hotpixels.c) ----
//
// Detect hot sensor pixels based on the 4 surrounding same-color sites.
// Pixels having 3 or 4 (depending on the `permissive` flag) surrounding
// pixels that are dimmer than value*multiplier are considered "hot" and are
// replaced by the maximum of the qualifying neighbour pixels. The permissive
// variant allows correcting pairs of hot pixels in adjacent sites.
// Replacement using the maximum produces fewer artifacts when inadvertently
// replacing non-hot pixels. (Verbatim algorithm rationale from hotpixels.c.)
//
// Framework adaptations from darktable's source:
//   - iop layer stripped: no dt_iop_hotpixels_data_t / roi_out / module /
//     piece; params arrive directly as function arguments.
//   - DT_OMP FOR(...) -> `#pragma omp parallel for schedule(static)
//     reduction(+:fixed)` under the project's RA_USE_OPENMP guard.
//   - Double-buffering preserved (darktable ivoid/ovoid split): neighbors are
//     read from a pristine copy `in`; replacements write back to m.data. Since
//     neighbors sit at radial offset ±2 a single buffer could chain a replaced
//     pixel into a later pixel's read, so the copy is kept.
//   - FCNxtrans(...) -> cfaColor(m, r, c) (already negative-index safe for
//     X-Trans and filter-bit based for Bayer). The Bayer path keeps darktable's
//     hardcoded ±2 / ±width*2 same-color offsets (no FC() to replace).
//   - markfixed visualization dropped (no GUI marker in this project).
//   - gboolean -> bool.
void fixHotPixels(RawMosaic& m, float strength, float threshold, bool permissive) {
    const float multiplier = strength * 0.5f;       // darktable: strength / 2.0
    const int min_neighbours = permissive ? 3 : 4;
    const int W = m.width;
    const int H = m.height;

    // Border ring of 2 needs at least a 5x5 interior core; otherwise the
    // row/col loops below are empty. Early-out also skips the buffer copy.
    if (H < 5 || W < 5) return;

    // Double buffer: read neighbors from `in`, write replacements to m.data.
    // Equivalent to darktable copying ovoid<-ivoid then overwriting hot pixels;
    // non-hot pixels in m.data keep their original value untouched.
    std::vector<float> in(m.data);
    const float* const inBuf = in.data();
    float* const outBuf = m.data.data();

    int fixed = 0;

    if (m.filters == 9) {
        // X-Trans path — port of _process_xtrans (hotpixels.c L219-325).

        // For each cell of the 6x6 X-Trans array, pre-calculate the (x, y)
        // offsets of the four radially nearest same-color pixels. Built once
        // (serially) before the parallel row loop; the table is read-only and
        // shared across threads.
        int offsets[6][6][4][2];
        // Candidate offsets searched, nearest first, for like-colored pixels.
        const int search[20][2] = { { -1, 0 },
                                    { 1, 0 },
                                    { 0, -1 },
                                    { 0, 1 },
                                    { -1, -1 },
                                    { -1, 1 },
                                    { 1, -1 },
                                    { 1, 1 },
                                    { -2, 0 },
                                    { 2, 0 },
                                    { 0, -2 },
                                    { 0, 2 },
                                    { -2, -1 },
                                    { -2, 1 },
                                    { 2, -1 },
                                    { 2, 1 },
                                    { -1, -2 },
                                    { 1, -2 },
                                    { -1, 2 },
                                    { 1, 2 } };
        for (int j = 0; j < 6; ++j) {
            for (int i = 0; i < 6; ++i) {
                const int c = cfaColor(m, j, i);
                for (int s = 0, found = 0; s < 20 && found < 4; ++s) {
                    if (c == cfaColor(m, j + search[s][1], i + search[s][0])) {
                        offsets[j][i][found][0] = search[s][0];
                        offsets[j][i][found][1] = search[s][1];
                        ++found;
                    }
                }
            }
        }

        #ifdef RA_USE_OPENMP
        #pragma omp parallel for schedule(static) reduction(+:fixed)
        #endif
        for (int row = 2; row < H - 2; row++) {
            const float* inRow = inBuf + static_cast<size_t>(W) * row + 2;
            float* outRow = outBuf + static_cast<size_t>(W) * row + 2;
            for (int col = 2; col < W - 2; col++, inRow++, outRow++) {
                const float pix = *inRow;
                const float mid = pix * multiplier;
                if (pix > threshold) {
                    int count = 0;
                    float maxin = 0.0f;
                    for (int n = 0; n < 4; ++n) {
                        const int xx = offsets[row % 6][col % 6][n][0];
                        const int yy = offsets[row % 6][col % 6][n][1];
                        // size_t multiply mirrors darktable's `yy * (size_t)width`:
                        // modular arithmetic reproduces the signed offset exactly.
                        const float other = *(inRow + xx + yy * static_cast<size_t>(W));
                        if (mid > other) {
                            count++;
                            if (other > maxin) maxin = other;
                        }
                    }
                    if (count >= min_neighbours) {
                        *outRow = maxin;
                        fixed++;
                    }
                }
            }
        }
    } else {
        // Bayer path — port of _process_bayer (hotpixels.c L105-158).
        // Same-color neighbors at radial offset ±2 cols / ±2 rows: stride 2
        // always lands on the same CFA color for any standard 2x2 Bayer pattern.
        const int widthx2 = W * 2;

        #ifdef RA_USE_OPENMP
        #pragma omp parallel for schedule(static) reduction(+:fixed)
        #endif
        for (int row = 2; row < H - 2; row++) {
            const float* inRow = inBuf + static_cast<size_t>(W) * row + 2;
            float* outRow = outBuf + static_cast<size_t>(W) * row + 2;
            for (int col = 2; col < W - 2; col++, inRow++, outRow++) {
                const float pix = *inRow;
                const float mid = pix * multiplier;
                if (pix > threshold) {
                    int count = 0;
                    float maxin = 0.0f;
                    float other;
                    #define TESTONE(OFFSET)                                  \
                        other = inRow[OFFSET];                               \
                        if (mid > other) {                                   \
                            count++;                                         \
                            if (other > maxin) maxin = other;                \
                        }
                    TESTONE(-2);
                    TESTONE(-widthx2);
                    TESTONE(+2);
                    TESTONE(+widthx2);
                    #undef TESTONE
                    if (count >= min_neighbours) {
                        *outRow = maxin;
                        fixed++;
                    }
                }
            }
        }
    }

    (void)fixed;  // diagnostic counter kept for OpenMP reduction fidelity
}

} // namespace rawalchemy
