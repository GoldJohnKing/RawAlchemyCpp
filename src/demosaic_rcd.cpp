/**
 * @file demosaic_rcd.cpp
 * @brief Phase 3 — RCD (Ratio Corrected Demosaicing) for Bayer sensors.
 *
 * CameraFTP - A Cross-platform FTP companion for camera photo transfer
 * Copyright (C) 2026 GoldJohnKing <GoldJohnKing@Live.cn>
 * SPDX-License-Identifier: AGPL-3.0-or-later
 *
 * ---
 *
 * Verbatim tiled port of RawTherapee's `rcd_demosaic` (dev branch) by
 * Luis Sanz Rodriguez (luis.sanz.rodriguez@gmail.com) and Ingo Weyrich
 * (heckflosse67@gmx.de). Original algorithm:
 *   https://github.com/LuisSR/RCD-Demosaicing  (Release 2.3 @ 171125)
 * Licensed under GNU GPL version 3 (GPL-3.0 → AGPL-3.0 compatible).
 *
 * Reference: rcd_rt.cc lines referenced in step comments below. The 5-step
 * algorithm, tiling structure, constants, and per-tile buffer reuse are
 * preserved verbatim. The only adaptations are data-layout (RT's 2D
 * rawData[row][col] / red[row][col] members → project's RawMosaic::data /
 * ImageBuffer::pixel), scale removal (RT's 16-bit domain → project's
 * pre-normalised [0,1] float), and CFA lookup (RT's FC() macro /
 * cfarray[2][2] → project's cfaColor(m,r,c)).
 */

#include "demosaic.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef RA_USE_OPENMP
#include <omp.h>
#endif

namespace rawalchemy {

// RT math helpers (rt_math.h). intp(a,b,c) = a*b + (1-a)*c — `a` weights `b`.
template <typename T>
static inline T SQR(T x) { return x * x; }

static inline float intp(float a, float b, float c) {
    return a * b + (1.0f - a) * c;
}

// Tiling constants (rcd_rt.cc L81-84). rcdBorder == tileBorder == 9 here;
// the distinction is preserved for faithfulness to RT's outermost-tile logic.
static constexpr int tileBorder  = 9;  // avoid tile-overlap errors
static constexpr int rcdBorder   = 9;
static constexpr int tileSize    = 194;
static constexpr int tileSizeN   = tileSize - 2 * tileBorder;  // 176
static constexpr int w1 = tileSize;            // 194
static constexpr int w2 = 2 * tileSize;        // 388
static constexpr int w3 = 3 * tileSize;        // 582
static constexpr int w4 = 4 * tileSize;        // 776

// Tolerance to avoid dividing by zero (rcd_rt.cc L89-90).
static constexpr float eps   = 1e-5f;
static constexpr float epssq = 1e-10f;

// Per-tile buffer sizes.
static constexpr int kPlaneSize   = tileSize * tileSize;          // 37636
static constexpr int kHalfPlane   = tileSize * tileSize / 2;      // 18818

// ============================================================
// Border fallback: 3x3 bilinear by CFA color
// ============================================================
// Replaces RT's border_interpolate(W,H,rcdBorder,...). Fills the outer
// rcdBorder-px ring with a per-pixel 3x3 average grouped by CFA color
// (R/G/B). Reads the original mosaic (m.data) with max(0,.) applied.
// CFA color 3 (G2, only on non-standard codes) collapses to 1 (G) for
// indexing; standard Bayer returns 0/1/2 only.
//
// Width matches rcdBorder (=9): the tiles write the interior
// [rcdBorder, H-rcdBorder) x [rcdBorder, W-rcdBorder), so this fills the
// complement outer ring. Iterates only the ring to avoid H*W interior no-ops.
static void borderInterpolate(const RawMosaic& m, ImageBuffer& out) {
    const int W = m.width;
    const int H = m.height;
    const float* bayer = m.data.data();
    constexpr int border = rcdBorder;

    auto body = [&](int row, int col) {
        float sums[3]   = {0.0f, 0.0f, 0.0f};
        float counts[3] = {0.0f, 0.0f, 0.0f};

        for (int dy = -1; dy <= 1; ++dy) {
            const int y = row + dy;
            if (y < 0 || y >= H) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                const int x = col + dx;
                if (x < 0 || x >= W) continue;
                int c = cfaColor(m, y, x);
                if (c == 3) c = 1;  // G2 -> G for indexing
                const float val = std::max(0.0f, bayer[static_cast<size_t>(y) * W + x]);
                sums[c]   += val;
                counts[c] += 1.0f;
            }
        }

        float* px = out.pixel(row, col);
        for (int c = 0; c < 3; ++c) {
            if (counts[c] > 0.0f) {
                px[c] = sums[c] / counts[c];
            }
            // else: leave the existing value (0.0f from ImageBuffer init).
        }
    };

    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int row = 0; row < H; ++row) {
        if (row < border || row >= H - border) {
            for (int col = 0; col < W; ++col) body(row, col);
        } else {
            for (int col = 0; col < border; ++col) body(row, col);
            for (int col = W - border; col < W; ++col) body(row, col);
        }
    }
}

// ============================================================
// rcdFC — RCD-specific CFA color lookup with G2->G collapse.
// ============================================================
// RT's FC() returns only 0/1/2 for standard Bayer. The project's cfaColor
// may return 3 for G2 on some Bayer encodings (e.g. Nikon NEF filters).
// RCD treats both greens as one channel (rgb[1]), so collapse 3 -> 1.
// Without this, `rgb[c0]` would index OOB and `2 - fc` would go negative.
static inline int rcdFC(const RawMosaic& m, int r, int c) {
    const int k = cfaColor(m, r, c);
    return k == 3 ? 1 : k;
}

// ============================================================
// Public API: rcdDemosaic
// ============================================================
// Tiled RCD demosaic. Each tile runs the 5-step algorithm on a local
// tileSize x tileSize buffer with tileBorder=9 overlap, then writes its
// interior (rcdBorder for outermost tiles, tileBorder for inner) to `out`.
// OpenMP parallelises the (tr, tc) tile loop; per-thread tile buffers are
// allocated once per thread and reused across tiles.
ImageBuffer rcdDemosaic(const RawMosaic& m) {
    const int W = m.width;
    const int H = m.height;

    if (m.filters == 9) {
        throw std::runtime_error("rcdDemosaic requires Bayer CFA (got X-Trans)");
    }
    // Note: the project's cfaColor may return 3 for G2 on some Bayer encodings
    // (Nikon etc.). RCD treats both greens as one channel, so rcdFC() below
    // collapses 3->1. RT's original 4-colour rejection check is therefore
    // not applicable (the project has no true 4-primary CFA, only R/G/B/G2).

    ImageBuffer out(W, H, 3);  // zero-initialised

    const int numTh = H / tileSizeN + ((H % tileSizeN) ? 1 : 0);
    const int numTw = W / tileSizeN + ((W % tileSizeN) ? 1 : 0);

    #ifdef RA_USE_OPENMP
    #pragma omp parallel
    #endif
    {
        // Per-thread tile buffers (rcd_rt.cc L98-104). std::vector zero-
        // initialises, matching calloc. lpf aliases PQ_Dir's storage
        // (rcd_rt.cc L102 comment: "reuse buffer, they don't overlap").
        std::vector<float> cfaStorage(static_cast<size_t>(kPlaneSize));
        float* const cfa = cfaStorage.data();

        std::vector<float> rgbStorage(static_cast<size_t>(3) * kPlaneSize);
        float* const rgb[3] = {
            rgbStorage.data(),
            rgbStorage.data() + static_cast<size_t>(kPlaneSize),
            rgbStorage.data() + static_cast<size_t>(2) * kPlaneSize,
        };

        std::vector<float> VH_DirStorage(static_cast<size_t>(kPlaneSize));
        float* const VH_Dir = VH_DirStorage.data();

        std::vector<float> PQ_DirStorage(static_cast<size_t>(kHalfPlane));
        float* const PQ_Dir = PQ_DirStorage.data();
        float* const lpf    = PQ_Dir;  // alias — non-overlapping usage

        std::vector<float> P_CDiff_HpfStorage(static_cast<size_t>(kHalfPlane));
        float* const P_CDiff_Hpf = P_CDiff_HpfStorage.data();

        std::vector<float> Q_CDiff_HpfStorage(static_cast<size_t>(kHalfPlane));
        float* const Q_CDiff_Hpf = Q_CDiff_HpfStorage.data();

        #ifdef RA_USE_OPENMP
        #pragma omp for schedule(dynamic) collapse(2) nowait
        #endif
        for (int tr = 0; tr < numTh; ++tr) {
            for (int tc = 0; tc < numTw; ++tc) {
                // Tile bounds (rcd_rt.cc L111-123). Degenerate tiles
                // (no interior) are skipped.
                const int rowStart = tr * tileSizeN;
                const int rowEnd   = std::min(rowStart + tileSize, H);
                if (rowStart + tileBorder == rowEnd - tileBorder) continue;

                const int colStart = tc * tileSizeN;
                const int colEnd   = std::min(colStart + tileSize, W);
                if (colStart + tileBorder == colEnd - tileBorder) continue;

                const int tileRows = std::min(rowEnd - rowStart, tileSize);
                const int tilecols = std::min(colEnd - colStart, tileSize);

                // ----------------------------------------------------------
                // Step 0: Populate CFA + both native-colour RGB channels
                // (rcd_rt.cc L125-131).
                // Adaptation: rawData[row][col]/scale → max(0, m.data[..]);
                // input is already normalised to [0,1].
                // ----------------------------------------------------------
                for (int row = rowStart; row < rowEnd; ++row) {
                    const int c0 = rcdFC(m, row, colStart);
                    const int c1 = rcdFC(m, row, colStart + 1);
                    for (int col = colStart, indx = (row - rowStart) * tileSize;
                         col < colEnd; ++col, ++indx) {
                        const float v = std::max(0.0f,
                            m.data[static_cast<size_t>(row) * W + col]);
                        cfa[indx] = rgb[c0][indx] = rgb[c1][indx] = v;
                    }
                }

                // ----------------------------------------------------------
                // Step 1: Find cardinal and diagonal interpolation directions
                // (rcd_rt.cc L133-165).
                // ----------------------------------------------------------
                float bufferV[3][tileSize - 8];

                // Step 1.1: Square of vertical & horizontal colour-difference
                // high-pass filter (rcd_rt.cc L137-141).
                for (int row = 3; row < std::min(tileRows - 3, 5); ++row) {
                    for (int col = 4, indx = row * tileSize + col;
                         col < tilecols - 4; ++col, ++indx) {
                        bufferV[row - 3][col - 4] = SQR(
                            (cfa[indx - w3] - cfa[indx - w1] - cfa[indx + w1] + cfa[indx + w3])
                            - 3.f * (cfa[indx - w2] + cfa[indx + w2])
                            + 6.f * cfa[indx]);
                    }
                }

                // Step 1.2: Vertical/horizontal directional discrimination
                // strength (rcd_rt.cc L143-165). bufferH is recomputed per
                // row; V0/V1/V2 rotate via swaps to slide the 3-row window.
                float bufferH[tileSize - 6];
                float* V0 = bufferV[0];
                float* V1 = bufferV[1];
                float* V2 = bufferV[2];
                for (int row = 4; row < tileRows - 4; ++row) {
                    for (int col = 3, indx = row * tileSize + col;
                         col < tilecols - 3; ++col, ++indx) {
                        bufferH[col - 3] = SQR(
                            (cfa[indx -  3] - cfa[indx -  1] - cfa[indx +  1] + cfa[indx +  3])
                            - 3.f * (cfa[indx -  2] + cfa[indx +  2])
                            + 6.f * cfa[indx]);
                    }
                    for (int col = 4, indx = (row + 1) * tileSize + col;
                         col < tilecols - 4; ++col, ++indx) {
                        V2[col - 4] = SQR(
                            (cfa[indx - w3] - cfa[indx - w1] - cfa[indx + w1] + cfa[indx + w3])
                            - 3.f * (cfa[indx - w2] + cfa[indx + w2])
                            + 6.f * cfa[indx]);
                    }
                    for (int col = 4, indx = row * tileSize + col;
                         col < tilecols - 4; ++col, ++indx) {
                        const float V_Stat = std::max(epssq, V0[col - 4] + V1[col - 4] + V2[col - 4]);
                        const float H_Stat = std::max(epssq, bufferH[col - 4] + bufferH[col - 3] + bufferH[col - 2]);
                        VH_Dir[indx] = V_Stat / (V_Stat + H_Stat);
                    }
                    // rotate pointers: row0,row1,row2 -> row1,row2,row0
                    std::swap(V0, V2);
                    std::swap(V0, V1);
                }

                // ----------------------------------------------------------
                // Step 2: Low-pass filter incorporating green, red and blue
                // local samples (rcd_rt.cc L167-174). Iterates non-green
                // CFA sites via column parity. lpf is half-resolution,
                // indexed by indx/2.
                // ----------------------------------------------------------
                for (int row = 2; row < tileRows - 2; ++row) {
                    for (int col = 2 + (rcdFC(m, row, 0) & 1),
                                 indx = row * tileSize + col,
                                 lpindx = indx / 2;
                         col < tilecols - 2;
                         col += 2, indx += 2, ++lpindx) {
                        lpf[lpindx] = cfa[indx]
                            + 0.5f  * (cfa[indx - w1] + cfa[indx + w1]
                                       + cfa[indx - 1]  + cfa[indx + 1])
                            + 0.25f * (cfa[indx - w1 - 1] + cfa[indx - w1 + 1]
                                       + cfa[indx + w1 - 1] + cfa[indx + w1 + 1]);
                    }
                }

                // ----------------------------------------------------------
                // Step 3: Populate green at blue/red CFA positions
                // (rcd_rt.cc L176-205). Cardinal gradients ±4, ratio-
                // corrected estimates, refined VH discrimination, intp blend.
                // ----------------------------------------------------------
                for (int row = 4; row < tileRows - 4; ++row) {
                    for (int col = 4 + (rcdFC(m, row, 0) & 1),
                                 indx = row * tileSize + col,
                                 lpindx = indx / 2;
                         col < tilecols - 4;
                         col += 2, indx += 2, ++lpindx) {
                        const float cfai = cfa[indx];

                        // Cardinal gradients.
                        const float N_Grad = eps + (std::fabs(cfa[indx - w1] - cfa[indx + w1]) + std::fabs(cfai - cfa[indx - w2])) + (std::fabs(cfa[indx - w1] - cfa[indx - w3]) + std::fabs(cfa[indx - w2] - cfa[indx - w4]));
                        const float S_Grad = eps + (std::fabs(cfa[indx - w1] - cfa[indx + w1]) + std::fabs(cfai - cfa[indx + w2])) + (std::fabs(cfa[indx + w1] - cfa[indx + w3]) + std::fabs(cfa[indx + w2] - cfa[indx + w4]));
                        const float W_Grad = eps + (std::fabs(cfa[indx -  1] - cfa[indx +  1]) + std::fabs(cfai - cfa[indx -  2])) + (std::fabs(cfa[indx -  1] - cfa[indx -  3]) + std::fabs(cfa[indx -  2] - cfa[indx -  4]));
                        const float E_Grad = eps + (std::fabs(cfa[indx -  1] - cfa[indx +  1]) + std::fabs(cfai - cfa[indx +  2])) + (std::fabs(cfa[indx +  1] - cfa[indx +  3]) + std::fabs(cfa[indx +  2] - cfa[indx +  4]));

                        // Cardinal pixel estimations (ratio-corrected).
                        const float lpfi = lpf[lpindx];
                        const float N_Est = cfa[indx - w1] * (lpfi + lpfi) / (eps + lpfi + lpf[lpindx - w1]);
                        const float S_Est = cfa[indx + w1] * (lpfi + lpfi) / (eps + lpfi + lpf[lpindx + w1]);
                        const float W_Est = cfa[indx -  1] * (lpfi + lpfi) / (eps + lpfi + lpf[lpindx -  1]);
                        const float E_Est = cfa[indx +  1] * (lpfi + lpfi) / (eps + lpfi + lpf[lpindx +  1]);

                        const float V_Est = (S_Grad * N_Est + N_Grad * S_Est) / (N_Grad + S_Grad);
                        const float H_Est = (W_Grad * E_Est + E_Grad * W_Est) / (E_Grad + W_Grad);

                        // Refined vertical/horizontal local discrimination.
                        const float VH_Central_Value = VH_Dir[indx];
                        const float VH_Neighbourhood_Value = 0.25f * ((VH_Dir[indx - w1 - 1] + VH_Dir[indx - w1 + 1]) + (VH_Dir[indx + w1 - 1] + VH_Dir[indx + w1 + 1]));
                        const float VH_Disc = std::fabs(0.5f - VH_Central_Value) < std::fabs(0.5f - VH_Neighbourhood_Value)
                                              ? VH_Neighbourhood_Value : VH_Central_Value;

                        // G@B and G@R interpolation.
                        rgb[1][indx] = intp(VH_Disc, H_Est, V_Est);
                    }
                }

                // ----------------------------------------------------------
                // Step 4: Populate the red and blue channels.
                // ----------------------------------------------------------

                // Step 4.0: Square of P/Q diagonal colour-difference high-
                // pass filter (rcd_rt.cc L211-217). Half-resolution.
                for (int row = 3; row < tileRows - 3; ++row) {
                    for (int col = 3, indx = row * tileSize + col, indx2 = indx / 2;
                         col < tilecols - 3;
                         col += 2, indx += 2, ++indx2) {
                        P_CDiff_Hpf[indx2] = SQR(
                            (cfa[indx - w3 - 3] - cfa[indx - w1 - 1] - cfa[indx + w1 + 1] + cfa[indx + w3 + 3])
                            - 3.f * (cfa[indx - w2 - 2] + cfa[indx + w2 + 2])
                            + 6.f * cfa[indx]);
                        Q_CDiff_Hpf[indx2] = SQR(
                            (cfa[indx - w3 + 3] - cfa[indx - w1 + 1] - cfa[indx + w1 - 1] + cfa[indx + w3 - 3])
                            - 3.f * (cfa[indx - w2 + 2] + cfa[indx + w2 - 2])
                            + 6.f * cfa[indx]);
                    }
                }

                // Step 4.1: P/Q diagonals directional discrimination
                // (rcd_rt.cc L219-226).
                for (int row = 4; row < tileRows - 4; ++row) {
                    for (int col = 4 + (rcdFC(m, row, 0) & 1),
                                 indx = row * tileSize + col,
                                 indx2 = indx / 2,
                                 indx3 = (indx - w1 - 1) / 2,
                                 indx4 = (indx + w1 - 1) / 2;
                         col < tilecols - 4;
                         col += 2, indx += 2, ++indx2, ++indx3, ++indx4) {
                        const float P_Stat = std::max(epssq, P_CDiff_Hpf[indx3] + P_CDiff_Hpf[indx2] + P_CDiff_Hpf[indx4 + 1]);
                        const float Q_Stat = std::max(epssq, Q_CDiff_Hpf[indx3 + 1] + Q_CDiff_Hpf[indx2] + Q_CDiff_Hpf[indx4]);
                        PQ_Dir[indx2] = P_Stat / (P_Stat + Q_Stat);
                    }
                }

                // Step 4.2: Populate red & blue at blue & red CFA positions
                // (rcd_rt.cc L228-257). `c = 2 - fc` selects the opposite
                // channel; constant within a Bayer row.
                for (int row = 4; row < tileRows - 4; ++row) {
                    for (int col = 4 + (rcdFC(m, row, 0) & 1),
                                 indx = row * tileSize + col,
                                 c = 2 - rcdFC(m, row, col),
                                 pqindx = indx / 2,
                                 pqindx2 = (indx - w1 - 1) / 2,
                                 pqindx3 = (indx + w1 - 1) / 2;
                         col < tilecols - 4;
                         col += 2, indx += 2, ++pqindx, ++pqindx2, ++pqindx3) {

                        // Refined P/Q diagonal local discrimination.
                        const float PQ_Central_Value = PQ_Dir[pqindx];
                        const float PQ_Neighbourhood_Value = 0.25f * (PQ_Dir[pqindx2] + PQ_Dir[pqindx2 + 1] + PQ_Dir[pqindx3] + PQ_Dir[pqindx3 + 1]);
                        const float PQ_Disc = (std::fabs(0.5f - PQ_Central_Value) < std::fabs(0.5f - PQ_Neighbourhood_Value))
                                              ? PQ_Neighbourhood_Value : PQ_Central_Value;

                        // Diagonal gradients.
                        const float NW_Grad = eps + std::fabs(rgb[c][indx - w1 - 1] - rgb[c][indx + w1 + 1]) + std::fabs(rgb[c][indx - w1 - 1] - rgb[c][indx - w3 - 3]) + std::fabs(rgb[1][indx] - rgb[1][indx - w2 - 2]);
                        const float NE_Grad = eps + std::fabs(rgb[c][indx - w1 + 1] - rgb[c][indx + w1 - 1]) + std::fabs(rgb[c][indx - w1 + 1] - rgb[c][indx - w3 + 3]) + std::fabs(rgb[1][indx] - rgb[1][indx - w2 + 2]);
                        const float SW_Grad = eps + std::fabs(rgb[c][indx - w1 + 1] - rgb[c][indx + w1 - 1]) + std::fabs(rgb[c][indx + w1 - 1] - rgb[c][indx + w3 - 3]) + std::fabs(rgb[1][indx] - rgb[1][indx + w2 - 2]);
                        const float SE_Grad = eps + std::fabs(rgb[c][indx - w1 - 1] - rgb[c][indx + w1 + 1]) + std::fabs(rgb[c][indx + w1 + 1] - rgb[c][indx + w3 + 3]) + std::fabs(rgb[1][indx] - rgb[1][indx + w2 + 2]);

                        // Diagonal colour differences.
                        const float NW_Est = rgb[c][indx - w1 - 1] - rgb[1][indx - w1 - 1];
                        const float NE_Est = rgb[c][indx - w1 + 1] - rgb[1][indx - w1 + 1];
                        const float SW_Est = rgb[c][indx + w1 - 1] - rgb[1][indx + w1 - 1];
                        const float SE_Est = rgb[c][indx + w1 + 1] - rgb[1][indx + w1 + 1];

                        // P/Q estimations.
                        const float P_Est = (NW_Grad * SE_Est + SE_Grad * NW_Est) / (NW_Grad + SE_Grad);
                        const float Q_Est = (NE_Grad * SW_Est + SW_Grad * NE_Est) / (NE_Grad + SW_Grad);

                        // R@B and B@R interpolation.
                        rgb[c][indx] = rgb[1][indx] + intp(PQ_Disc, Q_Est, P_Est);
                    }
                }

                // Step 4.3: Populate red & blue at green CFA positions
                // (rcd_rt.cc L259-301). Dual-channel (c=0; c<=2; c+=2)
                // cardinal colour-difference interpolation.
                for (int row = 4; row < tileRows - 4; ++row) {
                    for (int col = 4 + (rcdFC(m, row, 1) & 1),
                                 indx = row * tileSize + col;
                         col < tilecols - 4;
                         col += 2, indx += 2) {

                        const float VH_Central_Value = VH_Dir[indx];
                        const float VH_Neighbourhood_Value = 0.25f * ((VH_Dir[indx - w1 - 1] + VH_Dir[indx - w1 + 1]) + (VH_Dir[indx + w1 - 1] + VH_Dir[indx + w1 + 1]));
                        const float VH_Disc = (std::fabs(0.5f - VH_Central_Value) < std::fabs(0.5f - VH_Neighbourhood_Value))
                                              ? VH_Neighbourhood_Value : VH_Central_Value;
                        const float rgb1 = rgb[1][indx];
                        const float N1 = eps + std::fabs(rgb1 - rgb[1][indx - w2]);
                        const float S1 = eps + std::fabs(rgb1 - rgb[1][indx + w2]);
                        const float W1 = eps + std::fabs(rgb1 - rgb[1][indx -  2]);
                        const float E1 = eps + std::fabs(rgb1 - rgb[1][indx +  2]);

                        const float rgb1mw1 = rgb[1][indx - w1];
                        const float rgb1pw1 = rgb[1][indx + w1];
                        const float rgb1m1  = rgb[1][indx - 1];
                        const float rgb1p1  = rgb[1][indx + 1];
                        for (int c = 0; c <= 2; c += 2) {
                            // Cardinal gradients.
                            const float SNabs = std::fabs(rgb[c][indx - w1] - rgb[c][indx + w1]);
                            const float EWabs = std::fabs(rgb[c][indx -  1] - rgb[c][indx +  1]);
                            const float N_Grad = N1 + SNabs + std::fabs(rgb[c][indx - w1] - rgb[c][indx - w3]);
                            const float S_Grad = S1 + SNabs + std::fabs(rgb[c][indx + w1] - rgb[c][indx + w3]);
                            const float W_Grad = W1 + EWabs + std::fabs(rgb[c][indx -  1] - rgb[c][indx -  3]);
                            const float E_Grad = E1 + EWabs + std::fabs(rgb[c][indx +  1] - rgb[c][indx +  3]);

                            // Cardinal colour differences.
                            const float N_Est = rgb[c][indx - w1] - rgb1mw1;
                            const float S_Est = rgb[c][indx + w1] - rgb1pw1;
                            const float W_Est = rgb[c][indx -  1] - rgb1m1;
                            const float E_Est = rgb[c][indx +  1] - rgb1p1;

                            const float V_Est = (N_Grad * S_Est + S_Grad * N_Est) / (N_Grad + S_Grad);
                            const float H_Est = (E_Grad * W_Est + W_Grad * E_Est) / (E_Grad + W_Grad);

                            // R@G and B@G interpolation.
                            rgb[c][indx] = rgb1 + intp(VH_Disc, H_Est, V_Est);
                        }
                    }
                }

                // ----------------------------------------------------------
                // Step 5: Write tile interior to output (rcd_rt.cc L303-315).
                // Outermost tiles use rcdBorder, interior tiles use
                // tileBorder — disjoint rectangles, safe parallel writes.
                // Adaptation: red/green/blue[row][col] = max(0, rgb[c][idx])
                // with no scale multiplication (output stays in [0,1]).
                // ----------------------------------------------------------
                const int firstVertical   = rowStart + ((tr == 0)          ? rcdBorder : tileBorder);
                const int lastVertical    = rowEnd   - ((tr == numTh - 1)  ? rcdBorder : tileBorder);
                const int firstHorizontal = colStart + ((tc == 0)          ? rcdBorder : tileBorder);
                const int lastHorizontal  = colEnd   - ((tc == numTw - 1)  ? rcdBorder : tileBorder);
                for (int row = firstVertical; row < lastVertical; ++row) {
                    for (int col = firstHorizontal; col < lastHorizontal; ++col) {
                        const int idx = (row - rowStart) * tileSize + (col - colStart);
                        float* px = out.pixel(row, col);
                        px[0] = std::max(0.f, rgb[0][idx]);
                        px[1] = std::max(0.f, rgb[1][idx]);
                        px[2] = std::max(0.f, rgb[2][idx]);
                    }
                }
            }
        }
    }
    // Per-thread buffers freed by std::vector destructors at block exit.

    // Fill the outer rcdBorder-px ring with bilinear estimates
    // (rcd_rt.cc L341: border_interpolate(W,H,rcdBorder,...)).
    borderInterpolate(m, out);

    return out;
}

} // namespace rawalchemy
