// SPDX-License-Identifier: AGPL-3.0-or-later
//
// RATIO CORRECTED DEMOSAICING (RCD) for Bayer CFA sensors.
//
// Ported from darktable's rcd.c (Copyright (C) 2010-2026 darktable developers;
// original algorithm by Luis Sanz Rodríguez, RCD 2.3 @ 171125; tiling by
// Ingo Weyrich; performance tuning by Hanno Schwalm). Original code licensed
// under GNU GPL v3; this port retains GPL-compatibility via AGPL-3.0-or-later.
//
// The canonical darktable source lives at .reference/darktable/rcd.c. This
// port preserves the algorithmic structure 1:1 (border fill → directional
// discrimination → low-pass filter → green/R-B interpolation) while
// substituting STL/RAII scratch management for darktable's allocators.
//
// Helper substitutions (see Task 9 brief):
//   FC(row,col)                  → detail::fcColor(row, col, filters)
//   dt_calloc_align_float(n)     → AlignedVector<float>(n)  (zero-init, 64-byte align)
//   dt_alloc_align_float(n)      → AlignedVector<float>(n)
//   dt_free_align(p)             → (RAII destructor, no manual call)
//   DT_OMP_PRAGMA(parallel ...)  → #pragma omp parallel
//   DT_OMP_PRAGMA(for ...)       → #pragma omp for schedule(static) collapse(2)
//   DT_OMP_DECLARE_SIMD(...)     → omitted; -ffast-math + -mavx2 auto-vectorize
//   interpolatef(r, a, b)        → inline (see below)
//   CLIP(x)                      → inline (see below)
//   sqrf(x)                      → inline (see below)
//
// Algorithmic deviations from darktable:
//   1. Output format: darktable writes float4 (RGBA-interleaved); we write
//      planar RGB float[3*w*h] (R plane | G plane | B plane) per our public
//      contract in include/demosaic_rcd.h.
//   2. Border fallback: darktable calls demosaic_ppg() (a full PPG green/
//      red/blue pass) for the outer rim. darktable's demosaic_ppg is a
//      separate translation unit not provided in our reference bundle; we
//      substitute borderInterpolate(), a CFA-aware 3x3 average that matches
//      the spirit of darktable's border_interpolate(). Visual impact is
//      confined to a 7..9-pixel rim.
//   3. Scaler: darktable's rcd_demosaic takes a `scaler` parameter to undo
//      a normalization done by the caller; our public contract takes raw
//      [0,1] input and emits [0,1] output, so scaler and revscaler are
//      fixed to 1.0f.

#include "demosaic_rcd.h"
#include "aligned_allocator.h"
#include "cfa_lookup.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rawalchemy {
namespace {

// Linear-stride helpers in units of the tile width. Match darktable's
// w1..w4 macros; constexpr so they fold into address arithmetic.
constexpr int w1 = RCD_TILESIZE;
constexpr int w2 = 2 * RCD_TILESIZE;
constexpr int w3 = 3 * RCD_TILESIZE;
constexpr int w4 = 4 * RCD_TILESIZE;

// Tolerance constants (rcd.c eps / epssq).
constexpr float kEps   = 1e-5f;
constexpr float kEpsSq = 1e-10f;

// Our public contract takes raw [0,1] input and emits [0,1] output, so
// the darktable scaler / revscaler normalization is a no-op.
constexpr float kScaler    = 1.0f;
constexpr float kRevScaler = 1.0f;

inline float safeIn(float a, float scale) {
    return std::max(0.0f, a) * scale;
}

inline float sqrf(float x) { return x * x; }

// CLIP in darktable clamps to [0,1]; only applied to discrimination values
// already in that range. Under -ffast-math, std::min/max compile to MINSS/MAXSS.
inline float clipUnit(float x) {
    return std::max(0.0f, std::min(1.0f, x));
}

// r * a + (1 - r) * b — linear blend. Matches darktable's interpolatef.
inline float interpolatef(float r, float a, float b) {
    return r * a + (1.0f - r) * b;
}

// Border-region fallback: CFA-aware 3x3 averaging for the outer rim.
// Equivalent in spirit to darktable's border_interpolate() (the simpler
// variant used outside the full PPG path). Populates the rim of width
// `border` (pixels) with averaged neighbor samples; the rim's own color
// channel uses the pixel's raw value directly.
//
// Output is planar RGB: out[idx], out[idx + plane], out[idx + 2*plane].
void borderInterpolate(const float* in, float* out, int w, int h,
                       unsigned filters, int border) {
    const size_t plane = static_cast<size_t>(w) * static_cast<size_t>(h);
    for (int row = 0; row < h; ++row) {
        const bool row_in_border = (row < border) || (row >= h - border);
        for (int col = 0; col < w; ++col) {
            // Skip interior pixels — RCD tiles overwrite them later.
            if (!row_in_border && col >= border && col < w - border) continue;

            float sum[3] = {0.0f, 0.0f, 0.0f};
            int   cnt[3] = {0, 0, 0};
            for (int dy = -1; dy <= 1; ++dy) {
                const int y = row + dy;
                if (y < 0 || y >= h) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int x = col + dx;
                    if (x < 0 || x >= w) continue;
                    const unsigned c = detail::fcColor(y, x, filters);
                    sum[c] += std::max(0.0f, in[static_cast<size_t>(y) * w + x]);
                    cnt[c] += 1;
                }
            }
            const unsigned center_c = detail::fcColor(row, col, filters);
            const float center_val = std::max(0.0f, in[static_cast<size_t>(row) * w + col]);
            float rgb[3] = {0.0f, 0.0f, 0.0f};
            for (int c = 0; c < 3; ++c) {
                if (c == static_cast<int>(center_c)) {
                    rgb[c] = center_val;
                } else if (cnt[c] > 0) {
                    rgb[c] = sum[c] / static_cast<float>(cnt[c]);
                }
            }
            const size_t idx = static_cast<size_t>(row) * w + col;
            out[idx]                 = rgb[0];
            out[idx + plane]         = rgb[1];
            out[idx + 2 * plane]     = rgb[2];
        }
    }
}

} // namespace

void rcd_demosaic(const float* in, float* out, int w, int h, unsigned filters) {
    if (!in || !out || w <= 0 || h <= 0) return;

    // Stage 0: outer rim — CFA-aware averaging. Real RCD tiles overwrite
    // the interior; the rim stays as border interpolation.
    borderInterpolate(in, out, w, h, filters, RCD_BORDER);
    if (w < 2 * RCD_BORDER || h < 2 * RCD_BORDER) return;

    const int num_vertical   = 1 + (h - 2 * RCD_BORDER - 1) / RCD_TILEVALID;
    const int num_horizontal = 1 + (w - 2 * RCD_BORDER - 1) / RCD_TILEVALID;

    const size_t tile_area = static_cast<size_t>(RCD_TILESIZE) * RCD_TILESIZE;
    const size_t tile_half = tile_area / 2;
    const size_t rgb_area  = 3 * tile_area;
    const size_t out_plane = static_cast<size_t>(w) * h;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        // Per-thread scratch — allocated once per worker and reused across
        // every tile the thread processes. Total ~325 KB per thread.
        // AlignedVector zero-initializes (std::vector contract), matching
        // darktable's dt_calloc_align_float for VH_Dir (the only buffer
        // whose unwritten border is read by the refinement passes).
        AlignedVector<float> VH_Dir(tile_area);
        AlignedVector<float> PQ_Dir(tile_half);
        AlignedVector<float> cfa(tile_area);
        AlignedVector<float> P_CDiff_Hpf(tile_half);
        AlignedVector<float> Q_CDiff_Hpf(tile_half);
        AlignedVector<float> rgb(rgb_area);

        // lpf aliases PQ_Dir (no overlapping use — see rcd.c comment).
        float* const lpf = PQ_Dir.data();

        float* const VH_Dir_p = VH_Dir.data();
        float* const PQ_Dir_p = PQ_Dir.data();
        float* const cfa_p    = cfa.data();
        float* const P_p      = P_CDiff_Hpf.data();
        float* const Q_p      = Q_CDiff_Hpf.data();
        float* const rgb_p[3] = {
            rgb.data(),
            rgb.data() + tile_area,
            rgb.data() + 2 * tile_area,
        };

#ifdef _OPENMP
#pragma omp for schedule(static) collapse(2)
#endif
        for (int tile_vertical = 0; tile_vertical < num_vertical; ++tile_vertical) {
            for (int tile_horizontal = 0; tile_horizontal < num_horizontal; ++tile_horizontal) {
                const int rowStart = tile_vertical * RCD_TILEVALID;
                const int rowEnd   = std::min(rowStart + RCD_TILESIZE, h);
                const int colStart = tile_horizontal * RCD_TILEVALID;
                const int colEnd   = std::min(colStart + RCD_TILESIZE, w);
                const int tileRows = std::min(rowEnd - rowStart, RCD_TILESIZE);
                const int tileCols = std::min(colEnd - colStart, RCD_TILESIZE);

                // For partial tiles, zero the read-but-unwritten border of
                // VH_Dir and rgb so refinement reads deterministic zeros.
                if (rowStart + RCD_TILESIZE > h || colStart + RCD_TILESIZE > w) {
                    std::memset(VH_Dir_p, 0, sizeof(float) * tile_area);
                    std::memset(rgb.data(), 0, sizeof(float) * rgb_area);
                }

                // STEP 0: load mosaic data into cfa and seed the row's two
                // CFA-channel planes in rgb. The third plane for this row
                // is populated when the loop reaches the opposite-parity row.
                for (int row = rowStart; row < rowEnd; ++row) {
                    const int c0 = static_cast<int>(detail::fcColor(row, colStart, filters));
                    const int c1 = static_cast<int>(detail::fcColor(row, colStart + 1, filters));
                    int indx = (row - rowStart) * RCD_TILESIZE;
                    size_t in_indx = static_cast<size_t>(row) * w + colStart;
                    for (int col = colStart; col < colEnd; ++col, ++indx, ++in_indx) {
                        const float v = safeIn(in[in_indx], kRevScaler);
                        cfa_p[indx] = v;
                        rgb_p[c0][indx] = v;
                        rgb_p[c1][indx] = v;
                    }
                }

                // STEP 1.1: vertical HPF squared, rows 3..min(tileRows-3, 5).
                // These three (or fewer) row buffers feed the rolling-row
                // discriminator in step 1.2.
                float bufferV[3][RCD_TILESIZE - 8];
                for (int row = 3; row < std::min(tileRows - 3, 5); ++row) {
                    for (int col = 4, indx = row * RCD_TILESIZE + col;
                         col < tileCols - 4; ++col, ++indx) {
                        bufferV[row - 3][col - 4] = sqrf(
                            (cfa_p[indx - w3] - cfa_p[indx - w1]
                             - cfa_p[indx + w1] + cfa_p[indx + w3])
                            - 3.0f * (cfa_p[indx - w2] + cfa_p[indx + w2])
                            + 6.0f * cfa_p[indx]);
                    }
                }

                // STEP 1.2: rolling-row vertical HPF + horizontal HPF → VH_Dir.
                float bufferH[RCD_TILESIZE];
                float* V0 = bufferV[0];
                float* V1 = bufferV[1];
                float* V2 = bufferV[2];
                for (int row = 4; row < tileRows - 4; ++row) {
                    for (int col = 3, indx = row * RCD_TILESIZE + col;
                         col < tileCols - 3; ++col, ++indx) {
                        bufferH[col - 3] = sqrf(
                            (cfa_p[indx - 3] - cfa_p[indx - 1]
                             - cfa_p[indx + 1] + cfa_p[indx + 3])
                            - 3.0f * (cfa_p[indx - 2] + cfa_p[indx + 2])
                            + 6.0f * cfa_p[indx]);
                    }
                    for (int col = 4, indx = (row + 1) * RCD_TILESIZE + col;
                         col < tileCols - 4; ++col, ++indx) {
                        V2[col - 4] = sqrf(
                            (cfa_p[indx - w3] - cfa_p[indx - w1]
                             - cfa_p[indx + w1] + cfa_p[indx + w3])
                            - 3.0f * (cfa_p[indx - w2] + cfa_p[indx + w2])
                            + 6.0f * cfa_p[indx]);
                    }
                    for (int col = 4, indx = row * RCD_TILESIZE + col;
                         col < tileCols - 4; ++col, ++indx) {
                        const float V_Stat = std::max(kEpsSq,
                            V0[col - 4] + V1[col - 4] + V2[col - 4]);
                        const float H_Stat = std::max(kEpsSq,
                            bufferH[col - 4] + bufferH[col - 3] + bufferH[col - 2]);
                        VH_Dir_p[indx] = V_Stat / (V_Stat + H_Stat);
                    }
                    // Roll the line pointers: V0 := old V1, V1 := old V2,
                    // V2 := old V0 (which will be overwritten next iteration).
                    float* tmp = V0; V0 = V1; V1 = V2; V2 = tmp;
                }

                // STEP 2: low-pass filter (greens + R/B local samples).
                // Iterates non-green columns of each row (parity selected
                // by FC(row,0) & 1).
                for (int row = 2; row < tileRows - 2; ++row) {
                    const int parity = static_cast<int>(
                        detail::fcColor(row, 0, filters)) & 1;
                    int col = 2 + parity;
                    int indx = row * RCD_TILESIZE + col;
                    int lp_indx = indx / 2;
                    for (; col < tileCols - 2; col += 2, indx += 2, ++lp_indx) {
                        lpf[lp_indx] = cfa_p[indx]
                            + 0.5f  * (cfa_p[indx - w1]     + cfa_p[indx + w1]
                                     + cfa_p[indx - 1]      + cfa_p[indx + 1])
                            + 0.25f * (cfa_p[indx - w1 - 1] + cfa_p[indx - w1 + 1]
                                     + cfa_p[indx + w1 - 1] + cfa_p[indx + w1 + 1]);
                    }
                }

                // STEP 3: green at R and B CFA positions.
                for (int row = 4; row < tileRows - 4; ++row) {
                    const int parity = static_cast<int>(
                        detail::fcColor(row, 0, filters)) & 1;
                    int col = 4 + parity;
                    int indx = row * RCD_TILESIZE + col;
                    int lpindx = indx / 2;
                    for (; col < tileCols - 4; col += 2, indx += 2, ++lpindx) {
                        const float cfai = cfa_p[indx];

                        const float N_Grad = kEps
                            + std::abs(cfa_p[indx - w1] - cfa_p[indx + w1])
                            + std::abs(cfai - cfa_p[indx - w2])
                            + std::abs(cfa_p[indx - w1] - cfa_p[indx - w3])
                            + std::abs(cfa_p[indx - w2] - cfa_p[indx - w4]);
                        const float S_Grad = kEps
                            + std::abs(cfa_p[indx + w1] - cfa_p[indx - w1])
                            + std::abs(cfai - cfa_p[indx + w2])
                            + std::abs(cfa_p[indx + w1] - cfa_p[indx + w3])
                            + std::abs(cfa_p[indx + w2] - cfa_p[indx + w4]);
                        const float W_Grad = kEps
                            + std::abs(cfa_p[indx - 1]  - cfa_p[indx + 1])
                            + std::abs(cfai - cfa_p[indx - 2])
                            + std::abs(cfa_p[indx - 1]  - cfa_p[indx - 3])
                            + std::abs(cfa_p[indx - 2]  - cfa_p[indx - 4]);
                        const float E_Grad = kEps
                            + std::abs(cfa_p[indx + 1]  - cfa_p[indx - 1])
                            + std::abs(cfai - cfa_p[indx + 2])
                            + std::abs(cfa_p[indx + 1]  - cfa_p[indx + 3])
                            + std::abs(cfa_p[indx + 2]  - cfa_p[indx + 4]);

                        const float lpfi = lpf[lpindx];
                        const float N_Est = cfa_p[indx - w1] * (lpfi + lpfi)
                                          / (kEps + lpfi + lpf[lpindx - w1]);
                        const float S_Est = cfa_p[indx + w1] * (lpfi + lpfi)
                                          / (kEps + lpfi + lpf[lpindx + w1]);
                        const float W_Est = cfa_p[indx - 1]  * (lpfi + lpfi)
                                          / (kEps + lpfi + lpf[lpindx - 1]);
                        const float E_Est = cfa_p[indx + 1]  * (lpfi + lpfi)
                                          / (kEps + lpfi + lpf[lpindx + 1]);

                        const float V_Est = (S_Grad * N_Est + N_Grad * S_Est)
                                          / (N_Grad + S_Grad);
                        const float H_Est = (W_Grad * E_Est + E_Grad * W_Est)
                                          / (E_Grad + W_Grad);

                        const float VH_Central = VH_Dir_p[indx];
                        const float VH_Neighbourhood = 0.25f * (
                            VH_Dir_p[indx - w1 - 1] + VH_Dir_p[indx - w1 + 1] +
                            VH_Dir_p[indx + w1 - 1] + VH_Dir_p[indx + w1 + 1]);
                        const float VH_Disc =
                            (std::abs(0.5f - VH_Central) <
                             std::abs(0.5f - VH_Neighbourhood))
                            ? VH_Neighbourhood : VH_Central;

                        rgb_p[1][indx] = interpolatef(clipUnit(VH_Disc), H_Est, V_Est);
                    }
                }

                // STEP 4.0: P/Q diagonal HPF squared.
                for (int row = 3; row < tileRows - 3; ++row) {
                    for (int col = 3,
                              indx = row * RCD_TILESIZE + col,
                              indx2 = indx / 2;
                         col < tileCols - 3;
                         col += 2, indx += 2, ++indx2) {
                        P_p[indx2] = sqrf(
                            (cfa_p[indx - w3 - 3] - cfa_p[indx - w1 - 1]
                           - cfa_p[indx + w1 + 1] + cfa_p[indx + w3 + 3])
                            - 3.0f * (cfa_p[indx - w2 - 2] + cfa_p[indx + w2 + 2])
                            + 6.0f * cfa_p[indx]);
                        Q_p[indx2] = sqrf(
                            (cfa_p[indx - w3 + 3] - cfa_p[indx - w1 + 1]
                           - cfa_p[indx + w1 - 1] + cfa_p[indx + w3 - 3])
                            - 3.0f * (cfa_p[indx - w2 + 2] + cfa_p[indx + w2 - 2])
                            + 6.0f * cfa_p[indx]);
                    }
                }

                // STEP 4.1: P/Q discrimination strength.
                for (int row = 4; row < tileRows - 4; ++row) {
                    const int parity = static_cast<int>(
                        detail::fcColor(row, 0, filters)) & 1;
                    int col = 4 + parity;
                    int indx = row * RCD_TILESIZE + col;
                    int indx2 = indx / 2;
                    int indx3 = (indx - w1 - 1) / 2;
                    int indx4 = (indx + w1 - 1) / 2;
                    for (; col < tileCols - 4;
                         col += 2, indx += 2,
                         ++indx2, ++indx3, ++indx4) {
                        const float P_Stat = std::max(kEpsSq,
                            P_p[indx3] + P_p[indx2] + P_p[indx4 + 1]);
                        const float Q_Stat = std::max(kEpsSq,
                            Q_p[indx3 + 1] + Q_p[indx2] + Q_p[indx4]);
                        PQ_Dir_p[indx2] = P_Stat / (P_Stat + Q_Stat);
                    }
                }

                // STEP 4.2: R@B and B@R at red/blue CFA positions.
                // c = 2 - FC selects the opposite non-green channel.
                for (int row = 4; row < tileRows - 4; ++row) {
                    const int parity = static_cast<int>(
                        detail::fcColor(row, 0, filters)) & 1;
                    int col = 4 + parity;
                    const int c = 2 - static_cast<int>(
                        detail::fcColor(row, col, filters));
                    int indx    = row * RCD_TILESIZE + col;
                    int pqindx  = indx / 2;
                    int pqindx2 = (indx - w1 - 1) / 2;
                    int pqindx3 = (indx + w1 - 1) / 2;
                    for (; col < tileCols - 4;
                         col += 2, indx += 2,
                         ++pqindx, ++pqindx2, ++pqindx3) {
                        const float PQ_Central = PQ_Dir_p[pqindx];
                        const float PQ_Neighbourhood = 0.25f * (
                            PQ_Dir_p[pqindx2]     + PQ_Dir_p[pqindx2 + 1] +
                            PQ_Dir_p[pqindx3]     + PQ_Dir_p[pqindx3 + 1]);
                        const float PQ_Disc =
                            (std::abs(0.5f - PQ_Central) <
                             std::abs(0.5f - PQ_Neighbourhood))
                            ? PQ_Neighbourhood : PQ_Central;

                        const float NW_Grad = kEps
                            + std::abs(rgb_p[c][indx - w1 - 1] - rgb_p[c][indx + w1 + 1])
                            + std::abs(rgb_p[c][indx - w1 - 1] - rgb_p[c][indx - w3 - 3])
                            + std::abs(rgb_p[1][indx] - rgb_p[1][indx - w2 - 2]);
                        const float NE_Grad = kEps
                            + std::abs(rgb_p[c][indx - w1 + 1] - rgb_p[c][indx + w1 - 1])
                            + std::abs(rgb_p[c][indx - w1 + 1] - rgb_p[c][indx - w3 + 3])
                            + std::abs(rgb_p[1][indx] - rgb_p[1][indx - w2 + 2]);
                        const float SW_Grad = kEps
                            + std::abs(rgb_p[c][indx - w1 + 1] - rgb_p[c][indx + w1 - 1])
                            + std::abs(rgb_p[c][indx + w1 - 1] - rgb_p[c][indx + w3 - 3])
                            + std::abs(rgb_p[1][indx] - rgb_p[1][indx + w2 - 2]);
                        const float SE_Grad = kEps
                            + std::abs(rgb_p[c][indx - w1 - 1] - rgb_p[c][indx + w1 + 1])
                            + std::abs(rgb_p[c][indx + w1 + 1] - rgb_p[c][indx + w3 + 3])
                            + std::abs(rgb_p[1][indx] - rgb_p[1][indx + w2 + 2]);

                        const float NW_Est = rgb_p[c][indx - w1 - 1] - rgb_p[1][indx - w1 - 1];
                        const float NE_Est = rgb_p[c][indx - w1 + 1] - rgb_p[1][indx - w1 + 1];
                        const float SW_Est = rgb_p[c][indx + w1 - 1] - rgb_p[1][indx + w1 - 1];
                        const float SE_Est = rgb_p[c][indx + w1 + 1] - rgb_p[1][indx + w1 + 1];

                        const float P_Est = (NW_Grad * SE_Est + SE_Grad * NW_Est)
                                          / (NW_Grad + SE_Grad);
                        const float Q_Est = (NE_Grad * SW_Est + SW_Grad * NE_Est)
                                          / (NE_Grad + SW_Grad);

                        rgb_p[c][indx] = rgb_p[1][indx]
                                       + interpolatef(clipUnit(PQ_Disc), Q_Est, P_Est);
                    }
                }

                // STEP 4.3: R@G and B@G at green CFA positions.
                // Iterates green columns of each row (parity from FC(row,1)).
                for (int row = 4; row < tileRows - 4; ++row) {
                    const int parity = static_cast<int>(
                        detail::fcColor(row, 1, filters)) & 1;
                    int col = 4 + parity;
                    int indx = row * RCD_TILESIZE + col;
                    for (; col < tileCols - 4; col += 2, indx += 2) {
                        const float VH_Central = VH_Dir_p[indx];
                        const float VH_Neighbourhood = 0.25f * (
                            VH_Dir_p[indx - w1 - 1] + VH_Dir_p[indx - w1 + 1] +
                            VH_Dir_p[indx + w1 - 1] + VH_Dir_p[indx + w1 + 1]);
                        const float VH_Disc =
                            (std::abs(0.5f - VH_Central) <
                             std::abs(0.5f - VH_Neighbourhood))
                            ? VH_Neighbourhood : VH_Central;
                        const float rgb1v  = rgb_p[1][indx];
                        const float N1 = kEps + std::abs(rgb1v - rgb_p[1][indx - w2]);
                        const float S1 = kEps + std::abs(rgb1v - rgb_p[1][indx + w2]);
                        const float W1 = kEps + std::abs(rgb1v - rgb_p[1][indx - 2]);
                        const float E1 = kEps + std::abs(rgb1v - rgb_p[1][indx + 2]);

                        const float rgb1mw1 = rgb_p[1][indx - w1];
                        const float rgb1pw1 = rgb_p[1][indx + w1];
                        const float rgb1m1  = rgb_p[1][indx - 1];
                        const float rgb1p1  = rgb_p[1][indx + 1];

                        for (int c = 0; c <= 2; c += 2) {
                            const float SNabs = std::abs(rgb_p[c][indx - w1] - rgb_p[c][indx + w1]);
                            const float EWabs = std::abs(rgb_p[c][indx - 1]  - rgb_p[c][indx + 1]);

                            const float N_Grad = N1 + SNabs
                                + std::abs(rgb_p[c][indx - w1] - rgb_p[c][indx - w3]);
                            const float S_Grad = S1 + SNabs
                                + std::abs(rgb_p[c][indx + w1] - rgb_p[c][indx + w3]);
                            const float W_Grad = W1 + EWabs
                                + std::abs(rgb_p[c][indx - 1]  - rgb_p[c][indx - 3]);
                            const float E_Grad = E1 + EWabs
                                + std::abs(rgb_p[c][indx + 1]  - rgb_p[c][indx + 3]);

                            const float N_Est = rgb_p[c][indx - w1] - rgb1mw1;
                            const float S_Est = rgb_p[c][indx + w1] - rgb1pw1;
                            const float W_Est = rgb_p[c][indx - 1]  - rgb1m1;
                            const float E_Est = rgb_p[c][indx + 1]  - rgb1p1;

                            const float V_Est = (N_Grad * S_Est + S_Grad * N_Est)
                                              / (N_Grad + S_Grad);
                            const float H_Est = (E_Grad * W_Est + W_Grad * E_Est)
                                              / (E_Grad + W_Grad);

                            rgb_p[c][indx] = rgb1v
                                           + interpolatef(clipUnit(VH_Disc), H_Est, V_Est);
                        }
                    }
                }

                // Writeback: tile interior → output planar RGB. The
                // outermost tile uses the smaller RCD_MARGIN rim so the
                // visible image edge keeps more real RCD pixels; interior
                // tile boundaries use RCD_BORDER (matched overlaps).
                const int first_vertical = rowStart +
                    ((tile_vertical == 0) ? RCD_MARGIN : RCD_BORDER);
                const int last_vertical = rowEnd -
                    ((tile_vertical == num_vertical - 1) ? RCD_MARGIN : RCD_BORDER);
                const int first_horizontal = colStart +
                    ((tile_horizontal == 0) ? RCD_MARGIN : RCD_BORDER);
                const int last_horizontal = colEnd -
                    ((tile_horizontal == num_horizontal - 1) ? RCD_MARGIN : RCD_BORDER);

                for (int row = first_vertical; row < last_vertical; ++row) {
                    int idx = (row - rowStart) * RCD_TILESIZE
                            + (first_horizontal - colStart);
                    size_t o_idx = static_cast<size_t>(row) * w + first_horizontal;
                    for (int col = first_horizontal; col < last_horizontal;
                         ++col, ++idx, ++o_idx) {
                        out[o_idx]                     = kScaler * std::max(0.0f, rgb_p[0][idx]);
                        out[o_idx + out_plane]         = kScaler * std::max(0.0f, rgb_p[1][idx]);
                        out[o_idx + 2 * out_plane]     = kScaler * std::max(0.0f, rgb_p[2][idx]);
                    }
                }
            }
        }
        // RAII frees per-thread scratch; no manual dt_free_align.
    }
}

} // namespace rawalchemy
