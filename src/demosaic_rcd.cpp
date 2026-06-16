/**
 * @file demosaic_rcd.cpp
 * @brief Phase 3 — RCD (Ratio Corrected Demosaicing) for Bayer sensors.
 *
 * Direct C++/OpenMP port of the Python Taichi reference
 * `raw_alchemy.demosaic` (demosaic.py:50-540), a Taichi GPU port of
 * darktable's `demosaic_rcd.cl` (algorithm by Luis Sanz Rodríguez).
 *
 * Each Taichi `@ti.kernel` over `ti.ndrange((4, h-4), (4, w-4))` becomes a
 * C++ `#pragma omp parallel for collapse(2)` over the same interior range.
 * The 4-pixel border is filled separately by bilinear interpolation.
 *
 * CFA lookups use the canonical `cfaColor(m, r, c)` from raw_mosaic.h
 * (returns 0=R, 1=G, 2=B, 3=G2). We treat color==1 || color==3 as green;
 * standard Bayer filter codes only ever produce 0/1/2 so this is a safe
 * superset that also handles non-standard codes encoding G2 distinctly.
 *
 * Intermediate planes (cfa, rgb0/1/2, VH_dir, lpf, p/q_diff, PQ_dir) use
 * DemosaicPlane: F16 storage on ARM64 / F32 elsewhere. Arithmetic is always
 * F32 — reads upcast via operator[], writes downcast via set(). Input
 * RawMosaic::data and output ImageBuffer::data stay float32; conversion
 * happens at the populate/write-output boundaries automatically.
 *
 * Constants (demosaic.py:22-23): EPS=1e-5f, EPSSQ=1e-10f, border=4.
 */

#include "demosaic.h"

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef RA_USE_OPENMP
#include <omp.h>
#endif

namespace rawalchemy {

// Reference constants (demosaic.py:22-23).
static constexpr float kEps   = 1e-5f;
static constexpr float kEpsSq = 1e-10f;
static constexpr int   kBorder = 4;

// Helpers (demosaic.py:36-43).
static inline float clipf(float x) { return std::min(std::max(x, 0.0f), 1.0f); }
static inline float fsquare(float x) { return x * x; }

// Free a vector's allocated memory immediately (mirrors Python `del`).
template <typename T>
static inline void freeVector(std::vector<T>& v) {
    std::vector<T>().swap(v);
}
// Overload for DemosaicPlane (intermediate planes use F16 storage on ARM64).
static inline void freeVector(DemosaicPlane& p) {
    p.clear();
}

// Green-position test: cfaColor returns 1 (G) or 3 (G2) for green sites.
static inline bool isGreen(int color) { return color == 1 || color == 3; }

// ============================================================
// Step 0: Populate CFA + RGB planes (demosaic.py:50-68)
// ============================================================
// cfa[r,c] = max(0, bayer[r,c]); route into rgb0 (R) / rgb1 (G) / rgb2 (B)
// by CFA color. Green sites (color 1 or 3) -> rgb1.
static void rcdPopulate(const RawMosaic& m,
                        DemosaicPlane& cfa,
                        DemosaicPlane& rgb0,
                        DemosaicPlane& rgb1,
                        DemosaicPlane& rgb2) {
    const int W = m.width;
    const int H = m.height;
    const float* bayer = m.data.data();

    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            const size_t idx = static_cast<size_t>(row) * W + col;
            const float val = std::max(0.0f, bayer[idx]);
            cfa.set(idx, val);
            const int color = cfaColor(m, row, col);
            if (color == 0) {
                rgb0.set(idx, val);
            } else if (isGreen(color)) {
                rgb1.set(idx, val);
            } else {  // color == 2 (B)
                rgb2.set(idx, val);
            }
        }
    }
}

// ============================================================
// Step 1: Vertical/Horizontal discrimination (demosaic.py:75-101)
// ============================================================
// 9-tap vertical & horizontal high-pass over a 3-point neighborhood;
// VH_dir = V_Stat / (V_Stat + H_Stat).
static void rcdStep1(const DemosaicPlane& cfa,
                     DemosaicPlane& VH_dir,
                     int W, int H) {
    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int row = kBorder; row < H - kBorder; ++row) {
        for (int col = kBorder; col < W - kBorder; ++col) {
            // Vertical high-pass (3-point neighborhood in row direction).
            float V_Stat = kEpsSq;
            for (int dr = -1; dr <= 1; ++dr) {
                const int r = row + dr;
                const float v = (cfa[(r - 3) * W + col] - cfa[(r - 1) * W + col]
                                 - cfa[(r + 1) * W + col] + cfa[(r + 3) * W + col]
                                 - 3.0f * (cfa[(r - 2) * W + col] + cfa[(r + 2) * W + col])
                                 + 6.0f * cfa[r * W + col]);
                V_Stat += v * v;
            }
            V_Stat = std::max(kEpsSq, V_Stat);

            // Horizontal high-pass (3-point neighborhood in col direction).
            float H_Stat = kEpsSq;
            for (int dc = -1; dc <= 1; ++dc) {
                const int c = col + dc;
                const float hv = (cfa[row * W + (c - 3)] - cfa[row * W + (c - 1)]
                                  - cfa[row * W + (c + 1)] + cfa[row * W + (c + 3)]
                                  - 3.0f * (cfa[row * W + (c - 2)] + cfa[row * W + (c + 2)])
                                  + 6.0f * cfa[row * W + c]);
                H_Stat += hv * hv;
            }
            H_Stat = std::max(kEpsSq, H_Stat);

            VH_dir.set(row * W + col, V_Stat / (V_Stat + H_Stat));
        }
    }
}

// ============================================================
// Step 2: Low-pass filter at R/B positions (demosaic.py:108-123)
// ============================================================
// At non-green CFA positions: lpf = cfa + 0.5*(N+S+E+W) + 0.25*(4 diag).
// Range is (2, h-2) x (2, w-2) — narrower border than other steps.
static void rcdStep2(const RawMosaic& m,
                     const DemosaicPlane& cfa,
                     DemosaicPlane& lpf,
                     int W, int H) {
    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int row = 2; row < H - 2; ++row) {
        for (int col = 2; col < W - 2; ++col) {
            if (isGreen(cfaColor(m, row, col))) continue;  // only R/B sites
            const size_t i = static_cast<size_t>(row) * W + col;
            lpf.set(i, (cfa[i]
                     + 0.5f * (cfa[(row - 1) * W + col] + cfa[(row + 1) * W + col]
                               + cfa[row * W + (col - 1)] + cfa[row * W + (col + 1)])
                     + 0.25f * (cfa[(row - 1) * W + (col - 1)] + cfa[(row - 1) * W + (col + 1)]
                                + cfa[(row + 1) * W + (col - 1)] + cfa[(row + 1) * W + (col + 1)])));
        }
    }
}

// ============================================================
// Step 3: Green interpolation at R/B positions (demosaic.py:130-182)
// ============================================================
// Refined VH discrimination; cardinal gradients (±4); ratio-corrected
// estimates; blend V_Est/H_Est by clipf(VH_Disc).
static void rcdStep3(const RawMosaic& m,
                     const DemosaicPlane& cfa,
                     const DemosaicPlane& lpf,
                     DemosaicPlane& rgb1,
                     const DemosaicPlane& VH_dir,
                     int W, int H) {
    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int row = kBorder; row < H - kBorder; ++row) {
        for (int col = kBorder; col < W - kBorder; ++col) {
            if (isGreen(cfaColor(m, row, col))) continue;  // skip green sites

            // Refined VH discrimination: pick central vs 4-corner-avg,
            // whichever is closer to 0.5.
            const float VH_Central = VH_dir[row * W + col];
            const float VH_Neigh = 0.25f * (VH_dir[(row - 1) * W + (col - 1)] + VH_dir[(row - 1) * W + (col + 1)]
                                            + VH_dir[(row + 1) * W + (col - 1)] + VH_dir[(row + 1) * W + (col + 1)]);
            const float VH_Disc = (std::fabs(0.5f - VH_Central) < std::fabs(0.5f - VH_Neigh))
                                  ? VH_Neigh : VH_Central;

            const float cfai = cfa[row * W + col];

            // Cardinal gradients.
            const float N_Grad = (kEps + std::fabs(cfa[(row - 1) * W + col] - cfa[(row + 1) * W + col])
                                  + std::fabs(cfai - cfa[(row - 2) * W + col])
                                  + std::fabs(cfa[(row - 1) * W + col] - cfa[(row - 3) * W + col])
                                  + std::fabs(cfa[(row - 2) * W + col] - cfa[(row - 4) * W + col]));
            const float S_Grad = (kEps + std::fabs(cfa[(row + 1) * W + col] - cfa[(row - 1) * W + col])
                                  + std::fabs(cfai - cfa[(row + 2) * W + col])
                                  + std::fabs(cfa[(row + 1) * W + col] - cfa[(row + 3) * W + col])
                                  + std::fabs(cfa[(row + 2) * W + col] - cfa[(row + 4) * W + col]));
            const float W_Grad = (kEps + std::fabs(cfa[row * W + (col - 1)] - cfa[row * W + (col + 1)])
                                  + std::fabs(cfai - cfa[row * W + (col - 2)])
                                  + std::fabs(cfa[row * W + (col - 1)] - cfa[row * W + (col - 3)])
                                  + std::fabs(cfa[row * W + (col - 2)] - cfa[row * W + (col - 4)]));
            const float E_Grad = (kEps + std::fabs(cfa[row * W + (col + 1)] - cfa[row * W + (col - 1)])
                                  + std::fabs(cfai - cfa[row * W + (col + 2)])
                                  + std::fabs(cfa[row * W + (col + 1)] - cfa[row * W + (col + 3)])
                                  + std::fabs(cfa[row * W + (col + 2)] - cfa[row * W + (col + 4)]));

            // Ratio-corrected estimations.
            const float lfpi = lpf[row * W + col];
            const float N_Est = cfa[(row - 1) * W + col] * (2.0f * lfpi) / (kEps + lfpi + lpf[(row - 2) * W + col]);
            const float S_Est = cfa[(row + 1) * W + col] * (2.0f * lfpi) / (kEps + lfpi + lpf[(row + 2) * W + col]);
            const float W_Est = cfa[row * W + (col - 1)] * (2.0f * lfpi) / (kEps + lfpi + lpf[row * W + (col - 2)]);
            const float E_Est = cfa[row * W + (col + 1)] * (2.0f * lfpi) / (kEps + lfpi + lpf[row * W + (col + 2)]);

            const float V_Est = (S_Grad * N_Est + N_Grad * S_Est) / (N_Grad + S_Grad);
            const float H_Est = (W_Grad * E_Est + E_Grad * W_Est) / (E_Grad + W_Grad);

            const float d = clipf(VH_Disc);
            rgb1.set(row * W + col, d * (H_Est - V_Est) + V_Est);
        }
    }
}

// ============================================================
// Border pre-fill of intermediate RGB planes
// ============================================================
// The reference skips the 4-px border in ALL RCD stages (populate only
// fills the native CFA color; step3/4_2/4_3 only run on [4, h-4)). This
// leaves rgb0/rgb1/rgb2 at zero in border non-native positions. When the
// first interior row (row=4) of step4_2/step4_3 reads diagonals/cardinals
// at row=3 (border), it gets 0 instead of a real R/G/B estimate, which
// contaminates a 4-px ring just inside the bilinear border.
//
// darktable avoids this by running every OpenCL kernel on the full image
// with CLAMP_TO_EDGE sampling. We mirror that effect by pre-filling the
// border ring of the intermediate planes with a simple 3x3 bilinear
// estimate (same kernel as borderInterpolate, but targeting the separate
// R/G/B planes). This is applied after step3 (green interp) so step4_2 /
// step4_3 read plausible values at border positions.
static void fillBorderIntermediates(const RawMosaic& m,
                                    DemosaicPlane& rgb0,
                                    DemosaicPlane& rgb1,
                                    DemosaicPlane& rgb2) {
    const int W = m.width;
    const int H = m.height;
    const float* bayer = m.data.data();

    // Per-pixel 3x3 bilinear estimate into rgb0/1/2 at a border position.
    auto body = [&](int row, int col) {
        float sums[3] = {0.0f, 0.0f, 0.0f};
        float counts[3] = {0.0f, 0.0f, 0.0f};
        for (int dy = -1; dy <= 1; ++dy) {
            const int y = row + dy;
            if (y < 0 || y >= H) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                const int x = col + dx;
                if (x < 0 || x >= W) continue;
                int c = cfaColor(m, y, x);
                if (c == 3) c = 1;  // G2 -> G
                sums[c] += std::max(0.0f, bayer[static_cast<size_t>(y) * W + x]);
                counts[c] += 1.0f;
            }
        }

        const size_t i = static_cast<size_t>(row) * W + col;
        if (counts[0] > 0.0f) rgb0.set(i, sums[0] / counts[0]);
        if (counts[1] > 0.0f) rgb1.set(i, sums[1] / counts[1]);
        if (counts[2] > 0.0f) rgb2.set(i, sums[2] / counts[2]);
    };

    // Iterate ONLY the border ring (avoid ~H*W interior no-ops): full-width
    // top/bottom bands plus the two side strips on interior rows. Identical
    // pixel set to the previous full-HxW loop with the interior `continue`.
    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int row = 0; row < H; ++row) {
        if (row < kBorder || row >= H - kBorder) {
            for (int col = 0; col < W; ++col) body(row, col);
        } else {
            for (int col = 0; col < kBorder; ++col) body(row, col);
            for (int col = W - kBorder; col < W; ++col) body(row, col);
        }
    }
}

// ============================================================
// Step 4.0: P/Q diagonal high-pass (demosaic.py:189-206)
// ============================================================
// fsquare of 9-tap diagonal high-pass. Range (3, h-3) x (3, w-3).
static void rcdStep4_0(const DemosaicPlane& cfa,
                       DemosaicPlane& p_diff,
                       DemosaicPlane& q_diff,
                       int W, int H) {
    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int row = 3; row < H - 3; ++row) {
        for (int col = 3; col < W - 3; ++col) {
            const size_t i = static_cast<size_t>(row) * W + col;
            p_diff.set(i, fsquare(
                cfa[(row - 3) * W + (col - 3)] - cfa[(row - 1) * W + (col - 1)]
                - cfa[(row + 1) * W + (col + 1)] + cfa[(row + 3) * W + (col + 3)]
                - 3.0f * (cfa[(row - 2) * W + (col - 2)] + cfa[(row + 2) * W + (col + 2)])
                + 6.0f * cfa[i]));
            q_diff.set(i, fsquare(
                cfa[(row - 3) * W + (col + 3)] - cfa[(row - 1) * W + (col + 1)]
                - cfa[(row + 1) * W + (col - 1)] + cfa[(row + 3) * W + (col - 3)]
                - 3.0f * (cfa[(row - 2) * W + (col + 2)] + cfa[(row + 2) * W + (col - 2)])
                + 6.0f * cfa[i]));
        }
    }
}

// ============================================================
// Step 4.1: P/Q discrimination at R/B (demosaic.py:213-228)
// ============================================================
static void rcdStep4_1(const RawMosaic& m,
                       const DemosaicPlane& p_diff,
                       const DemosaicPlane& q_diff,
                       DemosaicPlane& PQ_dir,
                       int W, int H) {
    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int row = kBorder; row < H - kBorder; ++row) {
        for (int col = kBorder; col < W - kBorder; ++col) {
            if (isGreen(cfaColor(m, row, col))) continue;
            const float P_Stat = std::max(kEpsSq,
                p_diff[(row - 1) * W + (col - 1)] + p_diff[row * W + col] + p_diff[(row + 1) * W + (col + 1)]);
            const float Q_Stat = std::max(kEpsSq,
                q_diff[(row - 1) * W + (col + 1)] + q_diff[row * W + col] + q_diff[(row + 1) * W + (col - 1)]);
            PQ_dir.set(row * W + col, P_Stat / (P_Stat + Q_Stat));
        }
    }
}

// step4_2 helper: interpolate one color channel at an R/B position
// (the opposite color). Port of _rcd_step4_2_color (demosaic.py:235-266).
static inline float rcdStep4_2_color(const DemosaicPlane& rgbc, const DemosaicPlane& rgb1,
                                      const DemosaicPlane& PQ_dir,
                                      int row, int col, int W) {
    const float PQ_Central = PQ_dir[row * W + col];
    const float PQ_Neigh = 0.25f * (PQ_dir[(row - 1) * W + (col - 1)] + PQ_dir[(row - 1) * W + (col + 1)]
                                    + PQ_dir[(row + 1) * W + (col - 1)] + PQ_dir[(row + 1) * W + (col + 1)]);
    const float PQ_Disc = (std::fabs(0.5f - PQ_Central) < std::fabs(0.5f - PQ_Neigh))
                          ? PQ_Neigh : PQ_Central;

    const float g = rgb1[row * W + col];
    const float NW = rgbc[(row - 1) * W + (col - 1)];
    const float NE = rgbc[(row - 1) * W + (col + 1)];
    const float SW = rgbc[(row + 1) * W + (col - 1)];
    const float SE = rgbc[(row + 1) * W + (col + 1)];

    const float NW_Grad = kEps + std::fabs(NW - SE) + std::fabs(NW - rgbc[(row - 3) * W + (col - 3)]) + std::fabs(g - rgb1[(row - 2) * W + (col - 2)]);
    const float NE_Grad = kEps + std::fabs(NE - SW) + std::fabs(NE - rgbc[(row - 3) * W + (col + 3)]) + std::fabs(g - rgb1[(row - 2) * W + (col + 2)]);
    const float SW_Grad = kEps + std::fabs(NE - SW) + std::fabs(SW - rgbc[(row + 3) * W + (col - 3)]) + std::fabs(g - rgb1[(row + 2) * W + (col - 2)]);
    const float SE_Grad = kEps + std::fabs(NW - SE) + std::fabs(SE - rgbc[(row + 3) * W + (col + 3)]) + std::fabs(g - rgb1[(row + 2) * W + (col + 2)]);

    const float NW_Est = NW - rgb1[(row - 1) * W + (col - 1)];
    const float NE_Est = NE - rgb1[(row - 1) * W + (col + 1)];
    const float SW_Est = SW - rgb1[(row + 1) * W + (col - 1)];
    const float SE_Est = SE - rgb1[(row + 1) * W + (col + 1)];

    const float P_Est = (NW_Grad * SE_Est + SE_Grad * NW_Est) / (NW_Grad + SE_Grad);
    const float Q_Est = (NE_Grad * SW_Est + SW_Grad * NE_Est) / (NE_Grad + SW_Grad);

    const float d = clipf(PQ_Disc);
    return g + d * (Q_Est - P_Est) + P_Est;
}

// ============================================================
// Step 4.2: R/B at R/B positions — opposite color (demosaic.py:269-286)
// ============================================================
// color = 2 - fc: fc=0/R -> interp B into rgb2; fc=2/B -> interp R into rgb0.
static void rcdStep4_2(const RawMosaic& m,
                       DemosaicPlane& rgb0,
                       DemosaicPlane& rgb1,
                       DemosaicPlane& rgb2,
                       const DemosaicPlane& PQ_dir,
                       int W, int H) {
    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int row = kBorder; row < H - kBorder; ++row) {
        for (int col = kBorder; col < W - kBorder; ++col) {
            const int fc = cfaColor(m, row, col);
            if (isGreen(fc)) continue;
            const int color = 2 - fc;  // opposite color
            if (color == 0) {
                rgb0.set(row * W + col, rcdStep4_2_color(rgb0, rgb1, PQ_dir, row, col, W));
            } else {  // color == 2
                rgb2.set(row * W + col, rcdStep4_2_color(rgb2, rgb1, PQ_dir, row, col, W));
            }
        }
    }
}

// step4_3 helper: interpolate one color channel at a green position.
// Port of _rcd_step4_3_color (demosaic.py:293-321). NOTE: the reference
// signature also takes the rgb1 plane, but all needed rgb1 values are
// pre-extracted (rgbi1, rgb1mw/pw/m1/p1) and passed as scalars — the rgb1
// array itself is never indexed in the body, so it is omitted here.
static inline float rcdStep4_3_color(const DemosaicPlane& rgbc,
                                      int row, int col, int W,
                                      float d, float rgbi1,
                                      float N1, float S1, float W1, float E1,
                                      float rgb1mw, float rgb1pw,
                                      float rgb1m1, float rgb1p1) {
    const float cn = rgbc[(row - 1) * W + col];
    const float cs = rgbc[(row + 1) * W + col];
    const float cw = rgbc[row * W + (col - 1)];
    const float ce = rgbc[row * W + (col + 1)];

    const float SNabs = std::fabs(cn - cs);
    const float EWabs = std::fabs(cw - ce);

    const float N_Grad = N1 + SNabs + std::fabs(cn - rgbc[(row - 3) * W + col]);
    const float S_Grad = S1 + SNabs + std::fabs(cs - rgbc[(row + 3) * W + col]);
    const float W_Grad = W1 + EWabs + std::fabs(cw - rgbc[row * W + (col - 3)]);
    const float E_Grad = E1 + EWabs + std::fabs(ce - rgbc[row * W + (col + 3)]);

    const float N_Est = cn - rgb1mw;
    const float S_Est = cs - rgb1pw;
    const float W_Est = cw - rgb1m1;
    const float E_Est = ce - rgb1p1;

    const float V_Est = (N_Grad * S_Est + S_Grad * N_Est) / (N_Grad + S_Grad);
    const float H_Est = (E_Grad * W_Est + W_Grad * E_Est) / (E_Grad + W_Grad);

    return rgbi1 + d * (H_Est - V_Est) + V_Est;
}

// ============================================================
// Step 4.3: R/B at green positions (demosaic.py:324-358)
// ============================================================
// At FC==1 positions, interpolate BOTH rgb0 (R) and rgb2 (B) using the
// green-plane VH discrimination + cardinal gradients.
static void rcdStep4_3(const RawMosaic& m,
                       DemosaicPlane& rgb0,
                       DemosaicPlane& rgb1,
                       DemosaicPlane& rgb2,
                       const DemosaicPlane& VH_dir,
                       int W, int H) {
    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int row = kBorder; row < H - kBorder; ++row) {
        for (int col = kBorder; col < W - kBorder; ++col) {
            if (!isGreen(cfaColor(m, row, col))) continue;

            const float VH_Central = VH_dir[row * W + col];
            const float VH_Neigh = 0.25f * (VH_dir[(row - 1) * W + (col - 1)] + VH_dir[(row - 1) * W + (col + 1)]
                                            + VH_dir[(row + 1) * W + (col - 1)] + VH_dir[(row + 1) * W + (col + 1)]);
            const float VH_Disc = (std::fabs(0.5f - VH_Central) < std::fabs(0.5f - VH_Neigh))
                                  ? VH_Neigh : VH_Central;

            const float rgbi1 = rgb1[row * W + col];
            const float N1 = kEps + std::fabs(rgbi1 - rgb1[(row - 2) * W + col]);
            const float S1 = kEps + std::fabs(rgbi1 - rgb1[(row + 2) * W + col]);
            const float W1 = kEps + std::fabs(rgbi1 - rgb1[row * W + (col - 2)]);
            const float E1 = kEps + std::fabs(rgbi1 - rgb1[row * W + (col + 2)]);

            const float rgb1mw = rgb1[(row - 1) * W + col];
            const float rgb1pw = rgb1[(row + 1) * W + col];
            const float rgb1m1 = rgb1[row * W + (col - 1)];
            const float rgb1p1 = rgb1[row * W + (col + 1)];

            const float d = clipf(VH_Disc);

            rgb0.set(row * W + col, rcdStep4_3_color(rgb0, row, col, W,
                                                      d, rgbi1, N1, S1, W1, E1,
                                                      rgb1mw, rgb1pw, rgb1m1, rgb1p1));
            rgb2.set(row * W + col, rcdStep4_3_color(rgb2, row, col, W,
                                                      d, rgbi1, N1, S1, W1, E1,
                                                      rgb1mw, rgb1pw, rgb1m1, rgb1p1));
        }
    }
}

// ============================================================
// Step 5: Write output (demosaic.py:365-377)
// ============================================================
// Interior only (skip kBorder px); out = max(., 0) per channel.
static void rcdWriteOutput(const DemosaicPlane& rgb0,
                           const DemosaicPlane& rgb1,
                           const DemosaicPlane& rgb2,
                           ImageBuffer& out) {
    const int W = out.width;
    const int H = out.height;

    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int row = kBorder; row < H - kBorder; ++row) {
        for (int col = kBorder; col < W - kBorder; ++col) {
            const size_t i = static_cast<size_t>(row) * W + col;
            float* px = out.pixel(row, col);
            px[0] = std::max(0.0f, rgb0[i]);
            px[1] = std::max(0.0f, rgb1[i]);
            px[2] = std::max(0.0f, rgb2[i]);
        }
    }
}

// ============================================================
// Border: bilinear fallback (demosaic.py:384-413)
// ============================================================
// 3x3 bilinear by FC for the kBorder-px border ring. Reads from the original
// mosaic (m.data) with max(0, .) applied — same as reference which re-uploads
// bayer. CFA color 3 (G2) is collapsed to 1 (G) for the sums[] index.
static void borderInterpolate(const RawMosaic& m, ImageBuffer& out) {
    const int W = m.width;
    const int H = m.height;
    const float* bayer = m.data.data();

    // Per-pixel 3x3 bilinear estimate into the output at a border position.
    auto body = [&](int row, int col) {
        float sums[3] = {0.0f, 0.0f, 0.0f};
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

    // Iterate ONLY the border ring (avoid ~H*W interior no-ops): full-width
    // top/bottom bands plus the two side strips on interior rows. Identical
    // pixel set to the previous full-HxW loop with the interior `continue`.
    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int row = 0; row < H; ++row) {
        if (row < kBorder || row >= H - kBorder) {
            for (int col = 0; col < W; ++col) body(row, col);
        } else {
            for (int col = 0; col < kBorder; ++col) body(row, col);
            for (int col = W - kBorder; col < W; ++col) body(row, col);
        }
    }
}

// ============================================================
// Public API: rcdDemosaic (demosaic.py:459-540)
// ============================================================
ImageBuffer rcdDemosaic(const RawMosaic& m) {
    const int W = m.width;
    const int H = m.height;
    const size_t N = static_cast<size_t>(H) * W;

    // Output (3-channel). Initialized to 0 — border pixels with no neighbor
    // of a given color stay 0.
    ImageBuffer out(W, H, 3);

    // Persistent intermediates (freed in stages below to match the reference
    // `del` ordering and keep peak memory at ~7 x H x W x 4 bytes; F16 storage
    // on ARM64 halves this).
    DemosaicPlane cfa(N), rgb0(N), rgb1(N), rgb2(N), VH_dir(N);

    // Steps 0-1: populate + VH discrimination.
    rcdPopulate(m, cfa, rgb0, rgb1, rgb2);
    rcdStep1(cfa, VH_dir, W, H);

    // Step 2-3: low-pass + green interp. lpf freed at scope exit.
    {
        DemosaicPlane lpf(N);
        rcdStep2(m, cfa, lpf, W, H);
        rcdStep3(m, cfa, lpf, rgb1, VH_dir, W, H);
    }

    // Pre-fill the border ring of the intermediate RGB planes with bilinear
    // estimates so step4_2/step4_3 at the first interior row (row=4) read
    // plausible values instead of zero-contaminated border samples. (The
    // reference skips border in all stages; darktable avoids this by
    // processing the full image with CLAMP_TO_EDGE.)
    fillBorderIntermediates(m, rgb0, rgb1, rgb2);

    // Step 4.0: P/Q diagonal high-pass (uses cfa, then cfa is no longer needed).
    DemosaicPlane p_diff(N), q_diff(N);
    rcdStep4_0(cfa, p_diff, q_diff, W, H);
    freeVector(cfa);  // mirror `del cfa` (demosaic.py:508)

    // Step 4.1: P/Q discrimination (uses p_diff, q_diff; FC via cfaColor(m,..)).
    DemosaicPlane PQ_dir(N);
    rcdStep4_1(m, p_diff, q_diff, PQ_dir, W, H);
    freeVector(p_diff);  // mirror `del p_diff, q_diff`
    freeVector(q_diff);

    // Step 4.2-4.3: R/B interpolation (uses PQ_dir, then VH_dir).
    rcdStep4_2(m, rgb0, rgb1, rgb2, PQ_dir, W, H);
    freeVector(PQ_dir);  // mirror `del PQ_dir`
    rcdStep4_3(m, rgb0, rgb1, rgb2, VH_dir, W, H);
    freeVector(VH_dir);  // mirror `del VH_dir`

    // Step 5: write interior (skip 4-px border).
    rcdWriteOutput(rgb0, rgb1, rgb2, out);

    // Border bilinear fallback (reads original mosaic; rgb0/1/2 no longer needed).
    freeVector(rgb0);
    freeVector(rgb1);
    freeVector(rgb2);
    borderInterpolate(m, out);

    return out;
}

} // namespace rawalchemy
