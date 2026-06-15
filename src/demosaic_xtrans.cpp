/**
 * @file demosaic_xtrans.cpp
 * @brief Phase 4 — Markesteijn 1-pass X-Trans demosaicing.
 *
 * Direct C++/OpenMP port of the Python Taichi reference
 * `raw_alchemy.xtrans_demosaic` (xtrans_demosaic.py:1-914), itself a Taichi
 * GPU port of darktable's `xtrans.c` (algorithm by Frank Markesteijn).
 *
 * Each Taichi `@ti.kernel` over `ti.ndrange((pad, H-pad), (pad, W-pad))`
 * becomes a C++ `#pragma omp parallel for collapse(2)` over the same
 * interior range. Per-kernel pad values: 3,3,6,6,8,9,12.
 *
 * CFA lookups use a local `FCxt(xt, r, c)` helper (negative-index safe,
 * identical to `cfaColor`'s X-Trans branch but taking the 6x6 pattern
 * directly for clarity, matching the reference's `FCxt(row, col, xtrans)`).
 *
 * Buffer layout: 4 direction buffers, each interleaved (H*W*3) float,
 * indexed as `(row*W + col)*3 + ch` — matches the reference's (H,W,3)
 * Taichi ndarray layout.
 *
 * Constants (xtrans_demosaic.py:24-25): BORDER=12, NDIR=4.
 *
 * Memory: peaks at ~19 x H x W x 4 bytes (4 dir buffers x 3 ch + 4 drv
 * planes + 1 out). Intermediates are freed in the same order as the
 * reference's `del` statements (xtrans_demosaic.py:860, 869, 880, 897-898).
 */

#include "demosaic.h"

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef RA_USE_OPENMP
#include <omp.h>
#endif

namespace rawalchemy {

// Reference constants (xtrans_demosaic.py:24-25).
static constexpr int kBORDER = 12;
static constexpr int kNDIR   = 4;

// ---- Helpers (xtrans_demosaic.py:32-57) ----

static inline float clampf(float x, float lo, float hi) {
    return std::min(std::max(x, lo), hi);
}
static inline float sqrf(float x) { return x * x; }

// Free a vector's allocated memory immediately (mirrors Python `del`).
template <typename T>
static inline void freeVector(std::vector<T>& v) {
    std::vector<T>().swap(v);
}

// X-Trans CFA color at (row, col). Negative-index safe — identical to
// `cfaColor`'s X-Trans branch but takes the 6x6 pattern directly, matching
// the reference's `FCxt(row, col, xtrans)`. Returns 0=R, 1=G, 2=B.
static inline int FCxt(const char xt[6][6], int row, int col) {
    return static_cast<int>(xt[((row % 6) + 6) % 6][((col % 6) + 6) % 6]);
}

// allhex flat index helper (xtrans_demosaic.py:49-57): [3][3][8] flattened.
// ah(arr, r3, c3, e) = arr[r3*24 + c3*8 + e].
static inline int ah(const int* arr, int r3, int c3, int e) {
    return arr[r3 * 24 + c3 * 8 + e];
}

// 3D interleaved index: (row, col, ch) -> linear offset.
static inline size_t i3(int row, int col, int ch, int W) {
    return (static_cast<size_t>(row) * W + col) * 3 + ch;
}

// ============================================================
// allhex precompute (xtrans_demosaic.py:64-107)
// ============================================================
// Builds the hexagonal neighbor lookup [3][3][8] (flattened to 72 ints for
// dr and dc) and finds the solitary-green position (sgrow, sgcol).
//
// Ported VERBATIM. The `idx = c ^ ((g*2) & d_idx)` xor logic (line 102) is
// the highest-risk bit — it permutes which entry slot each (c) lands in
// depending on the direction parity d_idx and the pixel color g. Do NOT
// simplify.
struct AllHex {
    int dr[72];
    int dc[72];
    int sgrow = 0;
    int sgcol = 0;
};

static AllHex buildAllhex(const char xtrans[6][6]) {
    // orth[12] and patt[2][16] copied exactly from xtrans_demosaic.py:73-77.
    static const int orth[12] = {1, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 1};
    static const int patt[2][16] = {
        {0, 1, 0, -1, 2, 0, -1, 0, 1, 1, 1, -1, 0, 0, 0, 0},
        {0, 1, 0, -2, 1, 0, -2, 0, 1, 1, -2, -2, 1, -1, -1, 1},
    };

    AllHex out;
    for (int i = 0; i < 72; ++i) { out.dr[i] = 0; out.dc[i] = 0; }
    out.sgrow = 0;
    out.sgcol = 0;

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            int ng = 0;
            // xtrans_demosaic.py:90 — `for d_idx in range(0, 10, 2)` -> 0,2,4,6,8.
            for (int d_idx = 0; d_idx <= 8; d_idx += 2) {
                const int g = (FCxt(xtrans, row, col) == 1) ? 1 : 0;
                if (FCxt(xtrans, row + orth[d_idx], col + orth[d_idx + 2]) == 1) {
                    ng = 0;
                } else {
                    ng += 1;
                }
                if (ng == 4) {
                    out.sgrow = row;
                    out.sgcol = col;
                }
                if (ng == g + 1) {
                    for (int c = 0; c < 8; ++c) {
                        const int v = orth[d_idx]     * patt[g][c * 2]
                                    + orth[d_idx + 1] * patt[g][c * 2 + 1];
                        const int h = orth[d_idx + 2] * patt[g][c * 2]
                                    + orth[d_idx + 3] * patt[g][c * 2 + 1];
                        // VERBATIM (xtrans_demosaic.py:102): idx = c ^ (g*2 & d_idx).
                        // C++ precedence: * > & > ^, so this equals c ^ ((g*2) & d_idx).
                        const int idx = c ^ (g * 2 & d_idx);
                        const int flat = row * 24 + col * 8 + idx;
                        out.dr[flat] = v;
                        out.dc[flat] = h;
                    }
                }
            }
        }
    }
    return out;
}

// ============================================================
// Kernel 1: Populate (xtrans_demosaic.py:114-127) + copy (130-136)
// ============================================================
// rgb[row,col,*] = 0; rgb[row,col,FCxt] = max(raw, 0). Then copy d0 -> d1/d2/d3.
static void xtmPopulate(const RawMosaic& m,
                        std::vector<float>& rgb_d0,
                        std::vector<float>& rgb_d1,
                        std::vector<float>& rgb_d2,
                        std::vector<float>& rgb_d3) {
    const int W = m.width;
    const int H = m.height;
    const float* raw = m.data.data();
    float* d0 = rgb_d0.data();

    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            const float val = std::max(0.0f, raw[static_cast<size_t>(row) * W + col]);
            const int f = FCxt(m.xtrans, row, col);
            float* px = d0 + i3(row, col, 0, W);
            px[0] = 0.0f;
            px[1] = 0.0f;
            px[2] = 0.0f;
            px[f] = val;
        }
    }

    // Copy d0 -> d1, d2, d3 (xtrans_demosaic.py:131-136).
    const size_t n3 = static_cast<size_t>(H) * W * 3;
    std::copy(d0, d0 + n3, rgb_d1.data());
    std::copy(d0, d0 + n3, rgb_d2.data());
    std::copy(d0, d0 + n3, rgb_d3.data());
}

// ============================================================
// BORDER PRE-FILL (stage-review Q2 — MANDATORY)
// ============================================================
// fillBorderIntermediates12: fill the full BORDER=12 ring of all 4 direction
// buffers, all 3 channels, with a 3x3 bilinear estimate from the raw mosaic
// (by FCxt color).
//
// WHY: the reference (and our port) skips the kBORDER-px border in ALL
// interpolation kernels (green_interp starts at pad=3, gminmax at pad=3,
// etc.). populate() only fills the native CFA color channel; the other two
// channels stay at 0 in the border. When an interior kernel reads into the
// border ring (e.g. green_interp at row=3 reads rgb_d0 at row=0 via the
// 3*hex[4|5] reach of the non-green hex stencil), it would read 0 in the
// non-native channels, contaminating a kBORDER-px ring just inside the
// bilinear border (cascade contamination).
//
// darktable avoids this by running every OpenCL kernel on the full image
// with CLAMP_TO_EDGE sampling. We mirror that effect by pre-filling the
// border ring with a plausible 3x3 bilinear estimate (darktable
// CLAMP_TO_EDGE equivalence). This is the same fix applied in Phase 3 RCD
// (fillBorderIntermediates), extended here to the wider kBORDER=12 ring and
// all 4 direction buffers.
//
// NOTE: max stencil reach of the interior kernels into the border is ±3
// (green_interp's 3*hex[4|5] for non-green pixels); the ±6 mentioned in the
// stage-review refers to the 6-element hex neighborhood sum, not a ±6 pixel
// offset. Either way, the kBORDER=12 ring comfortably covers all reaches.
//
// Called AFTER Kernel 1 (populate + copy) and BEFORE Kernel 2 (gminmax).
static void fillBorderIntermediates12(const RawMosaic& m,
                                       std::vector<float>& rgb_d0,
                                       std::vector<float>& rgb_d1,
                                       std::vector<float>& rgb_d2,
                                       std::vector<float>& rgb_d3) {
    const int W = m.width;
    const int H = m.height;
    const float* raw = m.data.data();

    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            // Only process the kBORDER-px border ring.
            if (row >= kBORDER && row < H - kBORDER && col >= kBORDER && col < W - kBORDER)
                continue;

            float sums[3] = {0.0f, 0.0f, 0.0f};
            float counts[3] = {0.0f, 0.0f, 0.0f};
            for (int dy = -1; dy <= 1; ++dy) {
                const int y = row + dy;
                if (y < 0 || y >= H) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int x = col + dx;
                    if (x < 0 || x >= W) continue;
                    const int c = FCxt(m.xtrans, y, x);
                    sums[c]   += std::max(0.0f, raw[static_cast<size_t>(y) * W + x]);
                    counts[c] += 1.0f;
                }
            }

            float est[3] = {0.0f, 0.0f, 0.0f};
            for (int c = 0; c < 3; ++c) {
                if (counts[c] > 0.0f) est[c] = sums[c] / counts[c];
            }
            // Write all 3 channels into all 4 direction buffers.
            for (int c = 0; c < 3; ++c) {
                rgb_d0[i3(row, col, c, W)] = est[c];
                rgb_d1[i3(row, col, c, W)] = est[c];
                rgb_d2[i3(row, col, c, W)] = est[c];
                rgb_d3[i3(row, col, c, W)] = est[c];
            }
        }
    }
}

// ============================================================
// Kernel 2: gmin / gmax (xtrans_demosaic.py:143-168)
// ============================================================
// At non-green pixels: gmin/gmax over the 6 hex neighbors (entries 0-5),
// reading rgb_d0[...][1] (green channel).
static void xtmGminmax(const std::vector<float>& rgb_d0,
                       std::vector<float>& gmin_buf,
                       std::vector<float>& gmax_buf,
                       const AllHex& hex,
                       const RawMosaic& m) {
    const int W = m.width;
    const int H = m.height;
    const float* d0 = rgb_d0.data();
    const int pad = 3;

    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int row = pad; row < H - pad; ++row) {
        for (int col = pad; col < W - pad; ++col) {
            if (FCxt(m.xtrans, row, col) == 1) continue;
            const int r3 = row % 3;
            const int c3 = col % 3;
            float mn = 1e30f;
            float mx = 0.0f;
            for (int k = 0; k < 6; ++k) {
                const int dr = ah(hex.dr, r3, c3, k);
                const int dc = ah(hex.dc, r3, c3, k);
                const float val = d0[i3(row + dr, col + dc, 1, W)];
                mn = std::min(mn, val);
                mx = std::max(mx, val);
            }
            const size_t i = static_cast<size_t>(row) * W + col;
            gmin_buf[i] = mn;
            gmax_buf[i] = mx;
        }
    }
}

// ============================================================
// _green_interp_one (xtrans_demosaic.py:175-221)
// ============================================================
// One directional green estimate (color_idx 0..3). Distinct formulas using
// allhex 0-5; color_idx 2,3 use 3*hex[4+c].
static inline float greenInterpOne(const float* d0,
                                   const AllHex& hex,
                                   const RawMosaic& m,
                                   int row, int col,
                                   int color_idx) {
    const int W = m.width;
    const int f = FCxt(m.xtrans, row, col);
    const int r3 = row % 3;
    const int c3 = col % 3;

    if (color_idx == 0) {
        // xtrans_demosaic.py:192-199
        const int dr0 = ah(hex.dr, r3, c3, 0);
        const int dc0 = ah(hex.dc, r3, c3, 0);
        const int dr1 = ah(hex.dr, r3, c3, 1);
        const int dc1 = ah(hex.dc, r3, c3, 1);
        return (0.6796875f  * (d0[i3(row + dr1, col + dc1, 1, W)] + d0[i3(row + dr0, col + dc0, 1, W)])
              - 0.1796875f  * (d0[i3(row + 2 * dr1, col + 2 * dc1, 1, W)]
                             + d0[i3(row + 2 * dr0, col + 2 * dc0, 1, W)]));
    } else if (color_idx == 1) {
        // xtrans_demosaic.py:202-209
        const int dr2 = ah(hex.dr, r3, c3, 2);
        const int dc2 = ah(hex.dc, r3, c3, 2);
        const int dr3 = ah(hex.dr, r3, c3, 3);
        const int dc3 = ah(hex.dc, r3, c3, 3);
        return (0.87109375f * d0[i3(row + dr3, col + dc3, 1, W)]
              + 0.13f       * d0[i3(row + dr2, col + dc2, 1, W)]
              + 0.359375f   * (d0[i3(row, col, f, W)] - d0[i3(row - dr2, col - dc2, f, W)]));
    } else {
        // xtrans_demosaic.py:212-220 — k = 4 + (color_idx - 2) -> hex entry 4 or 5.
        const int k = 4 + (color_idx - 2);
        const int drk = ah(hex.dr, r3, c3, k);
        const int dck = ah(hex.dc, r3, c3, k);
        return (0.640625f   * d0[i3(row + drk, col + dck, 1, W)]
              + 0.359375f   * d0[i3(row - 2 * drk, col - 2 * dck, 1, W)]
              + 0.12890625f * (2.0f * d0[i3(row, col, f, W)]
                             - d0[i3(row + 3 * drk, col + 3 * dck, f, W)]
                             - d0[i3(row - 3 * drk, col - 3 * dck, f, W)]));
    }
}

// ============================================================
// Kernel 3: Green interpolation (xtrans_demosaic.py:224-273)
// ============================================================
// 4 directional green estimates clamped to [gmin, gmax]; assigned to the 4
// direction buffers via flip = ((row-sgrow)%3==0): rgb_dk[..][1] = vals[k^flip].
static void xtmGreenInterp(std::vector<float>& rgb_d0,
                           std::vector<float>& rgb_d1,
                           std::vector<float>& rgb_d2,
                           std::vector<float>& rgb_d3,
                           const std::vector<float>& gmin_buf,
                           const std::vector<float>& gmax_buf,
                           const AllHex& hex,
                           const RawMosaic& m) {
    const int W = m.width;
    const int H = m.height;
    const float* d0 = rgb_d0.data();
    const float* gmin = gmin_buf.data();
    const float* gmax = gmax_buf.data();
    float* d0w = rgb_d0.data();
    float* d1w = rgb_d1.data();
    float* d2w = rgb_d2.data();
    float* d3w = rgb_d3.data();
    const int sgrow = hex.sgrow;
    const int pad = 3;

    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int row = pad; row < H - pad; ++row) {
        for (int col = pad; col < W - pad; ++col) {
            if (FCxt(m.xtrans, row, col) == 1) continue;

            const size_t i = static_cast<size_t>(row) * W + col;
            const float mn = gmin[i];
            const float mx = gmax[i];

            const float c0 = greenInterpOne(d0, hex, m, row, col, 0);
            const float c1 = greenInterpOne(d0, hex, m, row, col, 1);
            const float c2 = greenInterpOne(d0, hex, m, row, col, 2);
            const float c3 = greenInterpOne(d0, hex, m, row, col, 3);

            const float vals[4] = {
                clampf(c0, mn, mx),
                clampf(c1, mn, mx),
                clampf(c2, mn, mx),
                clampf(c3, mn, mx),
            };

            // VERBATIM (xtrans_demosaic.py:252): flip = 1 if (row-sgrow)%3==0 else 0.
            // (row-sgrow)%3==0 holds for multiples of 3 in both Python and C++.
            const int flip = ((row - sgrow) % 3 == 0) ? 1 : 0;

            // VERBATIM (xtrans_demosaic.py:270-273): rgb_dk[..][1] = vals[k ^ flip].
            d0w[i3(row, col, 1, W)] = vals[0 ^ flip];
            d1w[i3(row, col, 1, W)] = vals[1 ^ flip];
            d2w[i3(row, col, 1, W)] = vals[2 ^ flip];
            d3w[i3(row, col, 1, W)] = vals[3 ^ flip];
        }
    }
}

// ============================================================
// _rb_solitary_green_for_dir (xtrans_demosaic.py:280-379)
// ============================================================
// R/B at a solitary green pixel for one direction buffer.
// mode: 0=horiz only, 1=vert only, 2=best of both (by diff).
static inline void rbSolitaryGreenForDir(const float* rgb,
                                         int row, int col, int W,
                                         int h0, int mode,
                                         float& out_r, float& out_b) {
    const float g_pixel = rgb[i3(row, col, 1, W)];

    const int h_first  = h0;
    const int h_second = h0 ^ 2;

    // Horizontal estimates (xtrans_demosaic.py:302-306).
    const float g_h1 = 2.0f * g_pixel
                     - rgb[i3(row, col + 1, 1, W)] - rgb[i3(row, col - 1, 1, W)];
    const float color_h_c0 = g_h1
                           + rgb[i3(row, col + 1, h_first, W)] + rgb[i3(row, col - 1, h_first, W)];

    const float g_h2 = 2.0f * g_pixel
                     - rgb[i3(row, col + 2, 1, W)] - rgb[i3(row, col - 2, 1, W)];
    const float color_h_c1 = g_h2
                           + rgb[i3(row, col + 2, h_second, W)] + rgb[i3(row, col - 2, h_second, W)];

    // Vertical estimates (xtrans_demosaic.py:310-317).
    const int v_first  = h0 ^ 2;
    const int v_second = h0;

    const float g_v1 = 2.0f * g_pixel
                     - rgb[i3(row + 1, col, 1, W)] - rgb[i3(row - 1, col, 1, W)];
    const float color_v_c0 = g_v1
                           + rgb[i3(row + 1, col, v_first, W)] + rgb[i3(row - 1, col, v_first, W)];

    const float g_v2 = 2.0f * g_pixel
                     - rgb[i3(row + 2, col, 1, W)] - rgb[i3(row - 2, col, 1, W)];
    const float color_v_c1 = g_v2
                           + rgb[i3(row + 2, col, v_second, W)] + rgb[i3(row - 2, col, v_second, W)];

    // Assemble horiz R/B (xtrans_demosaic.py:325-332).
    float horiz_r, horiz_b;
    if (h_first == 0) {
        horiz_r = color_h_c0;
        horiz_b = color_h_c1;
    } else {
        horiz_b = color_h_c0;
        horiz_r = color_h_c1;
    }

    // Assemble vert R/B (xtrans_demosaic.py:335-342).
    float vert_r, vert_b;
    if (v_first == 0) {
        vert_r = color_v_c0;
        vert_b = color_v_c1;
    } else {
        vert_b = color_v_c0;
        vert_r = color_v_c1;
    }

    if (mode == 0) {
        out_r = horiz_r / 2.0f;
        out_b = horiz_b / 2.0f;
    } else if (mode == 1) {
        out_r = vert_r / 2.0f;
        out_b = vert_b / 2.0f;
    } else {
        // mode == 2: pick best of horiz/vert by summed squared diff.
        float h_diff = 0.0f;
        float v_diff = 0.0f;

        // Horiz diff c=0 (xtrans_demosaic.py:359-360).
        h_diff += sqrf(rgb[i3(row, col + 1, 1, W)] - rgb[i3(row, col - 1, 1, W)]
                     - rgb[i3(row, col + 1, h_first, W)] + rgb[i3(row, col - 1, h_first, W)])
                + sqrf(g_h1);
        // Horiz diff c=1 (xtrans_demosaic.py:362-363).
        h_diff += sqrf(rgb[i3(row, col + 2, 1, W)] - rgb[i3(row, col - 2, 1, W)]
                     - rgb[i3(row, col + 2, h_second, W)] + rgb[i3(row, col - 2, h_second, W)])
                + sqrf(g_h2);

        // Vert diff c=0 (xtrans_demosaic.py:366-367).
        v_diff += sqrf(rgb[i3(row + 1, col, 1, W)] - rgb[i3(row - 1, col, 1, W)]
                     - rgb[i3(row + 1, col, v_first, W)] + rgb[i3(row - 1, col, v_first, W)])
                + sqrf(g_v1);
        // Vert diff c=1 (xtrans_demosaic.py:369-370).
        v_diff += sqrf(rgb[i3(row + 2, col, 1, W)] - rgb[i3(row - 2, col, 1, W)]
                     - rgb[i3(row + 2, col, v_second, W)] + rgb[i3(row - 2, col, v_second, W)])
                + sqrf(g_v2);

        if (h_diff <= v_diff) {
            out_r = horiz_r / 2.0f;
            out_b = horiz_b / 2.0f;
        } else {
            out_r = vert_r / 2.0f;
            out_b = vert_b / 2.0f;
        }
    }
}

// ============================================================
// Kernel 4: R/B at solitary green (xtrans_demosaic.py:382-419)
// ============================================================
// At (row-sgrow)%3==0 && (col-sgcol)%3==0 && FCxt==1: compute R/B via
// _rb_solitary_green_for_dir modes 0/1/2/2 for the 4 direction buffers.
static void xtmRbAtGreen(std::vector<float>& rgb_d0,
                         std::vector<float>& rgb_d1,
                         std::vector<float>& rgb_d2,
                         std::vector<float>& rgb_d3,
                         const RawMosaic& m, const AllHex& hex) {
    const int W = m.width;
    const int H = m.height;
    const int sgrow = hex.sgrow;
    const int sgcol = hex.sgcol;
    const int pad = 6;

    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int row = pad; row < H - pad; ++row) {
        for (int col = pad; col < W - pad; ++col) {
            if ((row - sgrow) % 3 != 0 || (col - sgcol) % 3 != 0) continue;
            if (FCxt(m.xtrans, row, col) != 1) continue;

            const int h0 = FCxt(m.xtrans, row, col + 1);
            float r, b;

            rbSolitaryGreenForDir(rgb_d0.data(), row, col, W, h0, 0, r, b);
            rgb_d0[i3(row, col, 0, W)] = r;
            rgb_d0[i3(row, col, 2, W)] = b;

            rbSolitaryGreenForDir(rgb_d1.data(), row, col, W, h0, 1, r, b);
            rgb_d1[i3(row, col, 0, W)] = r;
            rgb_d1[i3(row, col, 2, W)] = b;

            rbSolitaryGreenForDir(rgb_d2.data(), row, col, W, h0, 2, r, b);
            rgb_d2[i3(row, col, 0, W)] = r;
            rgb_d2[i3(row, col, 2, W)] = b;

            rbSolitaryGreenForDir(rgb_d3.data(), row, col, W, h0, 2, r, b);
            rgb_d3[i3(row, col, 0, W)] = r;
            rgb_d3[i3(row, col, 2, W)] = b;
        }
    }
}

// ============================================================
// _rb_cross_for_dir (xtrans_demosaic.py:426-483)
// ============================================================
// Cross-color interpolation at R/B pixels for one direction buffer.
// f = target channel (0 or 2). Direction select via
//   d>1 || (d^c_val)&1 || grad_c < 2*grad_h.
static inline float rbCrossForDir(const float* rgb,
                                  int row, int col, int W,
                                  int f, int sgrow, int d) {
    // VERBATIM (xtrans_demosaic.py:437): is_vert_primary = (row-sgrow)%3 != 0.
    const bool is_vert_primary = ((row - sgrow) % 3 != 0);

    int c_dr, c_dc, h_dr, h_dc, c_val;
    if (is_vert_primary) {
        // c = TS (vertical, distance 1); h = 3 (horizontal, distance 3).
        c_dr = 1;  c_dc = 0;
        h_dr = 0;  h_dc = 3;
        c_val = 122;  // TS sentinel (xtrans_demosaic.py:453)
    } else {
        // c = 1 (horizontal, distance 1); h = 3*TS (vertical, distance 3).
        c_dr = 0;  c_dc = 1;
        h_dr = 3;  h_dc = 0;
        c_val = 1;
    }

    // Decision (xtrans_demosaic.py:463-474): use_c = True iff
    //   d > 1 || (d ^ c_val) & 1 || grad_c < 2*grad_h.
    bool use_c;
    if (d > 1) {
        use_c = true;
    } else if (((d ^ c_val) & 1) != 0) {
        use_c = true;
    } else {
        const float grad_c = (std::fabs(rgb[i3(row, col, 1, W)] - rgb[i3(row + c_dr, col + c_dc, 1, W)])
                            + std::fabs(rgb[i3(row, col, 1, W)] - rgb[i3(row - c_dr, col - c_dc, 1, W)]));
        const float grad_h = (std::fabs(rgb[i3(row, col, 1, W)] - rgb[i3(row + h_dr, col + h_dc, 1, W)])
                            + std::fabs(rgb[i3(row, col, 1, W)] - rgb[i3(row - h_dr, col - h_dc, 1, W)]));
        use_c = (grad_c < 2.0f * grad_h);
    }

    const int i_dr = use_c ? c_dr : h_dr;
    const int i_dc = use_c ? c_dc : h_dc;

    // xtrans_demosaic.py:479-483.
    return (rgb[i3(row + i_dr, col + i_dc, f, W)]
          + rgb[i3(row - i_dr, col - i_dc, f, W)]
          + 2.0f * rgb[i3(row, col, 1, W)]
          - rgb[i3(row + i_dr, col + i_dc, 1, W)]
          - rgb[i3(row - i_dr, col - i_dc, 1, W)]) / 2.0f;
}

// ============================================================
// Kernel 5: R/B cross-color interpolation (xtrans_demosaic.py:486-505)
// ============================================================
// f = 2 - FCxt (skip green f==1); _rb_cross_for_dir for the 4 dirs.
static void xtmRbInterp(std::vector<float>& rgb_d0,
                        std::vector<float>& rgb_d1,
                        std::vector<float>& rgb_d2,
                        std::vector<float>& rgb_d3,
                        const RawMosaic& m, const AllHex& hex) {
    const int W = m.width;
    const int H = m.height;
    const int sgrow = hex.sgrow;
    const int pad = 6;

    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int row = pad; row < H - pad; ++row) {
        for (int col = pad; col < W - pad; ++col) {
            const int f = 2 - FCxt(m.xtrans, row, col);
            if (f == 1) continue;  // green pixel

            rgb_d0[i3(row, col, f, W)] = rbCrossForDir(rgb_d0.data(), row, col, W, f, sgrow, 0);
            rgb_d1[i3(row, col, f, W)] = rbCrossForDir(rgb_d1.data(), row, col, W, f, sgrow, 1);
            rgb_d2[i3(row, col, f, W)] = rbCrossForDir(rgb_d2.data(), row, col, W, f, sgrow, 2);
            rgb_d3[i3(row, col, f, W)] = rbCrossForDir(rgb_d3.data(), row, col, W, f, sgrow, 3);
        }
    }
}

// ============================================================
// _fill_g22_for_dir (xtrans_demosaic.py:512-545)
// ============================================================
// R/B at a 2x2 green-block pixel using two hex neighbors.
static inline void fillG22ForDir(float* rgb,
                                 const AllHex& hex,
                                 int row, int col, int W,
                                 int hex_entry_a, int hex_entry_b) {
    const int r3 = row % 3;
    const int c3 = col % 3;
    const int dra = ah(hex.dr, r3, c3, hex_entry_a);
    const int dca = ah(hex.dc, r3, c3, hex_entry_a);
    const int drb = ah(hex.dr, r3, c3, hex_entry_b);
    const int dcb = ah(hex.dc, r3, c3, hex_entry_b);

    // xtrans_demosaic.py:530: sum_nonzero = (dra+drb != 0) || (dca+dcb != 0).
    const bool sum_nonzero = ((dra + drb) != 0) || ((dca + dcb) != 0);

    if (sum_nonzero) {
        const float g = (3.0f * rgb[i3(row, col, 1, W)]
                       - 2.0f * rgb[i3(row + dra, col + dca, 1, W)]
                       - rgb[i3(row + drb, col + dcb, 1, W)]);
        // ch = 0, 2 (R and B) — xtrans_demosaic.py:536.
        for (int ch = 0; ch <= 2; ch += 2) {
            rgb[i3(row, col, ch, W)] = (g + 2.0f * rgb[i3(row + dra, col + dca, ch, W)]
                                          + rgb[i3(row + drb, col + dcb, ch, W)]) / 3.0f;
        }
    } else {
        const float g = (2.0f * rgb[i3(row, col, 1, W)]
                       - rgb[i3(row + dra, col + dca, 1, W)]
                       - rgb[i3(row + drb, col + dcb, 1, W)]);
        for (int ch = 0; ch <= 2; ch += 2) {
            rgb[i3(row, col, ch, W)] = (g + rgb[i3(row + dra, col + dca, ch, W)]
                                         + rgb[i3(row + drb, col + dcb, ch, W)]) / 2.0f;
        }
    }
}

// ============================================================
// Kernel 6: Fill R/B at 2x2 green blocks (xtrans_demosaic.py:548-567)
// ============================================================
// Only dirs 0 and 1 (1-pass). hex entries (0,1) -> dir 0; (2,3) -> dir 1.
static void xtmFillGreen22(std::vector<float>& rgb_d0,
                           std::vector<float>& rgb_d1,
                           const AllHex& hex,
                           const RawMosaic& m) {
    const int W = m.width;
    const int H = m.height;
    const int sgrow = hex.sgrow;
    const int sgcol = hex.sgcol;
    const int pad = 8;

    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int row = pad; row < H - pad; ++row) {
        for (int col = pad; col < W - pad; ++col) {
            // xtrans_demosaic.py:561: skip if on solitary-green row/col grid.
            if ((row - sgrow) % 3 == 0 || (col - sgcol) % 3 == 0) continue;

            // d=0: hex[0], hex[1]; d=1: hex[2], hex[3] (xtrans_demosaic.py:565-567).
            fillG22ForDir(rgb_d0.data(), hex, row, col, W, 0, 1);
            fillG22ForDir(rgb_d1.data(), hex, row, col, W, 2, 3);
        }
    }
}

// ============================================================
// Kernel 7: YPbPr + directional derivatives (xtrans_demosaic.py:574-662)
// ============================================================
// BT.2020 YPbPr + 2nd-order directional derivatives, 4 dirs (H/V/diag/anti-diag).
static void xtmYuvDerivatives(const std::vector<float>& rgb_d0,
                              const std::vector<float>& rgb_d1,
                              const std::vector<float>& rgb_d2,
                              const std::vector<float>& rgb_d3,
                              std::vector<float>& drv0,
                              std::vector<float>& drv1,
                              std::vector<float>& drv2,
                              std::vector<float>& drv3,
                              const RawMosaic& m) {
    const int W = m.width;
    const int H = m.height;
    const int pad = 9;

    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int row = pad; row < H - pad; ++row) {
        for (int col = pad; col < W - pad; ++col) {
            // 4 directions (xtrans_demosaic.py:599-608).
            static const int dr_tab[4] = {0, 1, 1, 1};
            static const int dc_tab[4] = {1, 0, 1, -1};
            const float* rgb_dir[4] = {rgb_d0.data(), rgb_d1.data(), rgb_d2.data(), rgb_d3.data()};
            float*       drv_dir[4] = {drv0.data(),   drv1.data(),   drv2.data(),   drv3.data()};

            for (int d_idx = 0; d_idx < 4; ++d_idx) {
                const int dr = dr_tab[d_idx];
                const int dc = dc_tab[d_idx];
                const float* rx = rgb_dir[d_idx];

                const float rx0  = rx[i3(row, col, 0, W)];
                const float rx1  = rx[i3(row, col, 1, W)];
                const float rx2  = rx[i3(row, col, 2, W)];
                const float rxp0 = rx[i3(row + dr, col + dc, 0, W)];
                const float rxp1 = rx[i3(row + dr, col + dc, 1, W)];
                const float rxp2 = rx[i3(row + dr, col + dc, 2, W)];
                const float rxn0 = rx[i3(row - dr, col - dc, 0, W)];
                const float rxn1 = rx[i3(row - dr, col - dc, 1, W)];
                const float rxn2 = rx[i3(row - dr, col - dc, 2, W)];

                // BT.2020 YPbPr (xtrans_demosaic.py:639-649).
                const float y   = 0.2627f * rx0  + 0.6780f * rx1  + 0.0593f * rx2;
                const float yp  = 0.2627f * rxp0 + 0.6780f * rxp1 + 0.0593f * rxp2;
                const float yn  = 0.2627f * rxn0 + 0.6780f * rxn1 + 0.0593f * rxn2;

                const float pb   = (rx2  - y)  * 0.56433f;
                const float pbp  = (rxp2 - yp) * 0.56433f;
                const float pbn  = (rxn2 - yn) * 0.56433f;

                const float pr   = (rx0  - y)  * 0.67815f;
                const float prp  = (rxp0 - yp) * 0.67815f;
                const float prn  = (rxn0 - yn) * 0.67815f;

                const float deriv = (sqrf(2.0f * y  - yp  - yn)
                                   + sqrf(2.0f * pb - pbp - pbn)
                                   + sqrf(2.0f * pr - prp - prn));

                drv_dir[d_idx][static_cast<size_t>(row) * W + col] = deriv;
            }
        }
    }
}

// ============================================================
// Kernel 8: Homogeneity + merge (xtrans_demosaic.py:669-765)
// ============================================================
// For each pixel: build homogeneity counts (3x3 cells where drv <= 8*min_drv,
// summed over 5x5 per direction); merge dirs where homosum >= maxval-(maxval>>3).
static void xtmHomoMerge(const std::vector<float>& rgb_d0,
                         const std::vector<float>& rgb_d1,
                         const std::vector<float>& rgb_d2,
                         const std::vector<float>& rgb_d3,
                         const std::vector<float>& drv0,
                         const std::vector<float>& drv1,
                         const std::vector<float>& drv2,
                         const std::vector<float>& drv3,
                         ImageBuffer& out) {
    const int W = out.width;
    const int H = out.height;
    const int pad = kBORDER;

    const float* drv_dir[4] = {drv0.data(), drv1.data(), drv2.data(), drv3.data()};
    const float* rgb_dir[4] = {rgb_d0.data(), rgb_d1.data(), rgb_d2.data(), rgb_d3.data()};

    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int row = pad; row < H - pad; ++row) {
        for (int col = pad; col < W - pad; ++col) {
            int homosum[4] = {0, 0, 0, 0};

            // Phase 1+2 (xtrans_demosaic.py:691-721): 5x5 sum of 3x3 homogeneity
            // counts per direction. tr = 8 * min_drv over the 4 directions.
            for (int vv = -2; vv <= 2; ++vv) {
                for (int hh = -2; hh <= 2; ++hh) {
                    const int r2 = row + vv;
                    const int c2 = col + hh;
                    const float d0v = drv0[static_cast<size_t>(r2) * W + c2];
                    const float d1v = drv1[static_cast<size_t>(r2) * W + c2];
                    const float d2v = drv2[static_cast<size_t>(r2) * W + c2];
                    const float d3v = drv3[static_cast<size_t>(r2) * W + c2];
                    float tr = std::min(std::min(d0v, d1v), std::min(d2v, d3v));
                    tr *= 8.0f;

                    for (int di = 0; di < 4; ++di) {
                        const float* drv_di = drv_dir[di];
                        int cnt = 0;
                        for (int v3 = -1; v3 <= 1; ++v3) {
                            for (int h3 = -1; h3 <= 1; ++h3) {
                                const int r3 = r2 + v3;
                                const int c3 = c2 + h3;
                                const float dval = drv_di[static_cast<size_t>(r3) * W + c3];
                                if (dval <= tr) ++cnt;
                            }
                        }
                        homosum[di] += cnt;
                    }
                }
            }

            // Phase 3 (xtrans_demosaic.py:723-726): integer threshold.
            const int maxval = std::max(std::max(homosum[0], homosum[1]),
                                        std::max(homosum[2], homosum[3]));
            // VERBATIM (xtrans_demosaic.py:726): threshold = maxval - (maxval >> 3).
            const int threshold = maxval - (maxval >> 3);

            // Average directions at/above threshold (xtrans_demosaic.py:737-756).
            float avg_r = 0.0f, avg_g = 0.0f, avg_b = 0.0f;
            float cnt = 0.0f;
            for (int di = 0; di < 4; ++di) {
                if (homosum[di] >= threshold) {
                    const float* rx = rgb_dir[di];
                    avg_r += rx[i3(row, col, 0, W)];
                    avg_g += rx[i3(row, col, 1, W)];
                    avg_b += rx[i3(row, col, 2, W)];
                    cnt += 1.0f;
                }
            }

            float* px = out.pixel(row, col);
            if (cnt > 0.0f) {
                px[0] = std::max(0.0f, avg_r / cnt);
                px[1] = std::max(0.0f, avg_g / cnt);
                px[2] = std::max(0.0f, avg_b / cnt);
            } else {
                // Fallback (xtrans_demosaic.py:763-765): use dir 0.
                const float* rx = rgb_d0.data();
                px[0] = std::max(0.0f, rx[i3(row, col, 0, W)]);
                px[1] = std::max(0.0f, rx[i3(row, col, 1, W)]);
                px[2] = std::max(0.0f, rx[i3(row, col, 2, W)]);
            }
        }
    }
}

// ============================================================
// Kernel 9: Border bilinear interpolation (xtrans_demosaic.py:772-799)
// ============================================================
// 3x3 bilinear by FCxt for the kBORDER-px border ring. Reads original mosaic.
static void xtmBorderInterpolate(const RawMosaic& m, ImageBuffer& out) {
    const int W = m.width;
    const int H = m.height;
    const float* raw = m.data.data();
    const int border = kBORDER;

    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            if (row >= border && row < H - border && col >= border && col < W - border)
                continue;

            float sums[3] = {0.0f, 0.0f, 0.0f};
            float counts[3] = {0.0f, 0.0f, 0.0f};
            for (int dy = -1; dy <= 1; ++dy) {
                const int y = row + dy;
                if (y < 0 || y >= H) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int x = col + dx;
                    if (x < 0 || x >= W) continue;
                    const int c = FCxt(m.xtrans, y, x);
                    sums[c] += std::max(0.0f, raw[static_cast<size_t>(y) * W + x]);
                    counts[c] += 1.0f;
                }
            }

            float* px = out.pixel(row, col);
            for (int c = 0; c < 3; ++c) {
                if (counts[c] > 0.0f) {
                    px[c] = sums[c] / counts[c];
                }
                // else: leave existing value (kernel 8 output or 0.0f).
            }
        }
    }
}

// ============================================================
// Public API: xtransMarkesteijnDemosaic (xtrans_demosaic.py:806-914)
// ============================================================
ImageBuffer xtransMarkesteijnDemosaic(const RawMosaic& m) {
    const int W = m.width;
    const int H = m.height;
    const size_t N   = static_cast<size_t>(H) * W;
    const size_t N3  = N * 3;

    // Output (3-channel). Initialized to 0.
    ImageBuffer out(W, H, 3);

    // --- CPU precomputation (xtrans_demosaic.py:830) ---
    const AllHex hex = buildAllhex(m.xtrans);

    // 4 direction buffers, each (H, W, 3) float — interleaved RGB.
    std::vector<float> rgb_d0(N3, 0.0f), rgb_d1(N3, 0.0f),
                       rgb_d2(N3, 0.0f), rgb_d3(N3, 0.0f);

    // --- Kernel 1: Populate + copy (xtrans_demosaic.py:851-855) ---
    xtmPopulate(m, rgb_d0, rgb_d1, rgb_d2, rgb_d3);

    // --- BORDER PRE-FILL (stage-review Q2 — MANDATORY) ---
    // Fill the kBORDER=12 ring of all 4 dir buffers, all 3 channels, with
    // 3x3 bilinear from the raw mosaic. Prevents cascade contamination of
    // the interior by border zeros (darktable CLAMP_TO_EDGE equivalence).
    fillBorderIntermediates12(m, rgb_d0, rgb_d1, rgb_d2, rgb_d3);

    // --- Kernel 2: gmin/gmax (xtrans_demosaic.py:857-862) ---
    std::vector<float> gmin_buf(N, 0.0f), gmax_buf(N, 0.0f);
    xtmGminmax(rgb_d0, gmin_buf, gmax_buf, hex, m);

    // --- Kernel 3: Green interpolation (xtrans_demosaic.py:864-867) ---
    xtmGreenInterp(rgb_d0, rgb_d1, rgb_d2, rgb_d3,
                   gmin_buf, gmax_buf, hex, m);

    // xtrans_demosaic.py:869: del gmin_buf, gmax_buf.
    freeVector(gmin_buf);
    freeVector(gmax_buf);

    // --- Kernel 4: R/B at solitary green (xtrans_demosaic.py:871-872) ---
    xtmRbAtGreen(rgb_d0, rgb_d1, rgb_d2, rgb_d3, m, hex);

    // --- Kernel 5: R/B cross-color interpolation (xtrans_demosaic.py:874-875) ---
    xtmRbInterp(rgb_d0, rgb_d1, rgb_d2, rgb_d3, m, hex);

    // --- Kernel 6: Fill 2x2 green blocks (xtrans_demosaic.py:877-878) ---
    xtmFillGreen22(rgb_d0, rgb_d1, hex, m);

    // xtrans_demosaic.py:880: del ah_dr_gpu, ah_dc_gpu (allhex no longer needed).
    // (AllHex is on the stack; no explicit free needed.)

    // --- Kernel 7: Derivatives (xtrans_demosaic.py:882-889) ---
    std::vector<float> drv0(N, 0.0f), drv1(N, 0.0f),
                       drv2(N, 0.0f), drv3(N, 0.0f);
    xtmYuvDerivatives(rgb_d0, rgb_d1, rgb_d2, rgb_d3,
                      drv0, drv1, drv2, drv3, m);

    // --- Kernel 8: Homogeneity + merge (xtrans_demosaic.py:891-895) ---
    xtmHomoMerge(rgb_d0, rgb_d1, rgb_d2, rgb_d3,
                 drv0, drv1, drv2, drv3, out);

    // xtrans_demosaic.py:897-898: del rgb_d0..3, drv0..3.
    freeVector(rgb_d0);
    freeVector(rgb_d1);
    freeVector(rgb_d2);
    freeVector(rgb_d3);
    freeVector(drv0);
    freeVector(drv1);
    freeVector(drv2);
    freeVector(drv3);

    // --- Kernel 9: Border (xtrans_demosaic.py:900-903) ---
    // Overwrites the kBORDER=12 border ring with 3x3 bilinear from raw.
    xtmBorderInterpolate(m, out);

    return out;
}

} // namespace rawalchemy
