// SPDX-License-Identifier: AGPL-3.0-or-later
//
// FRANK MARKESTEIJN'S 3-PASS X-TRANS DEMOSAIC.
//
// Ported from darktable's xtrans.c function xtrans_markesteijn_interpolate
// (Copyright (C) 2010-2026 darktable developers; original algorithm by
// Frank Markesteijn, adapted from dcraw 9.20 by Dave Coffin). Original code
// licensed under GNU GPL v3; this port retains GPL-compatibility via
// AGPL-3.0-or-later.
//
// The canonical darktable source lives at .reference/darktable/xtrans.c.
// Only the 3-pass Markesteijn variant is ported here (xtrans_fdc_interpolate,
// the frequency-domain complex-wavelet method, is out of scope).
//
// The algorithmic structure is preserved 1:1: hexagonal neighborhood
// precomputation, per-tile mosaic load with edge mirroring, gmin/gmax
// seeding, first-pass green interpolation in four directions, then for each
// of the three passes: green recalculation, R/B at solitary green, R<->B
// cross-interpolation, R/B at 2x2 green blocks; followed by YPbPr conversion,
// directional derivatives, homogeneity maps, 5x5 homogeneity sum, and the
// final most-homogeneous-direction blend.
//
// Helper substitutions (see Task 11 brief):
//   FCNxtrans(row,col,xtrans)       -> xtransColor(row, col, xtrans)
//   dt_alloc_perthread / get        -> per-thread AlignedVector<float>/uint8_t
//   dt_free_align(p)                -> (RAII destructor, no manual call)
//   DT_OMP_FOR()                    -> #pragma omp for schedule(static)
//   dt_iop_image_copy(dst,src,n)    -> std::memcpy(dst, src, n * sizeof(float))
//   MIN/MAX                         -> std::min / std::max
//   CLAMPS(A,L,H)                   -> inline clampsTo(A, L, H)
//   TRANSLATE(n,size)               -> inline translateCoord(n, size)
//   sqrf(x)                         -> inline sqrf(x)
//   dt_aligned_pixel_t              -> float[4]
//
// Algorithmic deviations from darktable:
//   1. Output format: darktable writes RGBA-interleaved float4
//      (out[4*idx + c]). We write planar RGB float[3*w*h]
//      (out[idx + c*plane]) per our public contract.
//   2. Border fallback: darktable calls _vng_lininterpolate() for pixels
//      outside any tile (sub-tile-sized images and the outer rim). That
//      source is not in our reference bundle; we substitute
//      borderInterpolateXtrans(), an X-Trans-aware 3x3 averaging pass that
//      runs before the tile loop. For normal-sized images the tiles
//      overwrite the full interior, so the visual impact is confined to
//      sub-tile-sized images.
//   3. Memory layout: darktable aliases homo/homosum (uint8_t) onto the yuv
//      section of its single float buffer. We allocate a dedicated
//      AlignedVector<uint8_t> for homo+homosum to avoid float<->uint8_t
//      strict-aliasing UB. gmin/gmax still alias yuv[0]/yuv[1] (both float).
//      Net per-thread memory overhead is ~0.4 MB, acceptable for correctness.
//   4. Division-by-zero guard in edge-mirror interpolation: darktable uses
//      an implicit count>0 assumption. We guard explicitly.

#include "demosaic_markesteijn.h"
#include "aligned_allocator.h"
#include "cfa_lookup.h"


#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rawalchemy {
namespace {

// Padding constants for each sub-stage. darktable xtrans.c:23-25.
constexpr int PAD_G1_G3   = 3;   // gmin/gmax seed pass
constexpr int PAD_G_INTERP = 3;  // first-pass green interpolation
constexpr int PAD_G_RECALC = 6;  // green recalculation (passes > 0)

// Sub-stage interior borders for 3-pass mode. darktable xtrans.c:298,350,369.
constexpr int PAD_RB_G  = 5;  // R/B at solitary green
constexpr int PAD_RB_BR = 5;  // R<->B cross-interpolation
constexpr int PAD_G22   = 4;  // R/B at 2x2 green blocks

// YPbPr/derivative interior borders for 3-pass mode. darktable xtrans.c:412,432.
constexpr int PAD_YUV = 13;
constexpr int PAD_DRV = 14;

inline float sqrf(float x) { return x * x; }

// NaN-safe three-way clamp. Matches darktable's CLAMPS macro (darktable.h:285):
// NaN compares false and yields H, same as the macro.
inline float clampsTo(float a, float lo, float hi) {
    return (a > lo) ? ((a < hi) ? a : hi) : lo;
}

// Edge mirroring: reflect coordinates that fall off the image back into range.
// Matches darktable's TRANSLATE macro. n may be negative or >= size.
inline int translateCoord(int n, int size) {
    return (n >= size) ? (2 * size - n - 2) : std::abs(n);
}

// darktable's _hexmap (xtrans.c:28-40): looks up the precomputed hexagonal
// neighborhood offset table. row/col may be negative; adding 600 (a large
// multiple of 3) makes the subsequent % 3 well-defined for negative inputs.
// A reference parameter avoids pointer-decay qualification-conversion quirks.
inline const short* hexLookup(int row, int col,
                               const short (&allhex)[3][3][8]) {
    const int irow = row + 600;
    const int icol = col + 600;
    return allhex[irow % 3][icol % 3];
}

// X-Trans-aware 3x3 averaging for the outer rim and sub-tile-sized images.
// Replaces darktable's _vng_lininterpolate() call. Writes planar RGB.
void borderInterpolateXtrans(const float* in, float* out, int w, int h,
                              const char xtrans[6][6]) {
    const size_t plane = static_cast<size_t>(w) * static_cast<size_t>(h);
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            float sum[3] = {0.0f, 0.0f, 0.0f};
            int cnt[3] = {0, 0, 0};
            for (int dy = -1; dy <= 1; ++dy) {
                const int y = row + dy;
                if (y < 0 || y >= h) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int x = col + dx;
                    if (x < 0 || x >= w) continue;
                    const unsigned c = xtransColor(y, x, xtrans);
                    sum[c] += std::max(0.0f, in[static_cast<size_t>(y) * w + x]);
                    cnt[c] += 1;
                }
            }
            const unsigned center_c = xtransColor(row, col, xtrans);
            const float center_val =
                std::max(0.0f, in[static_cast<size_t>(row) * w + col]);
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

void markesteijn_demosaic(const float* in, float* out, int w, int h,
                           const char xtrans[6][6]) {
    if (!in || !out || w <= 0 || h <= 0 || !xtrans) return;

    // Stage 0: CFA-aware border fill. The tile loop overwrites the interior
    // for normal-sized images; for sub-tile images this is the final output.
    borderInterpolateXtrans(in, out, w, h, xtrans);

    // For images smaller than one tile, the tile loop produces no output.
    if (w <= 2 * MJK_PAD_TILE || h <= 2 * MJK_PAD_TILE) return;

    // Direction vectors for hexagonal neighborhood construction.
    // darktable xtrans.c:53-56. Ported verbatim.
    static const short orth[12] = {1, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 1};
    static const short patt[2][16] = {
        {0, 1, 0, -1, 2, 0, -1, 0, 1, 1, 1, -1, 0, 0, 0, 0},
        {0, 1, 0, -2, 1, 0, -2, 0, 1, 1, -2, -2, 1, -1, -1, 1}
    };
    // Strides for the four derivative directions: horizontal (1), vertical
    // (TS), and the two diagonals (TS+1, TS-1). darktable xtrans.c:56.
    static const int dir[4] = {1, MJK_TS, MJK_TS + 1, MJK_TS - 1};

    short allhex[3][3][8];
    // sgrow/sgcol: offset of the solitary green pixel in the 6x6 X-Trans unit.
    // Initialized to suppress -Wmaybe-uninitialized; assigned in the loop.
    unsigned short sgrow = 0, sgcol = 0;

    // Map a green hexagon around each non-green pixel and vice versa.
    // darktable xtrans.c:74-98. Ported verbatim.
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            for (int ng = 0, d = 0; d < 10; d += 2) {
                const int g = (xtransColor(row, col, xtrans) == 1u) ? 1 : 0;
                if (xtransColor(row + orth[d], col + orth[d + 2], xtrans) == 1u)
                    ng = 0;
                else
                    ++ng;
                // Four non-green neighbors in cardinal directions => this is
                // the solitary green pixel of the X-Trans unit cell.
                if (ng == 4) {
                    sgrow = static_cast<unsigned short>(row);
                    sgcol = static_cast<unsigned short>(col);
                }
                if (ng == g + 1) {
                    for (int c = 0; c < 8; ++c) {
                        const int v = orth[d] * patt[g][c * 2]
                                    + orth[d + 1] * patt[g][c * 2 + 1];
                        const int h2 = orth[d + 2] * patt[g][c * 2]
                                     + orth[d + 3] * patt[g][c * 2 + 1];
                        allhex[row][col][c ^ (g * 2 & d)] =
                            static_cast<short>(h2 + v * MJK_TS);
                    }
                }
            }
        }
    }

    const int pad_tile = MJK_PAD_TILE;
    const unsigned ndir = MJK_NDIR;
    const size_t out_plane = static_cast<size_t>(w) * static_cast<size_t>(h);

    // Per-thread scratch sizing. The float buffer holds rgb[ndir][TS][TS][3],
    // yuv[3][TS][TS] (aliased by gmin/gmax), and drv[ndir][TS][TS].
    // darktable xtrans.c:64.
    const size_t buf_floats =
        static_cast<size_t>(MJK_TS) * MJK_TS * (ndir * 4 + 3);
    // homo and homosum share one uint8_t allocation (sequential, not
    // overlapping) per the buffer-aliasing note in the brief.
    const size_t homo_bytes = 2 * ndir * static_cast<size_t>(MJK_TS) * MJK_TS;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        AlignedVector<float> buffer(buf_floats);
        AlignedVector<uint8_t> homo_buf(homo_bytes);

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        // Outer tile loop — the only parallelized loop. Each iteration
        // processes one row of tiles. Inner per-direction kernels have
        // data dependencies and run sequentially. darktable xtrans.c:102-105.
        for (int top = -pad_tile; top < h - pad_tile;
             top += MJK_TS - (pad_tile * 2)) {
            // Per-tile pointer aliases into the per-thread buffer.
            // rgb: ndir tiles of [TS][TS][3] — the directional estimates.
            float (*rgb)[MJK_TS][MJK_TS][3] =
                reinterpret_cast<float(*)[MJK_TS][MJK_TS][3]>(buffer.data());
            // yuv: 3 channels of [TS][TS], channels-first for the
            // derivative loop's yfx pointer trick. darktable xtrans.c:113.
            float (*yuv)[MJK_TS][MJK_TS] = reinterpret_cast<float(*)[MJK_TS][MJK_TS]>(
                buffer.data() + static_cast<size_t>(MJK_TS) * MJK_TS * (ndir * 3));
            // drv: ndir single-channel derivative tiles. darktable xtrans.c:115.
            float (*drv)[MJK_TS][MJK_TS] = reinterpret_cast<float(*)[MJK_TS][MJK_TS]>(
                buffer.data() + static_cast<size_t>(MJK_TS) * MJK_TS * (ndir * 3 + 3));
            // gmin/gmax alias yuv[0]/yuv[1] — both float, no type punning.
            // Used early (green interp), dead before yuv is computed.
            // darktable xtrans.c:118-119.
            float (*gmin)[MJK_TS] = reinterpret_cast<float(*)[MJK_TS]>(
                buffer.data() + static_cast<size_t>(MJK_TS) * MJK_TS * (ndir * 3));
            float (*gmax)[MJK_TS] = reinterpret_cast<float(*)[MJK_TS]>(
                buffer.data() + static_cast<size_t>(MJK_TS) * MJK_TS * (ndir * 3 + 1));

            // homo/homosum: ndir tiles each, sequential in a dedicated
            // uint8_t allocation. darktable xtrans.c:122-124.
            uint8_t (*homo)[MJK_TS][MJK_TS] =
                reinterpret_cast<uint8_t(*)[MJK_TS][MJK_TS]>(homo_buf.data());
            uint8_t (*homosum)[MJK_TS][MJK_TS] =
                reinterpret_cast<uint8_t(*)[MJK_TS][MJK_TS]>(
                    homo_buf.data() + ndir * static_cast<size_t>(MJK_TS) * MJK_TS);

            // Inner loop over horizontal tiles. darktable xtrans.c:126.
            for (int left = -pad_tile; left < w - pad_tile;
                 left += MJK_TS - (pad_tile * 2)) {
                int mrow = std::min(top + MJK_TS, h + pad_tile);
                int mcol = std::min(left + MJK_TS, w + pad_tile);

                // ---- Copy current tile into rgb[0], mirroring at edges ----
                // darktable xtrans.c:134-178.
                for (int row = top; row < mrow; ++row) {
                    for (int col = left; col < mcol; ++col) {
                        float* const pix = &rgb[0][row - top][col - left][0];
                        if (col >= 0 && row >= 0 && col < w && row < h) {
                            const unsigned f = xtransColor(row, col, xtrans);
                            const float v = std::max(
                                0.0f, in[static_cast<size_t>(w) * row + col]);
                            for (int c = 0; c < 3; ++c)
                                pix[c] = (c == static_cast<int>(f)) ? v : 0.0f;
                        } else {
                            const unsigned c = xtransColor(row, col, xtrans);
                            for (int cc = 0; cc < 3; ++cc) {
                                if (cc != static_cast<int>(c)) {
                                    pix[cc] = 0.0f;
                                } else {
                                    const int cy = translateCoord(row, h);
                                    const int cx = translateCoord(col, w);
                                    if (c == xtransColor(cy, cx, xtrans)) {
                                        pix[c] = std::max(0.0f,
                                            in[static_cast<size_t>(w) * cy + cx]);
                                    } else {
                                        // Mirror pixel is a different CFA
                                        // color: average same-color neighbors.
                                        float sum = 0.0f;
                                        int count = 0;
                                        for (int y = row - 1; y <= row + 1; ++y) {
                                            for (int x = col - 1; x <= col + 1; ++x) {
                                                const int yy = translateCoord(y, h);
                                                const int xx = translateCoord(x, w);
                                                if (xtransColor(yy, xx, xtrans) == c) {
                                                    sum += std::max(0.0f,
                                                        in[static_cast<size_t>(w) * yy + xx]);
                                                    ++count;
                                                }
                                            }
                                        }
                                        pix[c] = (count > 0)
                                            ? (sum / static_cast<float>(count))
                                            : 0.0f;
                                    }
                                }
                            }
                        }
                    }
                }

                // Duplicate rgb[0] to rgb[1..3]. darktable xtrans.c:180-182.
                for (int c = 1; c <= 3; ++c)
                    std::memcpy(rgb[c], rgb[0], sizeof(*rgb));

                // ---- Seed gmin/gmax (min/max of surrounding greens) ----
                // darktable xtrans.c:194-241. The switch manipulates row/col
                // to walk horizontal and vertical red/blue pairs. Ported
                // verbatim including the in-loop row/col mutation.
                for (int row = top + PAD_G1_G3; row < mrow - PAD_G1_G3; ++row) {
                    float gmin_val = std::numeric_limits<float>::max();
                    float gmax_val = 0.0f;
                    for (int col = left + PAD_G1_G3; col < mcol - PAD_G1_G3; ++col) {
                        if (xtransColor(row, col, xtrans) == 1u) {
                            gmin_val = std::numeric_limits<float>::max();
                            gmax_val = 0.0f;
                            continue;
                        }
                        // max == 0.0f flags a new red/blue pair needing a
                        // fresh min/max scan of its surrounding greens.
                        if (gmax_val == 0.0f) {
                            float (*const pix)[3] = &rgb[0][row - top][col - left];
                            const short* const hex = hexLookup(row, col, allhex);
                            for (int c = 0; c < 6; ++c) {
                                const float val = pix[hex[c]][1];
                                if (gmin_val > val) gmin_val = val;
                                if (gmax_val < val) gmax_val = val;
                            }
                        }
                        gmin[row - top][col - left] = gmin_val;
                        gmax[row - top][col - left] = gmax_val;
                        switch ((row - static_cast<int>(sgrow)) % 3) {
                            case 1:
                                if (row < mrow - 4) { ++row; --col; }
                                break;
                            case 2:
                                gmin_val = std::numeric_limits<float>::max();
                                gmax_val = 0.0f;
                                col += 2;
                                if (col < mcol - 4 && row > top + 3) --row;
                                break;
                            default:
                                break;
                        }
                    }
                }

                // ---- First-pass green interpolation (4 directions) ----
                // darktable xtrans.c:245-265. Populates the green channel
                // of rgb[0..3] at non-green CFA positions, clamped to the
                // gmin/gmax envelope. Constants from dcraw integer math.
                for (int row = top + PAD_G_INTERP; row < mrow - PAD_G_INTERP; ++row) {
                    for (int col = left + PAD_G_INTERP; col < mcol - PAD_G_INTERP; ++col) {
                        float color[8];
                        const unsigned f = xtransColor(row, col, xtrans);
                        if (f == 1u) continue;
                        float (*const pix)[3] = &rgb[0][row - top][col - left];
                        const short* const hex = hexLookup(row, col, allhex);
                        color[0] = 0.6796875f * (pix[hex[1]][1] + pix[hex[0]][1])
                                 - 0.1796875f * (pix[2 * hex[1]][1] + pix[2 * hex[0]][1]);
                        color[1] = 0.87109375f * pix[hex[3]][1]
                                 + pix[hex[2]][1] * 0.13f
                                 + 0.359375f * (pix[0][f] - pix[-hex[2]][f]);
                        for (int c = 0; c < 2; ++c)
                            color[2 + c] = 0.640625f * pix[hex[4 + c]][1]
                                         + 0.359375f * pix[-2 * hex[4 + c]][1]
                                         + 0.12890625f * (2 * pix[0][f]
                                             - pix[3 * hex[4 + c]][f]
                                             - pix[-3 * hex[4 + c]][f]);
                        for (int c = 0; c < 4; ++c)
                            rgb[c ^ !((row - static_cast<int>(sgrow)) % 3)]
                               [row - top][col - left][1] =
                                clampsTo(color[c],
                                    gmin[row - top][col - left],
                                    gmax[row - top][col - left]);
                    }
                }

                // ---- Multi-pass loop (passes 0, 1, 2) ----
                // darktable xtrans.c:267-393. Pass 0 works on rgb[0..3].
                // Pass 1 copies rgb[0..3] to rgb[4..7] and advances the rgb
                // pointer. Pass 2 refines the second set.
                for (int pass = 0; pass < MJK_PASSES; ++pass) {
                    if (pass == 1) {
                        std::memcpy(rgb + 4, rgb, sizeof(*rgb) * 4);
                        rgb += 4;
                    }

                    // Recalculate green from interpolated values (passes > 0).
                    // darktable xtrans.c:278-295.
                    if (pass) {
                        for (int row = top + PAD_G_RECALC;
                             row < mrow - PAD_G_RECALC; ++row) {
                            for (int col = left + PAD_G_RECALC;
                                 col < mcol - PAD_G_RECALC; ++col) {
                                const unsigned f = xtransColor(row, col, xtrans);
                                if (f == 1u) continue;
                                const short* const hex = hexLookup(row, col, allhex);
                                for (int d = 3; d < 6; ++d) {
                                    float (*rfx)[3] =
                                        &rgb[(d - 2) ^ !((row - static_cast<int>(sgrow)) % 3)]
                                            [row - top][col - left];
                                    const float val = rfx[-2 * hex[d]][1]
                                        + 2 * rfx[hex[d]][1]
                                        - rfx[-2 * hex[d]][f]
                                        - 2 * rfx[hex[d]][f]
                                        + 3 * rfx[0][f];
                                    rfx[0][1] = clampsTo(val / 3.0f,
                                        gmin[row - top][col - left],
                                        gmax[row - top][col - left]);
                                }
                            }
                        }
                    }

                    // Interpolate R/B at solitary green pixels.
                    // darktable xtrans.c:297-347. Walks the 3x3 X-Trans grid
                    // landing on each solitary green. Six alternating passes
                    // (hori/vert) feed rgb[0..3].
                    for (int row = (top - static_cast<int>(sgrow) + PAD_RB_G + 2) / 3 * 3
                                 + static_cast<int>(sgrow);
                         row < mrow - PAD_RB_G;
                         row += 3) {
                        for (int col = (left - static_cast<int>(sgcol) + PAD_RB_G + 2) / 3 * 3
                                     + static_cast<int>(sgcol);
                             col < mcol - PAD_RB_G;
                             col += 3) {
                            float (*rfx)[3] = &rgb[0][row - top][col - left];
                            int hh = static_cast<int>(xtransColor(row, col + 1, xtrans));
                            float diff[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                            float color[2][6];
                            for (int i = 1, d = 0; d < 6; ++d, i ^= MJK_TS ^ 1, hh ^= 2) {
                                for (int c = 0; c < 2; ++c, hh ^= 2) {
                                    const float g = 2 * rfx[0][1]
                                        - rfx[i << c][1]
                                        - rfx[-(i << c)][1];
                                    color[hh != 0][d] = g
                                        + rfx[i << c][hh]
                                        + rfx[-(i << c)][hh];
                                    if (d > 1)
                                        diff[d] += sqrf(
                                            rfx[i << c][1] - rfx[-(i << c)][1]
                                            - rfx[i << c][hh] + rfx[-(i << c)][hh])
                                            + sqrf(g);
                                }
                                if ((d < 2) || (d & 1)) {
                                    const int d_out = d - ((d > 1) && (diff[d - 1] < diff[d]));
                                    rfx[0][0] = color[0][d_out] / 2.0f;
                                    rfx[0][2] = color[1][d_out] / 2.0f;
                                    rfx += MJK_TS * MJK_TS;
                                }
                            }
                        }
                    }

                    // Interpolate R at B pixels and B at R pixels.
                    // darktable xtrans.c:349-366. Selects between horizontal
                    // and vertical axis based on green-gradient steepness.
                    for (int row = top + PAD_RB_BR; row < mrow - PAD_RB_BR; ++row) {
                        for (int col = left + PAD_RB_BR; col < mcol - PAD_RB_BR; ++col) {
                            const unsigned fcu = xtransColor(row, col, xtrans);
                            const int f = 2 - static_cast<int>(fcu);
                            if (f == 1) continue;
                            float (*rfx)[3] = &rgb[0][row - top][col - left];
                            const int c = (row - static_cast<int>(sgrow)) % 3 ? MJK_TS : 1;
                            const int h3 = 3 * (c ^ MJK_TS ^ 1);
                            for (int d = 0; d < 4; ++d, rfx += MJK_TS * MJK_TS) {
                                const int i = (d > 1 || ((d ^ c) & 1) ||
                                    ((std::fabs(rfx[0][1] - rfx[c][1])
                                      + std::fabs(rfx[0][1] - rfx[-c][1])) <
                                     2.0f * (std::fabs(rfx[0][1] - rfx[h3][1])
                                      + std::fabs(rfx[0][1] - rfx[-h3][1]))))
                                    ? c : h3;
                                rfx[0][f] = (rfx[i][f] + rfx[-i][f]
                                    + 2.0f * rfx[0][1]
                                    - rfx[i][1] - rfx[-i][1]) / 2.0f;
                            }
                        }
                    }

                    // Fill R/B for 2x2 blocks of green.
                    // darktable xtrans.c:368-392. Uses the hex offsets to
                    // find the nearest non-green pair for interpolation.
                    for (int row = top + PAD_G22; row < mrow - PAD_G22; ++row) {
                        if ((row - static_cast<int>(sgrow)) % 3) {
                            for (int col = left + PAD_G22; col < mcol - PAD_G22; ++col) {
                                if ((col - static_cast<int>(sgcol)) % 3) {
                                    float (*rfx)[3] = &rgb[0][row - top][col - left];
                                    const short* const hex = hexLookup(row, col, allhex);
                                    for (unsigned d = 0; d < ndir; d += 2, rfx += MJK_TS * MJK_TS) {
                                        if (hex[d] + hex[d + 1]) {
                                            const float g = 3.0f * rfx[0][1]
                                                - 2.0f * rfx[hex[d]][1]
                                                - rfx[hex[d + 1]][1];
                                            for (int c = 0; c < 4; c += 2)
                                                rfx[0][c] = (g + 2.0f * rfx[hex[d]][c]
                                                    + rfx[hex[d + 1]][c]) / 3.0f;
                                        } else {
                                            const float g = 2.0f * rfx[0][1]
                                                - rfx[hex[d]][1]
                                                - rfx[hex[d + 1]][1];
                                            for (int c = 0; c < 4; c += 2)
                                                rfx[0][c] = (g + rfx[hex[d]][c]
                                                    + rfx[hex[d + 1]][c]) / 2.0f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } // end multi-pass loop

                // Reset rgb to the first set and switch to tile-local coords.
                // darktable xtrans.c:395-402.
                rgb = reinterpret_cast<float(*)[MJK_TS][MJK_TS][3]>(buffer.data());
                mrow -= top;
                mcol -= left;

                // ---- Convert to YPbPr and differentiate ----
                // darktable xtrans.c:404-441. ITU-R BT.2020 YPbPr matrix.
                for (unsigned d = 0; d < ndir; ++d) {
                    for (int row = PAD_YUV; row < mrow - PAD_YUV; ++row) {
                        for (int col = PAD_YUV; col < mcol - PAD_YUV; ++col) {
                            const float* rx = rgb[d][row][col];
                            const float y = 0.2627f * rx[0]
                                          + 0.6780f * rx[1]
                                          + 0.0593f * rx[2];
                            yuv[0][row][col] = y;
                            yuv[1][row][col] = (rx[2] - y) * 0.56433f;
                            yuv[2][row][col] = (rx[0] - y) * 0.67815f;
                        }
                    }
                    // Derivative along this direction. f is the stride
                    // (1=horizontal, TS=vertical, TS±1=diagonals). The yfx
                    // cast lets us index all three YUV channels at once.
                    // darktable xtrans.c:431-440.
                    const int f = dir[d & 3];
                    for (int row = PAD_DRV; row < mrow - PAD_DRV; ++row) {
                        for (int col = PAD_DRV; col < mcol - PAD_DRV; ++col) {
                            const float (*yfx)[MJK_TS][MJK_TS] =
                                reinterpret_cast<const float(*)[MJK_TS][MJK_TS]>(
                                    &yuv[0][row][col]);
                            drv[d][row][col] =
                                sqrf(2 * yfx[0][0][0] - yfx[0][0][f] - yfx[0][0][-f])
                              + sqrf(2 * yfx[1][0][0] - yfx[1][0][f] - yfx[1][0][-f])
                              + sqrf(2 * yfx[2][0][0] - yfx[2][0][f] - yfx[2][0][-f]);
                        }
                    }
                }

                // ---- Build homogeneity maps ----
                // darktable xtrans.c:443-457. Counts neighbors (3x3) whose
                // derivative is below 8x the local minimum.
                std::memset(homo, 0, sizeof(uint8_t) * ndir
                    * static_cast<size_t>(MJK_TS) * MJK_TS);
                for (int row = MJK_PAD_HOMO; row < mrow - MJK_PAD_HOMO; ++row) {
                    for (int col = MJK_PAD_HOMO; col < mcol - MJK_PAD_HOMO; ++col) {
                        float tr = std::numeric_limits<float>::max();
                        for (unsigned d = 0; d < ndir; ++d)
                            if (tr > drv[d][row][col]) tr = drv[d][row][col];
                        tr *= 8;
                        for (unsigned d = 0; d < ndir; ++d) {
                            for (int v = -1; v <= 1; ++v) {
                                for (int hh2 = -1; hh2 <= 1; ++hh2) {
                                    homo[d][row][col] +=
                                        (drv[d][row + v][col + hh2] <= tr) ? 1 : 0;
                                }
                            }
                        }
                    }
                }

                // ---- 5x5 sum of homogeneity maps ----
                // darktable xtrans.c:459-476. Rolling column-sum update.
                for (unsigned d = 0; d < ndir; ++d) {
                    for (int row = pad_tile; row < mrow - pad_tile; ++row) {
                        int col = pad_tile - 5;
                        uint8_t v5sum[5] = {0, 0, 0, 0, 0};
                        homosum[d][row][col] = 0;
                        for (++col; col < mcol - pad_tile; ++col) {
                            uint8_t colsum = 0;
                            for (int v = -2; v <= 2; ++v)
                                colsum += homo[d][row + v][col + 2];
                            homosum[d][row][col] =
                                homosum[d][row][col - 1] - v5sum[col % 5] + colsum;
                            v5sum[col % 5] = colsum;
                        }
                    }
                }

                // ---- Average the most homogeneous directions ----
                // darktable xtrans.c:478-508. Blends directional estimates
                // whose homogeneity is within 7/8 of the peak.
                for (int row = pad_tile; row < mrow - pad_tile; ++row) {
                    for (int col = pad_tile; col < mcol - pad_tile; ++col) {
                        uint8_t hm[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                        uint8_t maxval = 0;
                        for (unsigned d = 0; d < ndir; ++d) {
                            hm[d] = homosum[d][row][col];
                            maxval = (maxval < hm[d]) ? hm[d] : maxval;
                        }
                        maxval -= maxval >> 3;
                        for (unsigned d = 0; d < ndir - 4; ++d) {
                            if (hm[d] < hm[d + 4])
                                hm[d] = 0;
                            else if (hm[d] > hm[d + 4])
                                hm[d + 4] = 0;
                        }
                        float avg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        for (unsigned d = 0; d < ndir; ++d) {
                            if (hm[d] >= maxval) {
                                for (int c = 0; c < 3; ++c)
                                    avg[c] += rgb[d][row][col][c];
                                avg[3] += 1.0f;
                            }
                        }
                        const size_t idx = static_cast<size_t>(w) * (row + top)
                                         + static_cast<size_t>(col + left);
                        out[idx] = std::max(0.0f, avg[0] / avg[3]);
                        out[idx + out_plane] =
                            std::max(0.0f, avg[1] / avg[3]);
                        out[idx + 2 * out_plane] =
                            std::max(0.0f, avg[2] / avg[3]);
                    }
                }
            } // end inner tile loop
        } // end outer tile loop
    } // end parallel region
}


} // namespace rawalchemy
