// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file demosaic_markesteijn.h
 * @brief 3-pass Markesteijn demosaicing for Fuji X-Trans CFA sensors.
 *
 * Ported from darktable's xtrans.c (xtrans_markesteijn_interpolate):
 *   Copyright (C) 2010-2026 darktable developers.
 *   Original algorithm: Frank Markesteijn's algorithm for X-Trans sensors,
 *     adapted from dcraw 9.20 by Dave Coffin.
 *   Reference: https://github.com/darktable-org/darktable (GPL-3.0).
 *
 * Only the 3-pass Markesteijn variant is ported here. The frequency-domain
 * method (xtrans_fdc_interpolate) is out of scope.
 *
 * The darktable source is preserved verbatim in .reference/darktable/xtrans.c.
 * Helper substitutions applied (see Task 11 brief):
 *   - FCNxtrans(row,col)        -> xtransColor(row, col, xtrans)  [cfa_lookup.h]
 *   - dt_alloc_perthread(...)   -> per-thread AlignedVector<float>/AlignedVector<uint8_t>
 *   - dt_get_perthread(...)     -> (RAII buffer per OpenMP worker)
 *   - dt_free_align(p)          -> (RAII destructor)
 *   - DT_OMP_FOR()              -> #pragma omp for schedule(static)
 *   - dt_iop_image_copy(d,s,n)  -> std::memcpy(d, s, n * sizeof(float))
 *   - MIN/MAX/CLAMPS/TRANSLATE  -> std::min / std::max / inline helpers
 *   - dt_aligned_pixel_t        -> float[4]
 *   - sqrf(x)                   -> inline sqrf(x)
 *
 * Algorithmic deviations from darktable (documented in demosaic_markesteijn.cpp):
 *   - Output is planar RGB float[3*w*h], not float4 interleaved.
 *   - Border fallback uses X-Trans-aware 3x3 averaging instead of darktable's
 *     _vng_lininterpolate (VNG linear source not in our reference bundle).
 *   - homo/homosum live in a dedicated uint8_t buffer rather than aliasing the
 *     float yuv section, to avoid strict-aliasing UB. gmin/gmax still alias
 *     yuv[0]/yuv[1] (both float, no type punning).
 */

namespace rawalchemy {

/// Algorithm constants — verified from darktable source.
/// Do not change without re-benchmarking quality/performance.
/// @{

/// Compile-time tile dimension. darktable.h:277 — DT_MARKESTEIJN_TS.
/// Sized for L2 cache residency of the per-tile working set.
constexpr int MJK_TS = 122;

/// 3-pass mode (the higher-quality variant this module implements).
constexpr int MJK_PASSES = 3;

/// Direction count = 4 << (passes > 1) = 8 for 3-pass mode.
/// Four cardinal + four diagonal directions.
constexpr unsigned MJK_NDIR = 8;

/// Tile-overlap border width. (passes==1) ? 12 : 17. darktable xtrans.c:101.
constexpr int MJK_PAD_TILE = 17;

/// Homogeneity-map interior border. (passes==1) ? 10 : 15. darktable xtrans.c:445.
constexpr int MJK_PAD_HOMO = 15;

/// @}

/// 3-pass Markesteijn demosaic for X-Trans CFA input.
///
/// @param in      single-channel CFA mosaic, float[w*h] in [0,1], preprocessed
/// @param out     planar RGB output, float[3*w*h], layout [R plane | G plane | B plane]
/// @param w, h    image dimensions
/// @param xtrans  6x6 X-Trans pattern (from LibRaw imgdata.idata.xtrans;
///                values are 0=Red, 1=Green, 2=Blue)
void markesteijn_demosaic(const float* in, float* out, int w, int h,
                           const unsigned char xtrans[6][6]);

} // namespace rawalchemy
