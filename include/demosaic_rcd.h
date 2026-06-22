// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file demosaic_rcd.h
 * @brief Ratio Corrected Demosaicing (RCD) for Bayer CFA sensors.
 *
 * Ported from darktable's rcd.c:
 *   Copyright (C) 2010-2026 darktable developers.
 *   Original algorithm: Luis Sanz Rodríguez, RCD 2.3 @ 171125.
 *   Tiling: Ingo Weyrich. Performance tuning: Hanno Schwalm.
 *   Original code: https://github.com/LuisSR/RCD-Demosaicing (GPL-3.0).
 *
 * The darktable source is preserved verbatim in .reference/darktable/rcd.c.
 * Helper substitutions applied (see Task 9 brief):
 *   - FC(row,col)              → detail::fcColor(row, col, filters)
 *   - dt_alloc_align_float(n)  → AlignedVector<float>(n)  (RAII, 64-byte align)
 *   - dt_free_align(p)         → (RAII destructor)
 *   - DT_OMP_PRAGMA(...)       → raw #pragma omp ...
 *   - interpolatef / CLIP / sqrf → inline functions in demosaic_rcd.cpp
 *
 * Deviations from darktable (documented in demosaic_rcd.cpp):
 *   - Output is planar RGB float[3*w*h], not float4 interleaved.
 *   - Border fallback uses CFA-aware 3x3 averaging (border_interpolate)
 *     instead of darktable's full PPG pass (demosaic_ppg source not in our
 *     reference bundle). Visual impact is confined to the rim.
 *   - No scaler normalization: input is [0,1], output is [0,1].
 */

namespace rawalchemy {

/// Algorithm constants — tuned values from darktable rcd.c.
/// Do not change without re-benchmarking quality/performance.
/// @{

/// Tile-overlap rim width. Real RCD output starts RCD_BORDER inside each
/// interior tile boundary (RCD_MARGIN inside the outermost tile).
/// darktable rcd.c:68 — "must be 10 to be stable".
constexpr int RCD_BORDER = 10;

/// Outermost-tile inner rim. The image edge is filled by the border fallback
/// out to RCD_BORDER pixels; the outermost RCD tile covers the rest starting
/// at RCD_MARGIN from the edge. darktable rcd.c:69.
constexpr int RCD_MARGIN = 9;

/// Compile-time tile dimension. darktable.h:265 — DT_RCD_TILESIZE default;
/// tuned for x86/64 cache behavior. The smaller value keeps per-thread
/// scratch at ~325 KB instead of ~33 MB.
constexpr int RCD_TILESIZE = 112;

/// Valid (non-overlapping) inner region per tile (= 92).
constexpr int RCD_TILEVALID = RCD_TILESIZE - 2 * RCD_BORDER;

/// @}

/// RCD demosaic for Bayer CFA input.
///
/// @param in       single-channel CFA mosaic, float[w*h] in [0,1], preprocessed
/// @param out      planar RGB output, float[3*w*h], layout [R plane | G plane | B plane]
/// @param w, h     image dimensions
/// @param filters  LibRaw CFA bitmask (any value other than 9 — X-Trans goes through markesteijn_demosaic)
void rcd_demosaic(const float* in, float* out, int w, int h, unsigned filters);

} // namespace rawalchemy
