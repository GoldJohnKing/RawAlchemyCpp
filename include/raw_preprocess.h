#pragma once
/**
 * @file raw_preprocess.h
 * @brief Phase 1 RAW preprocessing — black-level subtraction + hot-pixel fix.
 *
 * Direct ports of Python reference `raw_alchemy.core`:
 *   - subtract_black_level()  (core.py:20-31)
 *   - fix_hot_pixels()        (core.py:34-46)
 *
 * Both operate in-place on a RawMosaic. subtractBlackLevel() normalizes the
 * raw uint16-cast float data into [0, 1] per CFA plane. fixHotPixels() then
 * removes single-pixel outliers via per-plane 3x3 median comparison.
 */

#include "raw_mosaic.h"

namespace rawalchemy {

/**
 * @brief Per-channel black-level subtraction + normalization to [0, 1].
 *
 * Port of Python `subtract_black_level` (core.py:20-31).
 *
 * For each CFA cell offset (r, c) in [0, patSize):
 *   data[r::patSize, c::patSize] = max(val - bl_c, 0) / (maximum - bl_c)
 * where patSize = (filters==9) ? 6 : 2 and bl_c is the per-channel collapsed
 * black level for that CFA color.
 *
 * @param m  Mosaic (modified in-place; values become [0, 1] float).
 */
void subtractBlackLevel(RawMosaic& m);

/**
 * @brief Detect and replace hot/dead pixels per CFA plane.
 *
 * Port of Python `fix_hot_pixels` (core.py:34-46).
 *
 * Operates on each CFA plane independently (subsampled by patSize). For each
 * plane: 3x3 median filter (BORDER_REPLICATE), compute |plane - median|,
 * flag pixels where diff > threshold * std(diff), replace with median.
 *
 * Hand-rolled median (no OpenCV). OpenMP-parallelized across the
 * patSize x patSize plane offsets.
 *
 * @param m          Mosaic (modified in-place; assumed already normalized).
 * @param threshold  Outlier threshold in units of std (default 4.0, matches Python).
 */
void fixHotPixels(RawMosaic& m, float threshold = 4.0f);

} // namespace rawalchemy
