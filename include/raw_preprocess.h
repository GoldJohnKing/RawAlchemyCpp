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
 * @brief Detect and replace hot pixels — port of darktable `hotpixels.c`.
 *
 * Direct C++ port of darktable's `_process_bayer` + `_process_xtrans`
 * (src/iop/hotpixels.c), matching darktable's iop order position 6.0
 * (after highlights, before rawdenoise). Replaces the prior Python
 * `fix_hot_pixels` global-σ + median algorithm (weaker:漏检 + 边缘误检).
 *
 * Algorithm (darktable semantics):
 *   - Per-pixel, per-CFA-plane: 4 same-color radial neighbors
 *     (Bayer: ±2 / ±width*2 same-color sites; X-Trans: precomputed offsets).
 *   - Detection: `pixel * (strength/2) > neighbor` for >= min_neighbours
 *     neighbors (min_neighbours = permissive ? 3 : 4). Also requires
 *     `pixel > threshold` to enter detection (avoid dark-current false hits).
 *   - Replacement: hot pixel <- max of qualifying neighbors (NOT median;
 *     darktable notes MAX produces fewer false-replacement artifacts).
 *
 * Data state: operates on [0,1]-normalized, POST-white-balance float CFA
 * mosaic (RawMosaic::data) — matches darktable's IOP_CS_RAW assumption
 * (post rawprepare + temperature). Single-channel; WB is a uniform per-
 * channel scale so the relative-brightness test is WB-invariant anyway.
 *
 * @param m            Mosaic (modified in-place; assumed post-WB normalized).
 * @param strength     0..1, darktable default 0.25 (-> multiplier 0.125).
 * @param threshold    0..1, darktable default 0.05 (skip pixels below this).
 * @param permissive   If true, require 3 (not 4) qualifying neighbors.
 */
void fixHotPixels(RawMosaic& m, float strength = 0.25f,
                  float threshold = 0.05f, bool permissive = false);

} // namespace rawalchemy
