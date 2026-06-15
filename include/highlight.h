#pragma once
/**
 * @file highlight.h
 * @brief Phase 2 highlight reconstruction — segmentation-based "inpaint-opposed".
 *
 * Direct port of Python reference `raw_alchemy.core`:
 *   - highlight_inpaint_opposed()  (core.py:53-136)
 *
 * Plus the opposing-channel reference average from `raw_alchemy.math_ops`:
 *   - compute_hl_refavg()          (math_ops.py:1182-1230)
 *
 * All image primitives (morphology, connected components, max-filter) are
 * hand-rolled (no OpenCV) to match the project's minimal-deps stance.
 *
 * Operates pre-demosaic on a packed CFA mosaic (RawMosaic::data, normalized
 * [0,1] from Phase 1). Modifies clipped CFA samples in place.
 */

#include <cstdint>
#include <vector>

#include "raw_mosaic.h"

namespace rawalchemy {

/**
 * @brief Output of the opposing-channel reference-average pass.
 *
 * `refavg` and `clipped` are both row-major H*W arrays.
 */
struct HlRefavg {
    std::vector<float>  refavg;   ///< Per-pixel opposing-channel reference (linear)
    std::vector<uint8_t> clipped; ///< Per-pixel clip flag (1 if val >= raw_clips[color])
};

/**
 * @brief Opposing-channel reference average (port of math_ops.py:1182-1230).
 *
 * For each pixel: take a 3x3 neighborhood (BORDER_REPLICATE), accumulate per-color
 * mean of max(raw,0), compute cube roots, then form the "opposing" average:
 *
 *   cbrt_mean[c] = cbrt(wb_gains[c] * mean[c] / cnt[c])   if cnt[c]>0 else 0
 *   opp_cbrt[color] = 0.5 * (sum of cbrt_mean over the other two colors)
 *   ref = opp_cbrt[color]^3 / wb_gains[color]             (if wb_gains[color] > 1e-6)
 *
 * `clipped[y*W+x] = (m.data[y*W+x] >= raw_clips[color_map[y*W+x]])`.
 *
 * OpenMP-parallelized across rows.
 *
 * @param m          Input mosaic (read-only).
 * @param color_map  Row-major H*W color-index map in {0,1,2}.
 * @param wb_gains   3 white-balance gains (R, G, B) — green-normalized.
 * @param raw_clips  3 per-color clip thresholds (CLIP / wb_gains[c]).
 */
HlRefavg computeHlRefavg(const RawMosaic& m, const uint8_t* color_map,
                         const float wb_gains[3], const float raw_clips[3]);

/**
 * @brief Segmentation-based highlight reconstruction (port of core.py:53-136).
 *
 * Pipeline:
 *   1. Normalize WB gains by green channel; compute per-color raw_clips.
 *   2. Build color_map (Bayer FC or X-Trans) with values collapsed to {0,1,2}.
 *   3. computeHlRefavg() -> per-pixel reference average + clip mask.
 *   4. For each color plane c in {0,1,2}:
 *        - morphologically close (7x7) the clip mask -> connected components.
 *        - grey-dilate (7x7) the label image to find the segment border zone.
 *        - accumulate per-segment chroma from unclipped border pixels.
 *        - replace clipped pixels with max(orig, refavg + seg_chroma[label]).
 *
 * The morphological close uses BORDER_CONSTANT=0 for BOTH passes (dilate
 * and erode) — this matches the cv2 oracle exactly and is essential for
 * correct border-segment chroma accumulation.
 *
 * @param m  Mosaic (modified in place; only clipped CFA samples change).
 */
void highlightInpaintOpposed(RawMosaic& m);

} // namespace rawalchemy
