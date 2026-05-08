#pragma once
/**
 * @file stylize.h
 * @brief Step 3.5 — Camera-Match Boost (saturation + contrast).
 *
 * Direct port of Python's utils.apply_saturation_and_contrast().
 * Applies in-place saturation and contrast boost to prepare
 * the image for LUT conversion.
 */

#include "raw_alchemy/common.h"

namespace rawalchemy {

/**
 * @brief Apply saturation and contrast boost in-place.
 *
 * Matches Python: apply_saturation_and_contrast(img, saturation=1.25, contrast=1.1)
 *
 * Per-pixel logic:
 *   1. Compute luminance: Y = Lr*R + Lg*G + Lb*B (ProPhoto RGB coefficients)
 *   2. Saturation: out = Y + (in - Y) * saturation
 *   3. Contrast:   out = (out - pivot) * contrast + pivot   (pivot = 0.18)
 *   4. Clamp to >= 0
 *
 * @param img         Image buffer (modified in-place)
 * @param saturation  Saturation multiplier (default 1.25)
 * @param contrast    Contrast multiplier (default 1.10)
 */
void applySaturationContrast(ImageBuffer& img, float saturation = 1.25f, float contrast = 1.10f);

} // namespace rawalchemy
