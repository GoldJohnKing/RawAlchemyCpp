#pragma once
/**
 * @file lens_correction.h
 * @brief Lens distortion/TCA/vignetting correction via Lensfun.
 *
 * Equivalent to the Python project's lensfun_wrapper.py + utils.apply_lens_correction().
 * Applies lens correction on linear float32 ProPhoto RGB data, AFTER decode
 * but BEFORE any exposure adjustment or color space conversion.
 *
 * Requires Lensfun library. If not compiled with RA_LENSFUN_ENABLED,
 * applyLensCorrection() is a no-op that returns false.
 */

#include "common.h"
#include "raw_decoder.h"
#include <string>

namespace rawalchemy {

/// Configuration for lens correction — mirrors Python project's parameters
struct LensCorrectionParams {
    /// Enable lens correction at all
    bool enabled = true;

    /// Correct barrel/pincushion distortion
    bool correctDistortion = true;

    /// Correct lateral chromatic aberration (TCA)
    bool correctTca = true;

    /// Correct vignetting (requires aperture + distance)
    bool correctVignetting = true;

    /// Focus distance in meters (for vignetting correction)
    float distance = 1000.0f;

    /// Optional path to a custom Lensfun XML database file/directory
    std::string customDbPath;
};

/**
 * @brief Apply lens correction to an image in-place.
 *
 * The image must be linear float32 RGB, typically ProPhoto RGB from Step 1.
 * Camera and lens metadata are used to look up correction profiles from
 * the Lensfun database.
 *
 * Processing order (matches Lensfun's recommended pipeline):
 *   1. Vignetting correction (in-place color modification)
 *   2. Distortion + TCA correction (coordinate remapping with bicubic interpolation)
 *
 * @param img       The image to correct (modified in-place)
 * @param meta      Camera/lens metadata from extractMetadata()
 * @param params    Correction configuration
 * @return true if correction was applied, false if skipped (no Lensfun, lens not found, etc.)
 */
bool applyLensCorrection(ImageBuffer& img,
                         const CameraMetadata& meta,
                         const LensCorrectionParams& params = LensCorrectionParams{});

} // namespace rawalchemy
