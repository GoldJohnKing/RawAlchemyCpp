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
#include <cstdint>
#include <vector>

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

/**
 * @brief Two-stage safety scale + uniform OOB mask for the lens coord buffer.
 *
 * Direct C++ port of lensfun_wrapper.py:851-903. Operates on the coord buffer
 * IN PLACE. The coord buffer layout is [pixel][channel][xy] (per pixel:
 * [R_x, R_y, G_x, G_y, B_x, B_y]), which matches the reference's numpy
 * (H, W, 3, 2) row-major reshape exactly.
 *
 * Pipeline (all formulas verbatim from the Python reference):
 *   1. Stage-A auto-crop: if any coord falls outside [0,W)x[0,H), scale all
 *      coords about the image center so the bounding box fits.
 *   2. Stage-B safety: with interp_margin=2.0, scale about the center if any
 *      coord falls within interp_margin of any border.
 *   3. Uniform OOB mask: per pixel, 1 if ANY of the 3 channels is still within
 *      interp_margin of a border (this is the critical anti-fringing change —
 *      all 3 channels get zeroed together where any is OOB).
 *   4. Clamp all coords to [0, W-1] / [0, H-1].
 *
 * Pure coordinate math — no Lensfun dependency. Always available regardless
 * of RA_LENSFUN_ENABLED.
 *
 * @param coords        Float buffer of size W*H*6 (modified in place).
 * @param W             Image width.
 * @param H             Image height.
 * @param interp_margin Safety margin (default 2.0, matching the reference).
 * @return Uniform OOB mask (W*H bytes; 1 if ANY channel is OOB).
 */
std::vector<uint8_t> postProcessLensCoords(float* coords, int W, int H,
                                           float interp_margin = 2.0f);

} // namespace rawalchemy
