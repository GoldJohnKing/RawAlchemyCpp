#pragma once
/**
 * @file grading_fused.h
 * CameraFTP - A Cross-platform FTP companion for camera photo transfer
 * Copyright (C) 2026 GoldJohnKing <GoldJohnKing@Live.cn>
 * SPDX-License-Identifier: AGPL-3.0-or-later
 *
 * @brief Single-pass fused grading pipeline: gain → saturation/contrast → gamut → log → LUT.
 *
 * Fuses 5 separate image traversals into 1 pass, reducing memory bandwidth
 * from ~2,740 MB to ~548 MB per 24MP image (5× reduction).
 *
 * Output is visually identical to the separate-pass version. The only difference
 * is FMA contraction giving ≤ 2 ULP better precision in the gamut matrix multiply,
 * which is 5 orders of magnitude below LUT grid resolution and therefore invisible.
 */

#include "common.h"
#include "color_data.h"
#include "lut_applier.h"
#include <string>

namespace rawalchemy {

/// Parameters for the fused grading pipeline.
/// All parameters are optional — steps are skipped when their trigger is absent.
struct GradingParams {
    // --- Exposure ---
    float gain = 1.0f;           ///< Exposure gain multiplier (computeAutoGain result × 2^evOffset)

    // --- Saturation / Contrast ---
    bool  enableBoost = true;    ///< Apply saturation + contrast boost?
    float saturation  = 1.25f;   ///< Saturation multiplier
    float contrast    = 1.10f;   ///< Contrast multiplier
    float pivot       = 0.18f;   ///< Contrast pivot point

    // --- Gamut + Log ---
    const LogSpaceInfo* logSpaceInfo = nullptr;  ///< nullptr = skip gamut + log encoding

    // --- LUT ---
    const LUT3D* lut = nullptr;  ///< nullptr = skip LUT
};

/**
 * @brief Apply the complete grading pipeline in a single pass.
 *
 * Execution order per pixel:
 *   1. Gain:            r *= gain, g *= gain, b *= gain
 *   2. Sat + Contrast:  lum-based saturation + pivot contrast, clamp >= 0
 *   3. Gamut transform: 3×3 matrix multiply (ProPhoto → target gamut)
 *   4. Log encode:      clamp >= 1e-6, apply log OETF
 *   5. LUT:             tetrahedral interpolation
 *
 * Steps are skipped when their params are default/off (gain=1, enableBoost=false,
 * logSpaceInfo=nullptr, lut=nullptr).
 *
 * @param img    Image buffer (modified in-place)
 * @param params Grading parameters
 */
void applyGradingFused(ImageBuffer& img, const GradingParams& params);

} // namespace rawalchemy
