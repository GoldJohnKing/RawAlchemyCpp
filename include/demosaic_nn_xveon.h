// SPDX-License-Identifier: AGPL-3.0-or-later
// Dispatch entry point + tile loop for the x-veon NN demosaic.
// Ties together the ORT session (nn_session), preprocessing (nn_preprocess),
// postprocessing (nn_postprocess) and the NaN guard (nn_nan_guard) into one
// pipeline that runs on a CFA mosaic:
//   normalize -> per-site WB -> phase-align/mirror-pad -> tile -> infer (ORT)
//   -> trapezoid blend -> crop -> camRGB->sRGB.
// Algorithm: see docs/nn-demosaic-design.md §2.3-§2.4.
//
// ORT is PIMPL-isolated: this header does NOT include any ORT header. The
// single entry point takes a plain-CFA input bundle and fills an
// NnDemosaicOutput struct; callers never touch Ort:: types.
#pragma once

#include <vector>

namespace rawalchemy {

/** Input bundle for nnDemosaic(). All buffers are caller-owned and must outlive
 *  the call. `cfaMosaic` is read-only. */
struct NnDemosaicInput {
    int width = 0;          // CFA width in pixels (active area)
    int height = 0;         // CFA height in pixels (active area)
    unsigned filters = 0;   // LibRaw CFA bitmask (filters == 9 marks X-Trans)
    const float* cfaMosaic = nullptr;  // [width*height], raw CFA (pre-normalization)

    float blackLevel = 0.0f;     // single black point for all sites (design §2.3 step 3)
    float whiteLevel = 1.0f;     // single white point; >blackLevel or buffer is zeroed

    // Per-channel white balance multipliers (G normalized to ~1.0 by convention).
    float wbRgb[3] = {1.0f, 1.0f, 1.0f};

    // Camera's XYZ->camRGB matrix (row-major [3x3]), as delivered by
    // LibRaw imgdata.color.cam_xyz. Used to derive the camRGB->sRGB matrix.
    float xyzToCam[9] = {1.0f, 0.0f, 0.0f,
                         0.0f, 1.0f, 0.0f,
                         0.0f, 0.0f, 1.0f};

    /** If true, skip the camRGB->sRGB color matrix + low-side clamp and output
     *  the model's raw camRGB (white-balanced, linear). The caller applies the
     *  camera matrix themselves. Default false (sRGB output for backward compat).
     *  Use true when feeding a ProPhoto/vLog/LUT pipeline to avoid sRGB gamut clip. */
    bool outputCamRgb = false;
};

/** Output bundle for nnDemosaic(). `rgbInterleaved` is [width*height*3],
 *  linear sRGB with highlights low-clamped to the supported range. */
struct NnDemosaicOutput {
    std::vector<float> rgbInterleaved;  // [width*height*3], linear sRGB, low-clamped
    int width = 0;
    int height = 0;
};

/** Outcome of nnDemosaic(). Design §6: two reliability mechanisms exist —
 *  "session init failure -> permanent fallback" (SessionNotReady) and
 *  "NaN/Inf in output -> error, no fallback" (NaNOutput). */
enum class NnDemosaicStatus {
    Ok,
    SessionNotReady,
    NaNOutput,         // inference output contained NaN/Inf (design §6.2: error, do not fall back)
    InferenceFailed,   // Ort::Exception during Run (non-NaN); treat like init failure upstream
    InvalidParam       // null pointers, zero dimensions, etc.
};

/** Run the full x-veon NN demosaic pipeline.
 *
 *  @param in   CFA + metadata bundle (see NnDemosaicInput).
 *  @param out  filled with linear sRGB on Ok (rgbInterleaved resized to
 *              width*height*3, width/height set); contents undefined otherwise.
 *  @return status; see NnDemosaicStatus.
 *
 *  Thread-safety: the underlying ORT Session::Run is thread-safe (per ORT's
 *  contract) and this function parallelizes inference across tiles with OpenMP.
 *  The NnDemosaicSession singleton must be init()'d once from a single thread
 *  before any call to nnDemosaic(). */
NnDemosaicStatus nnDemosaic(const NnDemosaicInput& in, NnDemosaicOutput& out);

} // namespace rawalchemy
