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
// single entry point takes a plain-CFA input bundle and writes interleaved RGB
// into a caller-provided buffer; callers never touch Ort:: types.
#pragma once

namespace rawalchemy {

/** Input bundle for nnDemosaic(). All buffers are caller-owned and must outlive
 *  the call. `cfa` is read-only; `outRgbInterleaved` (passed to nnDemosaic) is
 *  written. */
struct NnDemosaicInput {
    int width = 0;          // CFA width in pixels (active area)
    int height = 0;         // CFA height in pixels (active area)
    unsigned filters = 0;   // LibRaw CFA bitmask (filters == 9 marks X-Trans)
    const float* cfa = nullptr;  // [width*height], raw CFA (pre-normalization)

    float blackLevel = 0.0f;     // single black point for all sites (design §2.3 step 3)
    float whiteLevel = 1.0f;     // single white point; >blackLevel or buffer is zeroed

    // Per-channel white balance multipliers (G normalized to ~1.0 by convention).
    float wb[3] = {1.0f, 1.0f, 1.0f};

    // Camera's XYZ->camRGB matrix (row-major [3x3]), as delivered by
    // LibRaw imgdata.color.cam_xyz. Used to derive the camRGB->sRGB matrix.
    float xyzToCam[9] = {1.0f, 0.0f, 0.0f,
                         0.0f, 1.0f, 0.0f,
                         0.0f, 0.0f, 1.0f};
};

/** Outcome of nnDemosaic(). Design §6: only two reliability mechanisms exist —
 *  "session init failure -> permanent fallback" (SessionNotReady) and
 *  "NaN/Inf in output -> error, no fallback" (NanDetected). */
enum class NnDemosaicStatus {
    Ok,               // success
    SessionNotReady,  // NnDemosaicSession not initialized, or no model for this CFA period
    InvalidInput,     // null pointers, zero dimensions, etc.
    InferenceFailed,  // Ort::Exception during Run (non-NaN); treat like init failure upstream
    NanDetected,      // inference output contained NaN/Inf (design §6.2: error, do not fall back)
};

/** Run the full x-veon NN demosaic pipeline.
 *
 *  @param in            CFA + metadata bundle (see NnDemosaicInput).
 *  @param outRgbInterleaved  caller-owned [in.width * in.height * 3] float buffer.
 *                            Filled with linear sRGB on Ok; contents undefined otherwise.
 *  @return status; see NnDemosaicStatus.
 *
 *  Thread-safety: the underlying ORT Session::Run is thread-safe (per ORT's
 *  contract) and this function parallelizes inference across tiles with OpenMP.
 *  The NnDemosaicSession singleton must be init()'d once from a single thread
 *  before any call to nnDemosaic(). */
NnDemosaicStatus nnDemosaic(const NnDemosaicInput& in, float* outRgbInterleaved);

} // namespace rawalchemy
