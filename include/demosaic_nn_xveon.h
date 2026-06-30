// SPDX-License-Identifier: AGPL-3.0-or-later
// Dispatch entry point + tile loop for the x-veon NN demosaic.
// Ties together the ORT session (nn_session), preprocessing (nn_preprocess),
// postprocessing (nn_postprocess) and the NaN guard (nn_nan_guard) into one
// pipeline that runs on a CFA mosaic:
//   normalize -> highlight recon -> per-site WB -> phase-align/mirror-pad -> tile -> infer (ORT)
//   -> trapezoid blend -> hand off accumulators (caller crops, normalizes,
//      applies color matrix + flip).
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
    // LibRaw imgdata.color.cam_xyz. Metadata for the caller's own color
    // transform — nnDemosaic outputs raw camRGB and applies no matrix itself.
    float xyzToCam[9] = {1.0f, 0.0f, 0.0f,
                         0.0f, 1.0f, 0.0f,
                         0.0f, 0.0f, 1.0f};

    /** Camera's actual X-Trans CFA pattern [6][6] (0=R, 1=G, 2=B).
     *  Populated from LibRaw's imgdata.idata.xtrans. Used for mask generation
     *  and WB color lookup instead of a hardcoded canonical — cameras ship
     *  different rotations/phases of the X-Trans arrangement. */
    int xtransPattern[6][6] = {};
};

/** Output bundle for nnDemosaic(). Holds the trapezoid-blended accumulation over
 *  the PADDED extent plus the geometry needed to finalize it. nnDemosaic stops
 *  here (no crop/normalize) so the caller can fuse crop + weight-normalize +
 *  color matrix + orientation flip into one pass — no full-image camRGB
 *  intermediate. To recover linear white-balanced camRGB for pixel (y,x):
 *    pIdx = (y + phaseDy) * paddedW + (x + phaseDx)
 *    invW = (weightAccum[pIdx] > 0) ? 1/weightAccum[pIdx] : 0
 *    rgb  = outAccum[pIdx*3 .. pIdx*3+2] * invW   (camera-native, pre-matrix) */
struct NnDemosaicOutput {
    std::vector<float> outAccum;     // [paddedH*paddedW*3], trapezoid-blended camRGB accumulation
    std::vector<float> weightAccum;  // [paddedH*paddedW], blend-weight sum per padded pixel
    int paddedW = 0;                 // padded width (>= width + phaseDx)
    int width = 0;                   // sensor-native active width
    int height = 0;                  // sensor-native active height
    int phaseDx = 0;                 // left mirror-pad (pixel (y,x) is at padded (y+phaseDy, x+phaseDx))
    int phaseDy = 0;                 // top mirror-pad
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

/** Run the x-veon NN demosaic pipeline up to (but not including) the finalize
 *  step: CFA prep, mirror-pad, tile inference and trapezoid-blended accumulation.
 *  `out` is filled with the padded accumulators + geometry; the caller finalizes
 *  (crop + weight-normalize + color matrix + orientation) — see NnDemosaicOutput.
 *
 *  @param in   CFA + metadata bundle (see NnDemosaicInput).
 *  @param out  on Ok, outAccum/weightAccum resized over the padded extent and
 *              paddedW/width/height/phaseDx/phaseDy set; contents undefined
 *              otherwise.
 *  @return status; see NnDemosaicStatus.
 *
 *  Thread-safety: the underlying ORT Session::Run is thread-safe (per ORT's
 *  contract) and this function parallelizes inference across tiles with OpenMP.
 *  The NnDemosaicSession singleton must be init()'d once from a single thread
 *  before any call to nnDemosaic(). */
NnDemosaicStatus nnDemosaic(const NnDemosaicInput& in, NnDemosaicOutput& out);

} // namespace rawalchemy
