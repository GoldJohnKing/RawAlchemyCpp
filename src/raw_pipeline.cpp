/**
 * @file raw_pipeline.cpp
 * @brief Shared custom CPU demosaic pipeline (Phases 1-5) — used by BOTH the
 *        CLI (src/main.cpp) and the C API (src/raw_alchemy_capi.cpp).
 *
 * Single source of truth for the 7-stage custom RAW decode. Extracted here so
 * the CLI and C API produce identical output and share EXIF handling.
 */

#include "raw_pipeline.h"

#include "raw_mosaic.h"
#include "raw_preprocess.h"
#include "highlight.h"
#include "demosaic.h"
#include "raw_postprocess.h"
#include "colorspace_matrices.h"
#include "raw_decoder.h"

namespace rawalchemy {

ImageBuffer decodeImageWithCustomPipeline(
    const std::string& inputPath,
    ExifCollector* exifCollector) {
    // Decode the packed CFA mosaic. The exifCollector is passed THROUGH so the
    // EXIF callback fires during open_file (single collection — JPEG output
    // retains EXIF). Non-CFA sensors throw here (raw_image==nullptr) and
    // propagate; callers route isNonCfa files to decodeRaw() first.
    auto mosaic = decodeRawMosaic(inputPath, exifCollector);

    // Phase 1: black-level subtraction + hot-pixel fix.
    subtractBlackLevel(mosaic);
    fixHotPixels(mosaic);

    // Phase 2: highlight reconstruction (inpaint-opposed).
    highlightInpaintOpposed(mosaic);

    // Phase 3-4: demosaic (RCD on Bayer, Markesteijn 1-pass on X-Trans).
    ImageBuffer img = (mosaic.filters == 9)
        ? xtransMarkesteijnDemosaic(mosaic)
        : rcdDemosaic(mosaic);

    // Phase 5: WB multiply (green-anchored) + orientation flip + camera->ProPhoto.
    applyWhiteBalance(img, mosaic.cam_mul);
    applyFlip(img, mosaic.flip);

    // Camera -> ProPhoto RGB matrix. cameraToProphotoMatrix() returns
    // std::array<std::array<float,3>,3>, which applyColorMatrix now consumes
    // directly (no float[3][3] cast needed).
    auto M = cameraToProphotoMatrix(mosaic);
    applyColorMatrix(img, M);

    // Clip to [0,1] — matches the Python reference (core.py) and the dcraw
    // path's uint16->float normalization. ImageBuffer::clamp() (common.h) is
    // the canonical helper.
    img.clamp();
    return img;
}

} // namespace rawalchemy
