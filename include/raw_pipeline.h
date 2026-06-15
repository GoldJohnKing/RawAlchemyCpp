#pragma once
/**
 * @file raw_pipeline.h
 * @brief Shared custom CPU demosaic pipeline (Phases 1-5).
 *
 * Single source of truth for the full custom RAW pipeline:
 *   decodeRawMosaic -> subtractBlackLevel -> fixHotPixels ->
 *   highlightInpaintOpposed -> (rcdDemosaic | xtransMarkesteijnDemosaic) ->
 *   applyWhiteBalance -> applyFlip -> cameraToProphotoMatrix ->
 *   applyColorMatrix -> clamp [0,1].
 *
 * Used by BOTH the CLI (src/main.cpp) and the C API (src/raw_alchemy_capi.cpp),
 * so the two share an identical decode. Returns a ProPhoto RGB linear
 * ImageBuffer consumed unchanged by the downstream lens/exposure/sat/log/LUT
 * pipeline.
 */

#include "common.h"
#include "raw_decoder.h"  // ExifCollector forward declaration

#include <string>

namespace rawalchemy {

/**
 * @brief Decode + run the custom CPU demosaic pipeline (Phases 1-5).
 *
 * The exifCollector is passed THROUGH to decodeRawMosaic, which wires the EXIF
 * callback before open_file (raw_decoder.cpp, same pattern as decodeRaw) —
 * single EXIF collection, so JPEG output retains EXIF.
 *
 * Callers MUST route non-CFA / Foveon / 4-color / 2D-darkframe sensors to
 * decodeRaw()/dcraw BEFORE calling this (extractMetadata().isNonCfa). Such
 * sensors throw from decodeRawMosaic (raw_image==nullptr) here and propagate.
 *
 * @param inputPath      Path to the RAW file.
 * @param exifCollector  Optional EXIF collector (populated during decode).
 * @return ProPhoto RGB linear ImageBuffer, clamped to [0,1].
 * @throws std::runtime_error on decode/pipeline failure or non-CFA sensor.
 */
ImageBuffer decodeImageWithCustomPipeline(
    const std::string& inputPath,
    ExifCollector* exifCollector = nullptr);

} // namespace rawalchemy
