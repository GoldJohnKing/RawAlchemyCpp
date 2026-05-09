#pragma once
/**
 * @file jpeg_writer.h
 * @brief 8-bit JPEG image output (high quality, no chroma subsampling).
 *
 * Matches the Python project's file_io._save_jpeg_or_other() behavior:
 *   - 8-bit unsigned integer output (float32 * 255, clamped)
 *   - Quality 95
 *   - No chroma subsampling (4:4:4)
 *   - Accurate DCT for best quality
 *   - Optional Huffman table optimization (optimize=True in Pillow)
 *
 * Uses libjpeg-turbo (TurboJPEG 3 API) for fast, high-quality JPEG encoding.
 */

#include "common.h"
#include <string>
#include <vector>
#include <cstdint>

namespace rawalchemy {

/**
 * @brief Save an ImageBuffer as an 8-bit JPEG file.
 *
 * @param img       Image in float32 [0.0, 1.0] — will be clamped and converted to uint8
 * @param outPath   Output file path (should end in .jpg or .jpeg)
 * @param quality   JPEG quality 1-100 (default: 95)
 * @param optimize  Compute optimal Huffman tables (slower, smaller file; default: false)
 * @return true on success, false on failure
 */
bool writeJpeg(const ImageBuffer& img, const std::string& outPath,
               int quality = 95, bool optimize = false,
               const std::vector<uint8_t>* exifData = nullptr);

} // namespace rawalchemy
