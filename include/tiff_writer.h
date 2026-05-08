#pragma once
/**
 * @file tiff_writer.h
 * @brief 16-bit TIFF image output with ZLIB compression.
 *
 * Matches the Python project's file_io._save_tiff() behavior:
 *   - 16-bit unsigned integer output
 *   - RGB photometric
 *   - ZLIB compression with horizontal differencing predictor
 */

#include "common.h"
#include <string>

namespace rawalchemy {

/**
 * @brief Save an ImageBuffer as a 16-bit TIFF file with ZLIB compression.
 *
 * @param img       Image in float32 [0.0, 1.0] — will be clamped and converted to uint16
 * @param outPath   Output file path (should end in .tif or .tiff)
 * @return true on success, false on failure
 */
bool writeTiff16(const ImageBuffer& img, const std::string& outPath);

/**
 * @brief Save an ImageBuffer as a 16-bit TIFF file (uncompressed).
 *
 * Simpler fallback if ZLIB is not available.
 */
bool writeTiff16Uncompressed(const ImageBuffer& img, const std::string& outPath);

} // namespace rawalchemy
