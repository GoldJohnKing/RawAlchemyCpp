#pragma once
/**
 * @file log_transform.h
 * @brief Core Philosophy Step 2 — Precise Log Signal Preparation.
 *
 * Converts linear ProPhoto RGB image data into the target Log color space:
 *   1. Gamut Transform: ProPhoto RGB (Linear) -> Target Gamut (Linear)
 *   2. Log Curve Encoding: Linear -> Log (e.g., F-Log2, S-Log3, etc.)
 *
 * This precisely simulates the process of generating Log video signals
 * inside a camera, ensuring color consistency with LUTs designed for
 * specific Log workflows.
 */

#include "common.h"
#include "color_data.h"
#include <string>
#include <vector>

namespace rawalchemy {

/// Get list of supported log space names (for CLI help)
std::vector<std::string> getSupportedLogSpaces();

/// Check if a log space name is supported
bool isLogSpaceSupported(const std::string& name);

/**
 * @brief Apply gamut transform: ProPhoto RGB (Linear) -> Target Gamut (Linear).
 *
 * In-place 3x3 matrix multiplication on every pixel.
 * Matches Python: colour.matrix_RGB_to_RGB(ProPhoto, TargetGamut)
 *
 * @param img       Image buffer (modified in-place)
 * @param logSpace  Target log space name (e.g., "F-Log2", "S-Log3")
 * @return true on success, false if log space not found
 */
bool applyGamutTransform(ImageBuffer& img, const std::string& logSpace);

/**
 * @brief Apply log curve encoding: Linear -> Log.
 *
 * In-place application of the log OETF (Opto-Electronic Transfer Function).
 * Matches Python: colour.cctf_encoding(img, function=logCurveName)
 *
 * Values are clamped to >= 1e-6 before log encoding to prevent log(0) or log(negative).
 *
 * @param img       Image buffer (modified in-place)
 * @param logSpace  Target log space name
 * @return true on success
 */
bool applyLogEncoding(ImageBuffer& img, const std::string& logSpace);

#if defined(__aarch64__)
struct HalfImageBuffer;  // forward declaration (defined in half_buffer.h)

/**
 * @brief ARM64 float16-optimized log curve encoding.
 *
 * Same semantics as applyLogEncoding() but operates on float16 image data.
 * Memory traffic is halved vs float32. Math is performed in float32 after
 * NEON register-level conversion.
 *
 * Available only on aarch64. On other platforms, use applyLogEncoding().
 */
bool applyLogEncodingF16(HalfImageBuffer& img, const std::string& logSpace);
#endif

/**
 * @brief Complete Step 2 pipeline: Gamut + Log Curve.
 *
 * ProPhoto RGB (Linear) -> Target Gamut (Linear) -> Target Log Curve
 *
 * @param img       Image buffer (modified in-place)
 * @param logSpace  Target log space name
 * @return true on success
 */
bool applyLogTransform(ImageBuffer& img, const std::string& logSpace);

} // namespace rawalchemy
