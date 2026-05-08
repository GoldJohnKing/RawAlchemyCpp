#pragma once
/**
 * @file lut_applier.h
 * @brief Core Philosophy Step 3 — Correct LUT Application.
 *
 * Loads a .cube 3D LUT file and applies it to the image using
 * tetrahedral interpolation — identical to the Python project's
 * Numba-accelerated apply_lut_inplace().
 *
 * Tetrahedral interpolation reads only 4 of 8 cube vertices,
 * giving higher color accuracy (especially for grayscale and skin tones)
 * and 50% less memory access than trilinear.
 */

#include "raw_alchemy/common.h"
#include <vector>
#include <string>
#include <array>

namespace rawalchemy {

/// 3D LUT data
struct LUT3D {
    int size = 0;                           // LUT dimension (e.g., 65 for 65³)
    float domainMin[3] = {0,0,0};           // Domain minimum (usually 0,0,0)
    float domainMax[3] = {1,1,1};           // Domain maximum (usually 1,1,1)
    std::vector<float> table;               // [size³ × 3] — row-major (R changes fastest)

    bool empty() const { return size == 0; }
};

/**
 * @brief Load a .cube LUT file from disk.
 *
 * Parses the standard .cube format:
 *   - Header keywords: LUT_3D_SIZE, DOMAIN_MIN, DOMAIN_MAX
 *   - N³ lines of "R G B" float values
 *
 * @param path  File path to the .cube file
 * @return LUT3D  Parsed LUT data
 * @throws std::runtime_error on parse errors
 */
LUT3D loadCubeLUT(const std::string& path);

/**
 * @brief Apply a 3D LUT to an image using tetrahedral interpolation.
 *
 * Matches the Python project's apply_lut_inplace() exactly:
 *   - 6-case tetrahedral decomposition based on dx/dy/dz ordering
 *   - Reads only 4 of 8 cube vertices per pixel
 *   - In-place modification
 *
 * @param img   Image buffer (modified in-place)
 * @param lut   3D LUT data
 */
void applyLUT3D(ImageBuffer& img, const LUT3D& lut);

} // namespace rawalchemy
