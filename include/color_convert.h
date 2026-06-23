// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file color_convert.h
 * @brief Color matrix multiplication and planar/interleaved layout conversion.
 *
 * Bridges LibRaw's storage layouts (4-channel image[], 3x4 camera matrix) to
 * the layouts consumed by the demosaic kernels and the output ImageBuffer.
 */

#include "common.h"
#include "cfa_lookup.h"
#include "aligned_allocator.h"

#include <cstddef>

namespace rawalchemy {

/// ProPhoto RGB primaries in XYZ (D65-adapted).
/// Matches LibRaw's LibRaw_constants::prophoto_rgb (colorconst.cpp:38-41).
/// LibRaw's convert_to_rgb() multiplies this by rgb_cam to produce the
/// final camera→ProPhoto matrix (postprocessing_utils_dcrdefs.cpp:98-101).
static constexpr double PROPHOTO_RGB[3][3] = {
    {0.529317, 0.330092, 0.140588},
    {0.098368, 0.873465, 0.028169},
    {0.016879, 0.117663, 0.865457}
};

/// Compute camera→ProPhoto matrix matching LibRaw's convert_to_rgb().
/// rgb_cam alone maps camera→sRGB; multiplying by prophoto_rgb converts
/// to ProPhoto, which is what the downstream pipeline expects.
inline void computeProPhotoMatrix(const float rgb_cam[3][4], float out_cam[3][4]) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            out_cam[i][j] = 0.0f;
            for (int k = 0; k < 3; k++)
                out_cam[i][j] += static_cast<float>(PROPHOTO_RGB[i][k] * rgb_cam[k][j]);
        }
    for (int i = 0; i < 3; i++)
        out_cam[i][3] = rgb_cam[i][3];
}

/// Apply a 3×3 color matrix (from a 3×4 array) to planar RGB data, in-place.
inline void applyCameraToProPhoto(float* planarRgb, int w, int h,
                                   const float matrix[3][4]) {
    const size_t n = static_cast<size_t>(w) * h;
    const float m00 = matrix[0][0], m01 = matrix[0][1], m02 = matrix[0][2];
    const float m10 = matrix[1][0], m11 = matrix[1][1], m12 = matrix[1][2];
    const float m20 = matrix[2][0], m21 = matrix[2][1], m22 = matrix[2][2];

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        const float r = planarRgb[i];
        const float g = planarRgb[i + n];
        const float b = planarRgb[i + 2 * n];
        planarRgb[i]         = m00 * r + m01 * g + m02 * b;
        planarRgb[i + n]     = m10 * r + m11 * g + m12 * b;
        planarRgb[i + 2 * n] = m20 * r + m21 * g + m22 * b;
    }
}

/// Convert planar RGB [R plane | G plane | B plane] to interleaved ImageBuffer.
/// Output ImageBuffer has 3 channels, row-major RGBRGBRGB...
inline ImageBuffer interleavePlanarRgb(const float* planarRgb, int w, int h) {
    ImageBuffer out(w, h, 3);
    const size_t n = static_cast<size_t>(w) * h;
    for (size_t i = 0; i < n; ++i) {
        out.data[i * 3 + 0] = planarRgb[i];
        out.data[i * 3 + 1] = planarRgb[i + n];
        out.data[i * 3 + 2] = planarRgb[i + 2 * n];
    }
    return out;
}

/// Extract single-channel CFA mosaic from LibRaw's 4-channel image[] buffer.
/// Each pixel of image[] has 4 floats (RGBG), but only one is valid (the CFA position).
///
/// The channel index returned by bayerColor()/xtransColor() is used directly as
/// the image[pixel][c] subscript: R=0, Gr=1, B=2, Gb=3 (matches LibRaw's layout).
///
/// @param image     LibRaw imgdata.image (cast to float(*)[4])
/// @param w, h      image dimensions (imgdata.sizes.width/height)
/// @param filters   LibRaw CFA bitmask (for Bayer) or 9 (for X-Trans)
/// @param xtrans    X-Trans pattern (only used if filters == 9)
/// @param out       output buffer, float[w*h], single-channel mosaic
inline void extractCfaFromImage(const float (*image)[4], int w, int h,
                                 unsigned filters, const unsigned char xtrans[6][6],
                                 float* out) {
    if (isXtrans(filters)) {
        for (int row = 0; row < h; ++row) {
            for (int col = 0; col < w; ++col) {
                unsigned c = xtransColor(row, col, xtrans);
                out[row * w + col] = image[row * w + col][c];
            }
        }
    } else {
        for (int row = 0; row < h; ++row) {
            for (int col = 0; col < w; ++col) {
                unsigned c = bayerColor(row, col, filters);
                out[row * w + col] = image[row * w + col][c];
            }
        }
    }
}

} // namespace rawalchemy
