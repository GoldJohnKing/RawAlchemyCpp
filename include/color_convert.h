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

/// Apply camera-RGB → output-RGB color matrix (3x3 subset of LibRaw's rgb_cam[3][4]).
/// Operates in-place on planar RGB float data: [R plane | G plane | B plane].
///
/// LibRaw populates imgdata.color.rgb_cam[3][4] where the 4 columns are R,G,G,B
/// (two greens). After LibRaw's mix_green, both green columns are identical,
/// so we use only columns 0,1,2 (R,G,B).
///
/// @param planarRgb  float[3*w*h], layout: [R(w*h) | G(w*h) | B(w*h)]
/// @param w, h       image dimensions
/// @param rgb_cam    3x4 matrix from LibRaw (we use the 3x3 R,G,B subset)
inline void applyCameraToProPhoto(float* planarRgb, int w, int h,
                                   const float rgb_cam[3][4]) {
    const size_t n = static_cast<size_t>(w) * h;
    const float m00 = rgb_cam[0][0], m01 = rgb_cam[0][1], m02 = rgb_cam[0][2];
    const float m10 = rgb_cam[1][0], m11 = rgb_cam[1][1], m12 = rgb_cam[1][2];
    const float m20 = rgb_cam[2][0], m21 = rgb_cam[2][1], m22 = rgb_cam[2][2];

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
