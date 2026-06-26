// SPDX-License-Identifier: AGPL-3.0-or-later
// camRGB -> ProPhoto via the camera's cam_xyz matrix. PROPHOTO_FROM_XYZ is the
// canonical XYZ->ProPhoto (ROMM RGB, D50) matrix from Bruce Lindbloom's primaries.
#include "nn_color_adapt.h"
#include <cmath>

namespace rawalchemy {

// XYZ -> linear ProPhoto RGB (ROMM RGB, D50), row-major 3x3.
// Source: Bruce Lindbloom RGB/XYZ matrices for ROMM RGB (ProPhoto).
static const float PROPHOTO_FROM_XYZ[9] = {
     2.34187440f, -1.02029350f, -0.26219110f,
    -0.98060158f,  1.95377353f,  0.02682407f,
     0.02595534f, -0.09186585f,  1.26542330f
};

static void matmul3(const float a[9], const float b[9], float out[9]) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < 3; ++k) sum += a[r * 3 + k] * b[k * 3 + c];
            out[r * 3 + c] = sum;
        }
    }
}

static bool invert3x3(const float m[9], float out[9]) {
    // Cofactor / determinant inversion
    float det = m[0] * (m[4] * m[8] - m[5] * m[7])
              - m[1] * (m[3] * m[8] - m[5] * m[6])
              + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (std::fabs(det) < 1e-20f) return false;
    float invDet = 1.0f / det;
    out[0] = (m[4] * m[8] - m[5] * m[7]) * invDet;
    out[1] = (m[2] * m[7] - m[1] * m[8]) * invDet;
    out[2] = (m[1] * m[5] - m[2] * m[4]) * invDet;
    out[3] = (m[5] * m[6] - m[3] * m[8]) * invDet;
    out[4] = (m[0] * m[8] - m[2] * m[6]) * invDet;
    out[5] = (m[2] * m[3] - m[0] * m[5]) * invDet;
    out[6] = (m[3] * m[7] - m[4] * m[6]) * invDet;
    out[7] = (m[1] * m[6] - m[0] * m[7]) * invDet;
    out[8] = (m[0] * m[4] - m[1] * m[3]) * invDet;
    return true;
}

void camRgbToProPhotoLinear(float* dst, const float* camRgb, size_t pixelCount,
                            const float camXyz[9]) {
    // Normalize cam_xyz rows to sum=1 (white-point handling), then
    // M = PROPHOTO_FROM_XYZ @ inv(normalizedCamXyz). cam_xyz maps XYZ->cam,
    // so inv maps cam->XYZ, and PROPHOTO_FROM_XYZ maps XYZ->ProPhoto.
    float normalized[9];
    for (int r = 0; r < 3; ++r) {
        float rowSum = camXyz[r * 3] + camXyz[r * 3 + 1] + camXyz[r * 3 + 2];
        float inv = (rowSum > 1e-20f) ? 1.0f / rowSum : 0.0f;
        for (int c = 0; c < 3; ++c) normalized[r * 3 + c] = camXyz[r * 3 + c] * inv;
    }
    float invCam[9];
    invert3x3(normalized, invCam);  // cam -> XYZ (return value ignored: matches the
                                    // classical-path behavior on singular cam_xyz)
    float M[9];
    matmul3(PROPHOTO_FROM_XYZ, invCam, M);  // cam -> ProPhoto

    // Read all three channels into locals before writing, so dst == camRgb (in-place)
    // is safe. No clamping: preserve HDR highlights and out-of-gamut negatives.
    for (size_t i = 0; i < pixelCount; ++i) {
        const float r = camRgb[i * 3 + 0];
        const float g = camRgb[i * 3 + 1];
        const float b = camRgb[i * 3 + 2];
        dst[i * 3 + 0] = M[0] * r + M[1] * g + M[2] * b;
        dst[i * 3 + 1] = M[3] * r + M[4] * g + M[5] * b;
        dst[i * 3 + 2] = M[6] * r + M[7] * g + M[8] * b;
    }
}

} // namespace rawalchemy
