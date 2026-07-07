// SPDX-License-Identifier: AGPL-3.0-or-later
#include "nn_postprocess.h"
#include <algorithm>
#include <cmath>

namespace rawalchemy {

void makeTrapezoidWeights(float* outWeights2d) {
    // Build 1D window first
    float w1d[NN_PATCH_SIZE];
    for (int i = 0; i < NN_PATCH_SIZE; ++i) {
        if (i < NN_OVERLAP) {
            w1d[i] = static_cast<float>(i) / static_cast<float>(NN_OVERLAP);
        } else if (i >= NN_PATCH_SIZE - NN_OVERLAP) {
            w1d[i] = static_cast<float>(NN_PATCH_SIZE - 1 - i) / static_cast<float>(NN_OVERLAP);
        } else {
            w1d[i] = 1.0f;
        }
    }
    // Outer product
    for (int y = 0; y < NN_PATCH_SIZE; ++y) {
        for (int x = 0; x < NN_PATCH_SIZE; ++x) {
            outWeights2d[y * NN_PATCH_SIZE + x] = w1d[y] * w1d[x];
        }
    }
}

// Standard sRGB-D65 -> XYZ (row-major), used to derive the inverse path.
static const float SRGB_TO_XYZ[9] = {
    0.4124564f, 0.3575761f, 0.1804375f,
    0.2126729f, 0.7151522f, 0.0721750f,
    0.0193339f, 0.1191920f, 0.9503041f
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

void computeCamRgbToSrgb(float outMatrix[9], const float xyzToCam[9]) {
    // Normalize rows of xyzToCam to sum=1
    float normalized[9];
    for (int r = 0; r < 3; ++r) {
        float rowSum = xyzToCam[r * 3] + xyzToCam[r * 3 + 1] + xyzToCam[r * 3 + 2];
        float inv = (rowSum > 1e-20f) ? 1.0f / rowSum : 0.0f;
        for (int c = 0; c < 3; ++c) normalized[r * 3 + c] = xyzToCam[r * 3 + c] * inv;
    }
    float srgbToCam[9];
    matmul3(normalized, SRGB_TO_XYZ, srgbToCam);
    invert3x3(srgbToCam, outMatrix);  // = camRGB -> sRGB
}

void applyColorMatrixInPlace(float* rgb, size_t pixelCount, const float m[9]) {
    for (size_t i = 0; i < pixelCount; ++i) {
        float r = rgb[i * 3 + 0];
        float g = rgb[i * 3 + 1];
        float b = rgb[i * 3 + 2];
        float nr = m[0] * r + m[1] * g + m[2] * b;
        float ng = m[3] * r + m[4] * g + m[5] * b;
        float nb = m[6] * r + m[7] * g + m[8] * b;
        rgb[i * 3 + 0] = std::max(0.0f, nr);  // low-side clamp only
        rgb[i * 3 + 1] = std::max(0.0f, ng);
        rgb[i * 3 + 2] = std::max(0.0f, nb);
    }
}

} // namespace rawalchemy
