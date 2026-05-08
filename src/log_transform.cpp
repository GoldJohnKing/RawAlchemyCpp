/**
 * @file log_transform.cpp
 * @brief Core Philosophy Step 2 — Precise Log Signal Preparation.
 *
 * Implements:
 *   1. Gamut Transform (3x3 matrix multiply, in-place)
 *   2. Log Curve Encoding (per-pixel OETF application)
 *
 * Matches Python project's core.py Step 4:
 *   M = colour.matrix_RGB_to_RGB(ProPhoto, TargetGamut)
 *   utils.apply_matrix_inplace(img, M)
 *   np.maximum(img, 1e-6, out=img)
 *   img = colour.cctf_encoding(img, function=logCurveName)
 */

#include "raw_alchemy/log_transform.h"
#include <cstdio>
#include <algorithm>

#ifdef RA_USE_OPENMP
#include <omp.h>
#endif

namespace rawalchemy {

// ---- Utility ----

std::vector<std::string> getSupportedLogSpaces() {
    std::vector<std::string> names;
    for (const auto& kv : LOG_SPACES) {
        names.push_back(kv.first);
    }
    return names;
}

bool isLogSpaceSupported(const std::string& name) {
    return LOG_SPACES.find(name) != LOG_SPACES.end();
}

// ---- Gamut Transform (3x3 Matrix Multiply, In-Place) ----

bool applyGamutTransform(ImageBuffer& img, const std::string& logSpace) {
    auto it = LOG_SPACES.find(logSpace);
    if (it == LOG_SPACES.end()) {
        fprintf(stderr, "[LogTransform] Unknown log space: %s\n", logSpace.c_str());
        return false;
    }

    const auto& M = *(it->second.gamutMatrix);
    const float m00 = M[0][0], m01 = M[0][1], m02 = M[0][2];
    const float m10 = M[1][0], m11 = M[1][1], m12 = M[1][2];
    const float m20 = M[2][0], m21 = M[2][1], m22 = M[2][2];

    const size_t nPixels = img.pixelCount();
    float* data = img.ptr();

    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static, 8192)
    #endif
    for (size_t i = 0; i < nPixels; i++) {
        float* p = data + i * 3;
        float r = p[0], g = p[1], b = p[2];

        p[0] = r * m00 + g * m01 + b * m02;
        p[1] = r * m10 + g * m11 + b * m12;
        p[2] = r * m20 + g * m21 + b * m22;
    }

    return true;
}

// ---- Log Curve Encoding (Per-Pixel OETF) ----

bool applyLogEncoding(ImageBuffer& img, const std::string& logSpace) {
    auto it = LOG_SPACES.find(logSpace);
    if (it == LOG_SPACES.end()) {
        fprintf(stderr, "[LogTransform] Unknown log space: %s\n", logSpace.c_str());
        return false;
    }

    LogCurve curve = it->second.curve;
    const size_t nPixels = img.pixelCount();
    float* data = img.ptr();

    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static, 8192)
    #endif
    for (size_t i = 0; i < nPixels; i++) {
        float* p = data + i * 3;

        // Clamp to 1e-6 to prevent log(0) / log(negative)
        // Matches: np.maximum(img, 1e-6, out=img)
        float r = std::max(p[0], 1e-6f);
        float g = std::max(p[1], 1e-6f);
        float b = std::max(p[2], 1e-6f);

        p[0] = logEncode(r, curve);
        p[1] = logEncode(g, curve);
        p[2] = logEncode(b, curve);
    }

    return true;
}

// ---- Complete Pipeline ----

bool applyLogTransform(ImageBuffer& img, const std::string& logSpace) {
    // Step 2a: Gamut Transform (ProPhoto Linear -> Target Gamut Linear)
    if (!applyGamutTransform(img, logSpace)) {
        return false;
    }

    // Step 2b: Log Curve Encoding (Linear -> Log)
    if (!applyLogEncoding(img, logSpace)) {
        return false;
    }

    return true;
}

} // namespace rawalchemy
