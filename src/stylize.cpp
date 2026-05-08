/**
 * @file stylize.cpp
 * @brief Step 3.5 — Camera-Match Boost.
 *
 * Direct port of Python's utils.apply_saturation_contrast_inplace() Numba kernel.
 * OpenMP parallelized for speed.
 */

#include "stylize.h"
#include "metering.h"  // for ProPhoto luma coefficients
#include <algorithm>

#ifdef RA_USE_OPENMP
#include <omp.h>
#endif

namespace rawalchemy {

void applySaturationContrast(ImageBuffer& img, float saturation, float contrast) {
    const size_t nPixels = img.pixelCount();
    float* data = img.ptr();
    constexpr float pivot = 0.18f;

    const float cr = PROPHOTO_LUMA_R;
    const float cg = PROPHOTO_LUMA_G;
    const float cb = PROPHOTO_LUMA_B;

    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static, 8192)
    #endif
    for (size_t i = 0; i < nPixels; i++) {
        float* p = data + i * 3;
        float r = p[0], g = p[1], b = p[2];

        // 1. Luminance
        float lum = cr * r + cg * g + cb * b;

        // 2. Saturation: out = lum + (in - lum) * sat
        float rs = lum + (r - lum) * saturation;
        float gs = lum + (g - lum) * saturation;
        float bs = lum + (b - lum) * saturation;

        // 3. Contrast: out = (in - pivot) * cont + pivot
        float rf = (rs - pivot) * contrast + pivot;
        float gf = (gs - pivot) * contrast + pivot;
        float bf = (bs - pivot) * contrast + pivot;

        // 4. Clamp to >= 0
        p[0] = std::max(0.0f, rf);
        p[1] = std::max(0.0f, gf);
        p[2] = std::max(0.0f, bf);
    }
}

} // namespace rawalchemy
