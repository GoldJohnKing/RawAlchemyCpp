// SPDX-License-Identifier: AGPL-3.0-or-later
// Image-processing primitives for the segmentation-based highlight reconstruction.
// Faithful ports of darktable utilities used by segbased.c:
//   - distanceTransform:   common/distance_transform.c (Felzenszwalb-Huttenlocher)
//   - gaussianBlur1ch:     common/gaussian.c dt_gaussian_fast_blur (ch=1)
//   - boxMean1ch:          common/box_filters.cc dt_box_mean (ch=1)
//   - scharrGradient:      common/math.h
//   - splitmix32/xoshiro128plus/poissonNoise: develop/noise_generator.h
// Adapted to C++ (std::vector, no GLib/darktable alloc). Algorithms copied 1:1;
// do not "improve" without re-deriving against upstream.
#pragma once

#include <cstdint>

namespace rawalchemy {

// Felzenszwalb-Huttenlocher squared-Euclidean distance transform.
// `out` is pre-filled by the caller: 0.0f = background, kDtMax = source pixel.
// Computes, for each pixel, the squared distance to the nearest source; takes sqrt
// in the final row pass. Returns the max (unsquared) distance found.
constexpr float kDtMax = 1e20f;
float distanceTransform(float* out, int width, int height);

// Separable Gaussian blur, single channel. Equivalent to darktable's
// dt_gaussian_fast_blur(src, dst, w, h, sigma, min, max, ch=1) — the min/max
// clamping is unused on the fast 9×9 path for ch=1, so omitted here.
void gaussianBlur1ch(const float* src, float* dst, int W, int H, float sigma);

// Separable box blur, single channel, N iterations. Equivalent to darktable's
// dt_box_mean(buf, h, w, ch=1, radius, iterations). Operates in place.
void boxMean1ch(float* data, int W, int H, int radius, int iterations);

// Scharr gradient magnitude at pixel p (3×3). p points at the center; stride = row width.
// Verbatim from darktable common/math.h (dt_fast_hypotf → std::hypot).
inline float scharrGradient(const float* p, int stride) {
    const float gx = 47.0f / 255.0f * (p[-stride - 1] - p[-stride + 1] + p[stride - 1] - p[stride + 1])
                   + 162.0f / 255.0f * (p[-1] - p[1]);
    const float gy = 47.0f / 255.0f * (p[-stride - 1] - p[stride - 1] + p[-stride + 1] - p[stride + 1])
                   + 162.0f / 255.0f * (p[-stride] - p[stride]);
    const float gx2 = gx * gx;
    const float gy2 = gy * gy;
    // darktable uses dt_fast_hypotf; std::hypot is the precise equivalent.
    return gx2 + gy2 > 0.0f ? __builtin_sqrt(gx2 + gy2) : 0.0f;
}

// PRNG: xoshiro128+ family. Verbatim from darktable develop/noise_generator.h.
using RngState = uint32_t[4];
uint32_t splitmix32(uint64_t seed);
float xoshiro128plus(RngState state);
float poissonNoise(float mu, float sigma, int flip, RngState state);

} // namespace rawalchemy
