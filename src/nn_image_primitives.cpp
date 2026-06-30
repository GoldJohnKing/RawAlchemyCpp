// SPDX-License-Identifier: AGPL-3.0-or-lator
// Image-processing primitives — see nn_image_primitives.h.
// Faithful ports from darktable; algorithms copied 1:1.

#include "nn_image_primitives.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace rawalchemy {
namespace {

constexpr float kTwoPi = 6.28318530717958647692f;

} // namespace

// ---------------------------------------------------------------------------
// Distance transform — Felzenszwalb & Huttenlocher, "Distance Transforms of
// Sampled Functions" (TR2004-1963). Verbatim port of darktable's
// _image_distance_transform (common/distance_transform.c:59-91) + the 2D wrapper.
// ---------------------------------------------------------------------------
namespace {

// 1-D parabola-envelope pass over n samples. f = input values, z = envelope
// intersection abscissae, d = output distances, v = parabola indices.
void dt1d(const float* f, float* z, float* d, int* v, int n) {
    int k = 0;
    v[0] = 0;
    z[0] = -kDtMax;
    z[1] = kDtMax;
    for (int q = 1; q <= n - 1; ++q) {
        float s = (f[q] + (float)(q * q)) - (f[v[k]] + (float)(v[k] * v[k]));
        while (s <= z[k] * (float)(2 * q - 2 * v[k])) {
            --k;
            s = (f[q] + (float)(q * q)) - (f[v[k]] + (float)(v[k] * v[k]));
        }
        s /= (float)(2 * q - 2 * v[k]);
        ++k;
        v[k] = q;
        z[k] = s;
        z[k + 1] = kDtMax;
    }
    k = 0;
    for (int q = 0; q <= n - 1; ++q) {
        while (z[k + 1] < (float)q) ++k;
        d[q] = (float)((q - v[k]) * (q - v[k])) + f[v[k]];
    }
}

} // namespace

float distanceTransform(float* out, int width, int height) {
    const int maxdim = std::max(width, height);
    std::vector<float> f(maxdim);
    std::vector<float> z(maxdim + 1);
    std::vector<float> d(maxdim);
    std::vector<int> v(maxdim);
    float maxDistance = 0.0f;

    // Transform along columns.
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y)
            f[y] = out[(size_t)y * width + x];
        dt1d(f.data(), z.data(), d.data(), v.data(), height);
        for (int y = 0; y < height; ++y)
            out[(size_t)y * width + x] = d[y];
    }
    // Transform along rows (take sqrt + track max here).
    for (int y = 0; y < height; ++y) {
        dt1d(out + (size_t)y * width, z.data(), d.data(), v.data(), width);
        for (int x = 0; x < width; ++x) {
            const float val = std::sqrt(d[x]);
            out[(size_t)y * width + x] = val;
            if (val > maxDistance) maxDistance = val;
        }
    }
    return maxDistance;
}

// ---------------------------------------------------------------------------
// Gaussian blur — separable kernel. darktable's dt_gaussian_fast_blur uses a
// 9-tap separable kernel for σ≈1.2; this computes the kernel dynamically for any
// σ (radius = ceil(3σ)) and applies it in two passes. Algorithmically equivalent
// for the σ=1.2 case segbased uses.
// ---------------------------------------------------------------------------
void gaussianBlur1ch(const float* src, float* dst, int W, int H, float sigma) {
    if (sigma <= 0.0f || W <= 0 || H <= 0) {
        if (src != dst) std::copy_n(src, (size_t)W * H, dst);
        return;
    }
    const int radius = std::max(1, (int)std::ceil(3.0f * sigma));
    const int ksize = 2 * radius + 1;
    std::vector<float> kernel(ksize);
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        const float v = std::exp(-0.5f * (float)(i * i) / (sigma * sigma));
        kernel[i + radius] = v;
        sum += v;
    }
    const float invSum = 1.0f / sum;
    for (auto& k : kernel) k *= invSum;

    std::vector<float> tmp((size_t)W * H);
    // Horizontal pass (clamp at borders).
    for (int y = 0; y < H; ++y) {
        const float* srow = src + (size_t)y * W;
        float* trow = tmp.data() + (size_t)y * W;
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int xx = x + k;
                if (xx < 0) xx = 0; else if (xx >= W) xx = W - 1;
                acc += kernel[k + radius] * srow[xx];
            }
            trow[x] = acc;
        }
    }
    // Vertical pass (clamp at borders).
    for (int y = 0; y < H; ++y) {
        float* drow = dst + (size_t)y * W;
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int yy = y + k;
                if (yy < 0) yy = 0; else if (yy >= H) yy = H - 1;
                acc += kernel[k + radius] * tmp[(size_t)yy * W + x];
            }
            drow[x] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Box blur — separable sliding-window, N iterations. darktable's dt_box_mean
// (ch=1). Operates in place; uses a scratch row buffer.
// ---------------------------------------------------------------------------
void boxMean1ch(float* data, int W, int H, int radius, int iterations) {
    if (radius <= 0 || iterations <= 0 || W <= 0 || H <= 0) return;
    const float invDenom = 1.0f / (float)(2 * radius + 1);
    std::vector<float> tmp((size_t)W * H);

    for (int iter = 0; iter < iterations; ++iter) {
        // Horizontal pass into tmp.
        for (int y = 0; y < H; ++y) {
            const float* srow = data + (size_t)y * W;
            float* trow = tmp.data() + (size_t)y * W;
            float sum = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int xx = k < 0 ? 0 : (k >= W ? W - 1 : k);
                sum += srow[xx];
            }
            for (int x = 0; x < W; ++x) {
                trow[x] = sum * invDenom;
                const int xOut = (x - radius < 0) ? 0 : (x - radius >= W ? W - 1 : x - radius);
                const int xIn = (x + radius + 1 < 0) ? 0 : (x + radius + 1 >= W ? W - 1 : x + radius + 1);
                sum += srow[xIn] - srow[xOut];
            }
        }
        // Vertical pass back into data.
        for (int x = 0; x < W; ++x) {
            float sum = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int yy = k < 0 ? 0 : (k >= H ? H - 1 : k);
                sum += tmp[(size_t)yy * W + x];
            }
            for (int y = 0; y < H; ++y) {
                data[(size_t)y * W + x] = sum * invDenom;
                const int yOut = (y - radius < 0) ? 0 : (y - radius >= H ? H - 1 : y - radius);
                const int yIn = (y + radius + 1 < 0) ? 0 : (y + radius + 1 >= H ? H - 1 : y + radius + 1);
                sum += tmp[(size_t)yIn * W + x] - tmp[(size_t)yOut * W + x];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PRNG — verbatim from darktable develop/noise_generator.h.
// ---------------------------------------------------------------------------
static inline uint32_t rol32(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

uint32_t splitmix32(uint64_t seed) {
    uint64_t result = (seed ^ (seed >> 33)) * 0x62a9d9ed799705f5ull;
    result = (result ^ (result >> 28)) * 0xcb24d0a5c88c35b3ull;
    return (uint32_t)(result >> 32);
}

float xoshiro128plus(RngState state) {
    const uint32_t result = state[0] + state[3];
    const uint32_t t = state[1] << 9;
    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];
    state[2] ^= t;
    state[3] = rol32(state[3], 11);
    // Take the first 24 bits into the mantissa → float in [0,1).
    return (float)(result >> 8) * 0x1.0p-24f;
}

float poissonNoise(float mu, float sigma, int flip, RngState state) {
    // Gaussian noise via Box-Muller, then Anscombe transform back to Poisson.
    const float u1 = std::max(xoshiro128plus(state), 1.0e-37f);  // FLT_MIN
    const float u2 = xoshiro128plus(state);
    const float noise = flip
        ? std::sqrt(-2.0f * std::log(u1)) * std::cos(kTwoPi * u2)
        : std::sqrt(-2.0f * std::log(u1)) * std::sin(kTwoPi * u2);
    const float r = noise * sigma + 2.0f * std::sqrt(std::max(mu + 3.0f / 8.0f, 0.0f));
    return (r * r - sigma * sigma) / 4.0f - 3.0f / 8.0f;
}

} // namespace rawalchemy
