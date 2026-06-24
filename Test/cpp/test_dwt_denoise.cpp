// SPDX-License-Identifier: AGPL-3.0-or-later
// Tests for the single-channel à trous wavelet denoiser (port of darktable dwt.c).
#include "../../include/dwt_denoise.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace {
// Mean squared high-frequency energy: average squared difference between
// horizontally adjacent pixels. A smooth signal has low HFE; pure noise has high.
double horizontalHFE(const std::vector<float>& v, int w, int h) {
    double e = 0.0;
    long c = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 1; x < w; ++x) {
            const float d = v[y * w + x] - v[y * w + x - 1];
            e += static_cast<double>(d) * d;
            ++c;
        }
    return e / std::max(1L, c);
}
} // namespace

int main() {
    using namespace rawalchemy;

    // Test 1: a flat signal is preserved (no detail to remove).
    {
        const int W = 64, H = 64;
        std::vector<float> img(static_cast<size_t>(W) * H, 0.5f);
        const float noise[DWT_DENOISE_BANDS] = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};
        dwt_denoise(img.data(), W, H, DWT_DENOISE_BANDS, noise);
        for (float v : img)
            assert(std::fabs(v - 0.5f) < 1e-4f);
    }

    // Test 2: high-frequency noise is substantially attenuated, smooth base kept.
    {
        const int W = 128, H = 128;
        std::vector<float> img(static_cast<size_t>(W) * H);
        std::mt19937 rng(12345);
        std::normal_distribution<float> nz(0.0f, 0.08f);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const float base = 0.2f + 0.5f * static_cast<float>(x) / W;
                img[static_cast<size_t>(y) * W + x] = base + nz(rng);
            }
        const double before = horizontalHFE(img, W, H);
        // Strong threshold so detail above the noise floor survives mostly as zero.
        const float noise[DWT_DENOISE_BANDS] = {0.4f, 0.4f, 0.4f, 0.4f, 0.4f};
        dwt_denoise(img.data(), W, H, DWT_DENOISE_BANDS, noise);
        const double after = horizontalHFE(img, W, H);
        assert(after < before * 0.5);  // at least 50% high-freq energy removed
    }

    // Test 3: a zero-threshold pass is an identity (all detail is kept).
    {
        const int W = 32, H = 32;
        std::vector<float> img(static_cast<size_t>(W) * H);
        std::vector<float> ref(static_cast<size_t>(W) * H);
        for (size_t i = 0; i < img.size(); ++i) {
            img[i] = ref[i] = static_cast<float>(i % 17) * 0.01f;
        }
        const float noise[DWT_DENOISE_BANDS] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        dwt_denoise(img.data(), W, H, DWT_DENOISE_BANDS, noise);
        for (size_t i = 0; i < img.size(); ++i)
            assert(std::fabs(img[i] - ref[i]) < 1e-4f);
    }

    std::cout << "test_dwt_denoise: PASS\n";
    return 0;
}
