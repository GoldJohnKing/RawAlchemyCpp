// SPDX-License-Identifier: AGPL-3.0-or-later
// Tests for the X-Trans CFA wavelet denoise driver and ISO threshold curve.
#include "../../include/cfa_lookup.h"
#include "../../include/denoise_xtrans.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    using namespace rawalchemy;

    // Standard Fujifilm X-Trans ILC 6x6 pattern (same as test_cfa_lookup.cpp).
    static const char xt[6][6] = {
        { 1, 2, 1, 1, 0, 1 },
        { 0, 1, 0, 2, 1, 2 },
        { 1, 2, 1, 1, 0, 1 },
        { 1, 0, 1, 1, 2, 1 },
        { 2, 1, 2, 0, 1, 0 },
        { 1, 0, 1, 1, 2, 1 }
    };

    // ---- computeXtransDenoiseThreshold: ISO curve mirrors Bayer (raw_decoder.cpp:226-239) ----
    assert(computeXtransDenoiseThreshold(50.f,    -1.f) == 0.00f);   // ISO<=100: off
    assert(computeXtransDenoiseThreshold(100.f,   -1.f) == 0.00f);   // boundary inclusive
    assert(computeXtransDenoiseThreshold(101.f,   -1.f) == 0.01f);   // 101..400: light
    assert(computeXtransDenoiseThreshold(400.f,   -1.f) == 0.01f);   // boundary inclusive
    assert(computeXtransDenoiseThreshold(800.f,   -1.f) >  0.01f);   // ramp begins >400
    assert(std::fabs(computeXtransDenoiseThreshold(12800.f, -1.f) - 0.05f) < 1e-5f); // cap
    assert(computeXtransDenoiseThreshold(51200.f, -1.f) == 0.05f);   // clamped at cap
    // manual override: 0 = off, >0 = literal
    assert(computeXtransDenoiseThreshold(6400.f, 0.00f) == 0.00f);
    assert(computeXtransDenoiseThreshold(6400.f, 0.03f) == 0.03f);

    // ---- denoise_xtrans: uniform per-channel signal is preserved (mean at CFA sites) ----
    // The port seeds the top/bottom buffer rows to 0.5 (darktable does too), so a
    // few boundary pixels deviate from the channel constant. Assert the per-color
    // MEAN instead, which is robust to those edges.
    {
        const int W = 60, H = 60;  // multiples of 6; >=60 so darktable's rightmost-green
                                   // gap + 0.5 seed rows dilute below the 0.03 tolerance
        std::vector<float> in(static_cast<size_t>(W) * H);
        std::vector<float> out(static_cast<size_t>(W) * H, -1.0f);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const unsigned c = xtransColor(y, x, xt);
                in[static_cast<size_t>(y) * W + x] = (c == 0) ? 0.20f
                                                              : (c == 1) ? 0.50f : 0.80f;
            }
        denoise_xtrans(in.data(), out.data(), W, H, xt, 0.02f);
        double sumR = 0, sumG = 0, sumB = 0; int nR = 0, nG = 0, nB = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const unsigned c = xtransColor(y, x, xt);
                const float v = out[static_cast<size_t>(y) * W + x];
                if (c == 0) { sumR += v; ++nR; }
                else if (c == 1) { sumG += v; ++nG; }
                else { sumB += v; ++nB; }
            }
        assert(nR > 0 && nG > 0 && nB > 0);
        // sqrt(v) is applied then squared back; a per-channel constant survives.
        assert(std::fabs(sumR / nR - 0.20f) < 0.03f);
        assert(std::fabs(sumG / nG - 0.50f) < 0.03f);
        assert(std::fabs(sumB / nB - 0.80f) < 0.03f);
    }

    // ---- denoise_xtrans: additive noise at CFA sites is reduced ----
    {
        const int W = 36, H = 36;
        std::vector<float> clean(static_cast<size_t>(W) * H);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const unsigned c = xtransColor(y, x, xt);
                clean[static_cast<size_t>(y) * W + x] = (c == 0) ? 0.25f
                                                                 : (c == 1) ? 0.50f : 0.75f;
            }
        auto noisy = clean;
        for (size_t i = 0; i < noisy.size(); ++i) noisy[i] += 0.06f * static_cast<float>(static_cast<int>(i * 7 % 11) - 5) / 5.0f;
        std::vector<float> out(noisy.size(), -1.0f);
        denoise_xtrans(noisy.data(), out.data(), W, H, xt, 0.05f);
        // At each CFA site, the denoised value should be closer to clean than noisy was.
        double errNoisy = 0, errDenoised = 0; long n = 0;
        for (size_t i = 0; i < noisy.size(); ++i) {
            errNoisy   += std::fabs(noisy[i] - clean[i]);
            errDenoised += std::fabs(out[i]  - clean[i]);
            ++n;
        }
        assert(errDenoised / n < errNoisy / n);
    }

    std::cout << "test_denoise_xtrans: PASS\n";
    return 0;
}
