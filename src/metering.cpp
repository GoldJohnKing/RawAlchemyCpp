/**
 * @file metering.cpp
 * @brief Auto exposure metering — direct port of Python metering.py.
 *
 * All strategies:
 *   1. Subsample image to ~1024px wide for speed
 *   2. Compute per-pixel luminance (ProPhoto RGB coefficients)
 *   3. Calculate gain based on the strategy's logic
 *   4. Return gain (caller applies it to the full image)
 */

#include "metering.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <cstdio>

namespace rawalchemy {

// ---- Helpers ----

/// Compute luminance for all pixels in a subsampled view
static std::vector<float> computeLuminance(const ImageBuffer& img, int step) {
    const int h = (img.height + step - 1) / step;
    const int w = (img.width + step - 1) / step;
    std::vector<float> lum(static_cast<size_t>(h) * w);

    const float* data = img.ptr();
    const int W = img.width;
    const int C = img.channels;

    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            const float* p = data + (static_cast<size_t>(r * step) * W + c * step) * C;
            lum[static_cast<size_t>(r) * w + c] =
                PROPHOTO_LUMA_R * p[0] + PROPHOTO_LUMA_G * p[1] + PROPHOTO_LUMA_B * p[2];
        }
    }
    return lum;
}

/// Get max(R,G,B) per pixel in subsampled view
static std::vector<float> computeMaxChannel(const ImageBuffer& img, int step, int& outH, int& outW) {
    outH = (img.height + step - 1) / step;
    outW = (img.width + step - 1) / step;
    std::vector<float> mx(static_cast<size_t>(outH) * outW);

    const float* data = img.ptr();
    const int W = img.width;

    for (int r = 0; r < outH; r++) {
        for (int c = 0; c < outW; c++) {
            const float* p = data + (static_cast<size_t>(r * step) * W + c * step) * 3;
            mx[static_cast<size_t>(r) * outW + c] = std::max({p[0], p[1], p[2]});
        }
    }
    return mx;
}

/// Percentile on a vector (modifies copy)
static float percentile(std::vector<float> v, float pct) {
    if (v.empty()) return 0.0f;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(pct / 100.0f * (v.size() - 1));
    idx = std::min(idx, v.size() - 1);
    return v[idx];
}

static int subsampleStep(const ImageBuffer& img, int targetSize = 1024) {
    return std::max(1, std::max(img.width, img.height) / targetSize);
}

// ---- Average (geometric mean) ----
static float meteringAverage(const ImageBuffer& img, float targetGray) {
    int step = subsampleStep(img);
    auto lum = computeLuminance(img, step);

    double sumLog = 0;
    size_t n = lum.size();
    for (size_t i = 0; i < n; i++) {
        sumLog += std::log(std::max(lum[i], 1e-10f) + 1e-6f);
    }
    float avgLum = static_cast<float>(std::exp(sumLog / n));

    float gain = (avgLum < 0.0001f) ? 1.0f : targetGray / avgLum;
    return std::max(1.0f, std::min(gain, 50.0f));
}

// ---- Center-Weighted ----
static float meteringCenterWeighted(const ImageBuffer& img, float targetGray) {
    int step = subsampleStep(img);
    auto lum = computeLuminance(img, step);

    int h = (img.height + step - 1) / step;
    int w = (img.width + step - 1) / step;

    float cy = h / 2.0f, cx = w / 2.0f;
    float sigma = std::min(h, w) / 2.0f;
    float inv2s2 = 1.0f / (2.0f * sigma * sigma);

    double wSum = 0, lwSum = 0;
    for (int r = 0; r < h; r++) {
        float dy = r - cy;
        for (int c = 0; c < w; c++) {
            float dx = c - cx;
            float wt = std::exp(-(dx * dx + dy * dy) * inv2s2);
            float l = lum[static_cast<size_t>(r) * w + c];
            wSum += wt;
            lwSum += l * wt;
        }
    }

    float avgLum = static_cast<float>(lwSum / wSum);
    float gain = (avgLum < 1e-6f) ? 1.0f : targetGray / avgLum;
    return std::max(0.1f, std::min(gain, 100.0f));
}

// ---- Highlight-Safe (ETTR) ----
static float meteringHighlightSafe(const ImageBuffer& img, float /*targetGray*/) {
    int step = subsampleStep(img);
    int h, w;
    auto mx = computeMaxChannel(img, step, h, w);

    float p99 = percentile(std::move(mx), 99.0f);

    float targetHigh = 0.9f;
    return (p99 < 1e-6f) ? 1.0f : targetHigh / p99;
}

// ---- Hybrid (average + highlight limit) ----
static float meteringHybrid(const ImageBuffer& img, float targetGray) {
    int step = subsampleStep(img);
    auto lum = computeLuminance(img, step);
    int h, w;
    auto mx = computeMaxChannel(img, step, h, w);

    // Average (geometric mean)
    double sumLog = 0;
    size_t n = lum.size();
    for (size_t i = 0; i < n; i++) {
        sumLog += std::log(std::max(lum[i], 1e-10f) + 1e-6f);
    }
    float avgLum = static_cast<float>(std::exp(sumLog / n));
    float baseGain = targetGray / (avgLum + 1e-6f);

    // Highlight guard
    float p99 = percentile(std::move(mx), 99.0f);
    constexpr float maxAllowedPeak = 6.0f;

    float gain = baseGain;
    if (p99 * baseGain > maxAllowedPeak) {
        gain = maxAllowedPeak / p99;
    }

    return std::max(0.1f, std::min(gain, 100.0f));
}

// ---- Matrix / Evaluative (7x7 grid) ----
static float meteringMatrix(const ImageBuffer& img, float targetGray) {
    int step = subsampleStep(img);
    auto lum = computeLuminance(img, step);
    int h = (img.height + step - 1) / step;
    int w = (img.width + step - 1) / step;

    constexpr int G = 7;
    int gh = h / G, gw = w / G;
    if (gh < 1 || gw < 1) return meteringHybrid(img, targetGray);

    // Grid averages
    float grid[G * G];
    for (int i = 0; i < G; i++) {
        for (int j = 0; j < G; j++) {
            double sum = 0;
            int count = 0;
            for (int r = i * gh; r < (i + 1) * gh && r < h; r++) {
                for (int c = j * gw; c < (j + 1) * gw && c < w; c++) {
                    sum += lum[static_cast<size_t>(r) * w + c];
                    count++;
                }
            }
            grid[i * G + j] = (count > 0) ? static_cast<float>(sum / count) : 0.0f;
        }
    }

    // Weights
    float weights[G * G];
    for (int i = 0; i < G * G; i++) weights[i] = 1.0f;

    // Center bias (Gaussian)
    float cy = (G - 1) / 2.0f, cx = (G - 1) / 2.0f;
    float sigma = G / 2.5f;
    float inv2s2 = 1.0f / (2.0f * sigma * sigma);
    for (int i = 0; i < G; i++) {
        float dy = i - cy;
        for (int j = 0; j < G; j++) {
            float dx = j - cx;
            float bias = std::exp(-(dx * dx + dy * dy) * inv2s2);
            weights[i * G + j] *= (1.0f + bias * 1.5f);
        }
    }

    // Highlight suppression
    std::vector<float> gridVec(grid, grid + G * G);
    float p90 = percentile(gridVec, 90.0f);
    for (int i = 0; i < G * G; i++) {
        if (grid[i] > p90) weights[i] *= 0.2f;
    }

    // Shadow boost
    float p10 = percentile(std::vector<float>(grid, grid + G * G), 10.0f);
    for (int i = 0; i < G * G; i++) {
        if (grid[i] < p10) weights[i] *= 1.2f;
    }

    // Weighted average
    double wSum = 0, lwSum = 0;
    for (int i = 0; i < G * G; i++) {
        wSum += weights[i];
        lwSum += grid[i] * weights[i];
    }
    float avgLum = static_cast<float>(lwSum / wSum);

    float gain = (avgLum < 1e-6f) ? 1.0f : targetGray / avgLum;

    // Protective highlight guard
    int hh, ww;
    auto mx = computeMaxChannel(img, step, hh, ww);
    float p99 = percentile(std::move(mx), 99.0f);
    constexpr float maxAllowedPeak = 6.0f;
    if (p99 * gain > maxAllowedPeak) {
        gain = maxAllowedPeak / p99;
    }

    return std::max(0.1f, std::min(gain, 100.0f));
}

// ---- Public API ----

float computeAutoGain(const ImageBuffer& img, const std::string& mode, float targetGray) {
    if (mode == "average")         return meteringAverage(img, targetGray);
    if (mode == "center-weighted") return meteringCenterWeighted(img, targetGray);
    if (mode == "highlight-safe")  return meteringHighlightSafe(img, targetGray);
    if (mode == "hybrid")          return meteringHybrid(img, targetGray);
    if (mode == "matrix")          return meteringMatrix(img, targetGray);
    throw std::runtime_error("[Metering] Unknown mode: " + mode);
}

std::vector<std::string> getSupportedMeteringModes() {
    return {"average", "center-weighted", "highlight-safe", "hybrid", "matrix"};
}

bool isMeteringModeSupported(const std::string& mode) {
    return mode == "average" || mode == "center-weighted" || mode == "highlight-safe"
        || mode == "hybrid" || mode == "matrix";
}

} // namespace rawalchemy
