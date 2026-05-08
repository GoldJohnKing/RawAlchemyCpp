#pragma once
/**
 * @file common.h
 * @brief Common types and utilities for Raw Alchemy.
 *
 * Core Philosophy Step 1 — Standardized Decoding:
 *   RAW (Camera Native) -> ProPhoto RGB (Linear) -> float32 [0.0, 1.0]
 */

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

namespace rawalchemy {

/// 3-channel float32 image buffer (row-major, interleaved RGB)
struct ImageBuffer {
    int width  = 0;
    int height = 0;
    int channels = 3;          // Always RGB

    /// Pixel data in float32, range [0.0, 1.0], row-major RGB interleaved
    std::vector<float> data;

    ImageBuffer() = default;

    ImageBuffer(int w, int h, int ch = 3)
        : width(w), height(h), channels(ch),
          data(static_cast<size_t>(w) * h * ch, 0.0f)
    {}

    /// Total pixel count
    size_t pixelCount() const {
        return static_cast<size_t>(width) * height;
    }

    /// Total element count (pixels * channels)
    size_t size() const {
        return data.size();
    }

    /// Raw pointer to pixel data
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }

    /// Pointer to a specific pixel (row, col)
    float* pixel(int row, int col) {
        return data.data() + (static_cast<size_t>(row) * width + col) * channels;
    }
    const float* pixel(int row, int col) const {
        return data.data() + (static_cast<size_t>(row) * width + col) * channels;
    }

    /// Clamp all values to [0.0, 1.0] in-place
    void clamp() {
        for (auto& v : data) {
            v = std::max(0.0f, std::min(1.0f, v));
        }
    }

    /// Apply gain (exposure multiplier) in-place
    void applyGain(float gain) {
        for (auto& v : data) {
            v *= gain;
        }
    }

    /// Convert from uint16 buffer (0-65535) to float32 [0.0, 1.0]
    static ImageBuffer fromUint16(const uint16_t* src, int w, int h, int channels = 3) {
        ImageBuffer buf(w, h, channels);
        const size_t n = buf.size();
        for (size_t i = 0; i < n; ++i) {
            buf.data[i] = static_cast<float>(src[i]) / 65535.0f;
        }
        return buf;
    }

    /// Convert to uint16 buffer (caller must allocate)
    void toUint16(uint16_t* dst) const {
        const size_t n = size();
        for (size_t i = 0; i < n; ++i) {
            dst[i] = static_cast<uint16_t>(
                std::max(0.0f, std::min(1.0f, data[i])) * 65535.0f + 0.5f
            );
        }
    }
};

} // namespace rawalchemy
