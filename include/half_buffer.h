#pragma once
/**
 * @file half_buffer.h
 * @brief Float16 (IEEE 754 binary16) image buffer with ARM NEON conversion utilities.
 *
 * Storing image data in float16 halves memory bandwidth compared to float32,
 * which is especially valuable on bandwidth-limited mobile ARM64 SoCs.
 * The NEON intrinsics vcvt_f16_f32 / vcvt_f32_f16 process 4 elements per
 * instruction, giving efficient bulk conversion between the two formats.
 */

#include "common.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#include <cstring>
#endif

namespace rawalchemy {

#if defined(__aarch64__)

/// 3-channel float16 image buffer (row-major, interleaved RGB).
/// Stores raw IEEE 754 binary16 bits in a std::vector<uint16_t>.
struct HalfImageBuffer {
    int width = 0;
    int height = 0;
    int channels = 3;

    /// Raw binary16 bits — use __fp16 / vld1_f16 / vst1_f16 to interpret
    std::vector<uint16_t> data;

    HalfImageBuffer() = default;

    HalfImageBuffer(int w, int h, int ch = 3)
        : width(w), height(h), channels(ch),
          data(static_cast<size_t>(w) * h * ch, 0) {}

    /// Total pixel count
    size_t pixelCount() const { return static_cast<size_t>(width) * height; }

    /// Total element count (pixels × channels)
    size_t size() const { return data.size(); }

    /// Raw pointer to binary16 data
    uint16_t* ptr() { return data.data(); }
    const uint16_t* ptr() const { return data.data(); }
};

/**
 * @brief Convert a float32 ImageBuffer to a float16 HalfImageBuffer.
 *
 * Uses NEON vcvt_f16_f32 to process 4 floats at a time.
 * Tail elements (0–3) are converted scalarily.
 */
inline HalfImageBuffer convertToF16(const ImageBuffer& src) {
    HalfImageBuffer dst(src.width, src.height, src.channels);

    const float* srcPtr = src.ptr();
    uint16_t* dstPtr = dst.ptr();

    const size_t total = src.size();
    const size_t vecEnd = total - (total % 4);

    for (size_t i = 0; i < vecEnd; i += 4) {
        float32x4_t f32 = vld1q_f32(srcPtr + i);
        float16x4_t f16 = vcvt_f16_f32(f32);
        vst1_f16(reinterpret_cast<__fp16*>(dstPtr + i), f16);
    }

    // Scalar tail
    for (size_t i = vecEnd; i < total; ++i) {
        __fp16 h = static_cast<__fp16>(srcPtr[i]);
        std::memcpy(&dstPtr[i], &h, sizeof(uint16_t));
    }

    return dst;
}

/**
 * @brief Convert a float16 HalfImageBuffer to a float32 ImageBuffer.
 *
 * Uses NEON vcvt_f32_f16 to process 4 halfs at a time.
 * Tail elements (0–3) are converted scalarily.
 */
inline ImageBuffer convertToF32(const HalfImageBuffer& src) {
    ImageBuffer dst(src.width, src.height, src.channels);

    const uint16_t* srcPtr = src.ptr();
    float* dstPtr = dst.ptr();

    const size_t total = src.size();
    const size_t vecEnd = total - (total % 4);

    for (size_t i = 0; i < vecEnd; i += 4) {
        float16x4_t f16 = vld1_f16(reinterpret_cast<const __fp16*>(srcPtr + i));
        float32x4_t f32 = vcvt_f32_f16(f16);
        vst1q_f32(dstPtr + i, f32);
    }

    // Scalar tail
    for (size_t i = vecEnd; i < total; ++i) {
        __fp16 h;
        std::memcpy(&h, &srcPtr[i], sizeof(__fp16));
        dstPtr[i] = static_cast<float>(h);
    }

    return dst;
}

#endif // __aarch64__

} // namespace rawalchemy
