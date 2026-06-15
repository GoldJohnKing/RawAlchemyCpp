/**
 * @file raw_postprocess.cpp
 * @brief Phase 5 RAW postprocessing — white-balance + flip + matrix apply.
 *
 * Direct ports of Python reference `raw_alchemy.core` (WB + flip) and a small
 * OpenMP-parallel matrix application helper. Operate on demosaiced camera-RGB
 * ImageBuffers.
 */

#include "raw_postprocess.h"

#include <cstdio>

#ifdef RA_USE_OPENMP
#include <omp.h>
#endif

namespace rawalchemy {

// ============================================================
//                       White balance
//  Port of core.py:214-216 (applyWhiteBalance) verbatim.
// ============================================================
void applyWhiteBalance(ImageBuffer& rgb, const float cam_mul[4]) {
    const float g = cam_mul[1] > 0.0f ? cam_mul[1] : 1.0f;
    const float rGain = cam_mul[0] / g;
    const float bGain = cam_mul[2] / g;

    const size_t nPixels = rgb.pixelCount();
    float* data = rgb.ptr();

    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static, 8192)
    for (int i = 0; i < static_cast<int>(nPixels); ++i) {
    #else
    for (size_t i = 0; i < nPixels; ++i) {
    #endif
        float* p = data + i * 3;
        p[0] *= rGain;
        // Green channel untouched (it's the WB anchor).
        p[2] *= bGain;
    }
}

// ============================================================
//                            Flip
//  Port of onnx/denoiser.py:412-431 (_apply_flip) verbatim.
//  rawpy.sizes.flip: 0=none, 3=180°, 5=90°CCW, 6=90°CW.
// ============================================================
void applyFlip(ImageBuffer& rgb, int flip) {
    const int W = rgb.width;
    const int H = rgb.height;
    const int C = rgb.channels;  // always 3

    if (flip == 0) {
        return;  // no rotation
    }

    if (flip == 3) {
        // 180°: result[i, j, ch] = in[H-1-i, W-1-j, ch]. Shape unchanged.
        ImageBuffer out(W, H, C);
        const float* src = rgb.ptr();
        float* dst = out.ptr();
        #ifdef RA_USE_OPENMP
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < H; ++i) {
        #else
        for (int i = 0; i < H; ++i) {
        #endif
            const float* srcRow = src + static_cast<size_t>(H - 1 - i) * W * C;
            float* dstRow = dst + static_cast<size_t>(i) * W * C;
            for (int j = 0; j < W; ++j) {
                const float* sp = srcRow + static_cast<size_t>(W - 1 - j) * C;
                float* dp = dstRow + static_cast<size_t>(j) * C;
                for (int c = 0; c < C; ++c) dp[c] = sp[c];
            }
        }
        rgb = std::move(out);
        return;
    }

    if (flip == 5) {
        // 90° CCW (np.rot90 k=1). New shape (W, H). result[i, j, ch] = in[j, W-1-i, ch].
        ImageBuffer out(H, W, C);  // new width = H, new height = W
        const float* src = rgb.ptr();
        float* dst = out.ptr();
        const int newH = W;   // new height = old width
        const int newW = H;   // new width  = old height
        #ifdef RA_USE_OPENMP
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < newH; ++i) {
        #else
        for (int i = 0; i < newH; ++i) {
        #endif
            float* dstRow = dst + static_cast<size_t>(i) * newW * C;
            for (int j = 0; j < newW; ++j) {
                const float* sp = src + (static_cast<size_t>(j) * W +
                                         static_cast<size_t>(W - 1 - i)) * C;
                float* dp = dstRow + static_cast<size_t>(j) * C;
                for (int c = 0; c < C; ++c) dp[c] = sp[c];
            }
        }
        rgb = std::move(out);
        return;
    }

    if (flip == 6) {
        // 90° CW (np.rot90 k=3). New shape (W, H). result[i, j, ch] = in[H-1-j, i, ch].
        ImageBuffer out(H, W, C);  // new width = H, new height = W
        const float* src = rgb.ptr();
        float* dst = out.ptr();
        const int newH = W;
        const int newW = H;
        #ifdef RA_USE_OPENMP
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < newH; ++i) {
        #else
        for (int i = 0; i < newH; ++i) {
        #endif
            float* dstRow = dst + static_cast<size_t>(i) * newW * C;
            for (int j = 0; j < newW; ++j) {
                const float* sp = src + (static_cast<size_t>(H - 1 - j) * W +
                                         static_cast<size_t>(i)) * C;
                float* dp = dstRow + static_cast<size_t>(j) * C;
                for (int c = 0; c < C; ++c) dp[c] = sp[c];
            }
        }
        rgb = std::move(out);
        return;
    }

    // Unknown flip code — warn and skip (matches reference).
    fprintf(stderr, "[applyFlip] Warning: unknown flip code %d, skipping rotation\n", flip);
}

// ============================================================
//                       Matrix apply
//  3x3 color-matrix multiply, in-place. OpenMP across pixels.
// ============================================================
void applyColorMatrix(ImageBuffer& rgb, const float M[3][3]) {
    const float m00 = M[0][0], m01 = M[0][1], m02 = M[0][2];
    const float m10 = M[1][0], m11 = M[1][1], m12 = M[1][2];
    const float m20 = M[2][0], m21 = M[2][1], m22 = M[2][2];

    const size_t nPixels = rgb.pixelCount();
    float* data = rgb.ptr();

    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static, 8192)
    for (int i = 0; i < static_cast<int>(nPixels); ++i) {
    #else
    for (size_t i = 0; i < nPixels; ++i) {
    #endif
        float* p = data + i * 3;
        const float r = p[0], g = p[1], b = p[2];
        p[0] = r * m00 + g * m01 + b * m02;
        p[1] = r * m10 + g * m11 + b * m12;
        p[2] = r * m20 + g * m21 + b * m22;
    }
}

} // namespace rawalchemy
