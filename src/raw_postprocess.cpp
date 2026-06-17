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
//           Mosaic white balance (pre-demosaic, v2)
//  Clean per-channel gain on the float CFA mosaic. Models darktable's
//  `temperature` iop (position 3.0). Green-anchored: G sites untouched,
//  R/B photosites scaled by cam_mul/green. G2 (cfaColor==3) treated as G.
//  Replaces the v1 design where WB was deferred to post-demosaic.
// ============================================================
void applyWhiteBalanceMosaic(RawMosaic& m) {
    const float g = m.cam_mul[1] > 0.0f ? m.cam_mul[1] : 1.0f;
    const float rGain = m.cam_mul[0] / g;
    const float bGain = m.cam_mul[2] / g;

    const int W = m.width;
    const int H = m.height;
    float* data = m.data.data();

    #pragma omp parallel for schedule(static, 8192) collapse(2)
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            const size_t idx = static_cast<size_t>(row) * W + col;
            const int color = cfaColor(m, row, col);
            if (color == 0) {
                data[idx] *= rGain;
            } else if (color == 2) {
                data[idx] *= bGain;
            }
            // color 1 (G) and 3 (G2): green anchor, untouched.
        }
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
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < H; ++i) {
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
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < newH; ++i) {
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
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < newH; ++i) {
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
void applyColorMatrix(ImageBuffer& rgb, const std::array<std::array<float, 3>, 3>& M) {
    const float m00 = M[0][0], m01 = M[0][1], m02 = M[0][2];
    const float m10 = M[1][0], m11 = M[1][1], m12 = M[1][2];
    const float m20 = M[2][0], m21 = M[2][1], m22 = M[2][2];

    const size_t nPixels = rgb.pixelCount();
    float* data = rgb.ptr();

    #pragma omp parallel for schedule(static, 8192)
    for (int i = 0; i < static_cast<int>(nPixels); ++i) {
        float* p = data + i * 3;
        const float r = p[0], g = p[1], b = p[2];
        p[0] = r * m00 + g * m01 + b * m02;
        p[1] = r * m10 + g * m11 + b * m12;
        p[2] = r * m20 + g * m21 + b * m22;
    }
}

} // namespace rawalchemy
