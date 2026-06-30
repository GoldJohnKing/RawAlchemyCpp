// CameraFTP - A Cross-platform FTP companion for camera photo transfer
// Copyright (C) 2026 GoldJohnKing <GoldJohnKing@Live.cn>
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file grading_fused.cpp
 * @brief Single-pass fused grading pipeline implementation.
 *
 * Fuses: gain → saturation/contrast → gamut transform → log encoding → 3D LUT
 * into one per-pixel loop, reducing memory bandwidth by ~5×.
 */

#include "grading_fused.h"
#include "metering.h"

#include <algorithm>
#include <cmath>

#ifdef RA_USE_OPENMP
#include <omp.h>
#endif

namespace rawalchemy {

void applyGradingFused(ImageBuffer& img, const GradingParams& params) {
    const size_t nPixels = img.pixelCount();
    float* data = img.ptr();

    const float gain = params.gain;
    const bool applyGain = (gain != 1.0f);

    const bool doBoost = params.enableBoost;
    const float sat = params.saturation;
    const float cont = params.contrast;
    const float pivot = params.pivot;
    const float Lr = PROPHOTO_LUMA_R;
    const float Lg = PROPHOTO_LUMA_G;
    const float Lb = PROPHOTO_LUMA_B;

    bool doGamut = (params.logSpaceInfo != nullptr);
    float m00 = 0, m01 = 0, m02 = 0;
    float m10 = 0, m11 = 0, m12 = 0;
    float m20 = 0, m21 = 0, m22 = 0;
    LogCurve curve = LogCurve::F_Log;
    if (doGamut) {
        const auto& M = *(params.logSpaceInfo->gamutMatrix);
        m00 = M[0][0]; m01 = M[0][1]; m02 = M[0][2];
        m10 = M[1][0]; m11 = M[1][1]; m12 = M[1][2];
        m20 = M[2][0]; m21 = M[2][1]; m22 = M[2][2];
        curve = params.logSpaceInfo->curve;
    }

    const bool doLut = (params.lut != nullptr && !params.lut->empty());
    const int lutSize = doLut ? params.lut->size : 0;
    const int lutSizeM1 = doLut ? lutSize - 1 : 0;
    const float lutSizeF = static_cast<float>(lutSizeM1);
    const float* lutTable = doLut ? params.lut->table.data() : nullptr;
    float scaleR = 0, scaleG = 0, scaleB = 0;
    float minR = 0, minG = 0, minB = 0;
    if (doLut) {
        scaleR = lutSizeF / (params.lut->domainMax[0] - params.lut->domainMin[0]);
        scaleG = lutSizeF / (params.lut->domainMax[1] - params.lut->domainMin[1]);
        scaleB = lutSizeF / (params.lut->domainMax[2] - params.lut->domainMin[2]);
        minR = params.lut->domainMin[0];
        minG = params.lut->domainMin[1];
        minB = params.lut->domainMin[2];
    }

#ifdef RA_USE_OPENMP
#pragma omp parallel for schedule(static, 8192)
    for (int i = 0; i < static_cast<int>(nPixels); i++) {
#else
    for (size_t i = 0; i < nPixels; i++) {
#endif
        float* p = data + i * 3;
        float r = p[0], g = p[1], b = p[2];

        if (applyGain) {
            r *= gain;
            g *= gain;
            b *= gain;
        }

        if (doBoost) {
            float lum = Lr * r + Lg * g + Lb * b;
            // Taper the saturation boost toward 1.0 in the highlight region so grading
            // doesn't re-amplify chroma that desaturateHighlightsLinear (raw_decoder.cpp)
            // just suppressed. Without this, the ×1.25 boost partially undoes the
            // desaturation and re-exposes residual chroma at highlight edges — the pink
            // rim. sat ramps sat→1.0 over luma [0.5, 0.9], keeping midtone punch while
            // leaving highlights desaturated. Complements the Gaussian desat's broad reach.
            float effSat = sat;
            if (lum > 0.5f) {
                float taper = (lum - 0.5f) * 2.5f;   // 0 at lum=0.5, 1 at lum=0.9
                if (taper > 1.0f) taper = 1.0f;
                effSat = sat + (1.0f - sat) * taper;  // interpolates sat → 1.0
            }
            float rs = lum + (r - lum) * effSat;
            float gs = lum + (g - lum) * effSat;
            float bs = lum + (b - lum) * effSat;
            r = std::max(0.0f, (rs - pivot) * cont + pivot);
            g = std::max(0.0f, (gs - pivot) * cont + pivot);
            b = std::max(0.0f, (bs - pivot) * cont + pivot);
        }

        if (doGamut) {
            float gr = r * m00 + g * m01 + b * m02;
            float gg = r * m10 + g * m11 + b * m12;
            float gb = r * m20 + g * m21 + b * m22;
            r = gr;
            g = gg;
            b = gb;
        }

        if (doGamut) {
            r = logEncode(std::max(r, 1e-6f), curve);
            g = logEncode(std::max(g, 1e-6f), curve);
            b = logEncode(std::max(b, 1e-6f), curve);
        }

        if (doLut) {
            float rawIdxR = (r - minR) * scaleR;
            float rawIdxG = (g - minG) * scaleG;
            float rawIdxB = (b - minB) * scaleB;

            float idxR = std::max(0.0f, std::min(rawIdxR, lutSizeF));
            float idxG = std::max(0.0f, std::min(rawIdxG, lutSizeF));
            float idxB = std::max(0.0f, std::min(rawIdxB, lutSizeF));

            int x0 = static_cast<int>(idxR);
            int y0 = static_cast<int>(idxG);
            int z0 = static_cast<int>(idxB);

            int x1 = (x0 < lutSizeM1) ? x0 + 1 : x0;
            int y1 = (y0 < lutSizeM1) ? y0 + 1 : y0;
            int z1 = (z0 < lutSizeM1) ? z0 + 1 : z0;

            float dx = idxR - x0;
            float dy = idxG - y0;
            float dz = idxB - z0;

            float rVal = 0.0f, gVal = 0.0f, bVal = 0.0f;

            #define TBL(r, g, b, c) lutTable[((r) + (g)*lutSize + (b)*lutSize*lutSize) * 3 + (c)]

            if (dx >= dy) {
                if (dy >= dz) {
                    float w0 = 1.0f - dx;
                    float w1 = dx - dy;
                    float w2 = dy - dz;
                    rVal = TBL(x0,y0,z0,0)*w0 + TBL(x1,y0,z0,0)*w1 + TBL(x1,y1,z0,0)*w2 + TBL(x1,y1,z1,0)*dz;
                    gVal = TBL(x0,y0,z0,1)*w0 + TBL(x1,y0,z0,1)*w1 + TBL(x1,y1,z0,1)*w2 + TBL(x1,y1,z1,1)*dz;
                    bVal = TBL(x0,y0,z0,2)*w0 + TBL(x1,y0,z0,2)*w1 + TBL(x1,y1,z0,2)*w2 + TBL(x1,y1,z1,2)*dz;
                }
                else if (dx >= dz) {
                    float w0 = 1.0f - dx;
                    float w1 = dx - dz;
                    float w2 = dz - dy;
                    rVal = TBL(x0,y0,z0,0)*w0 + TBL(x1,y0,z0,0)*w1 + TBL(x1,y0,z1,0)*w2 + TBL(x1,y1,z1,0)*dy;
                    gVal = TBL(x0,y0,z0,1)*w0 + TBL(x1,y0,z0,1)*w1 + TBL(x1,y0,z1,1)*w2 + TBL(x1,y1,z1,1)*dy;
                    bVal = TBL(x0,y0,z0,2)*w0 + TBL(x1,y0,z0,2)*w1 + TBL(x1,y0,z1,2)*w2 + TBL(x1,y1,z1,2)*dy;
                }
                else {
                    float w0 = 1.0f - dz;
                    float w1 = dz - dx;
                    float w2 = dx - dy;
                    rVal = TBL(x0,y0,z0,0)*w0 + TBL(x0,y0,z1,0)*w1 + TBL(x1,y0,z1,0)*w2 + TBL(x1,y1,z1,0)*dy;
                    gVal = TBL(x0,y0,z0,1)*w0 + TBL(x0,y0,z1,1)*w1 + TBL(x1,y0,z1,1)*w2 + TBL(x1,y1,z1,1)*dy;
                    bVal = TBL(x0,y0,z0,2)*w0 + TBL(x0,y0,z1,2)*w1 + TBL(x1,y0,z1,2)*w2 + TBL(x1,y1,z1,2)*dy;
                }
            }
            else {
                if (dz >= dy) {
                    float w0 = 1.0f - dz;
                    float w1 = dz - dy;
                    float w2 = dy - dx;
                    rVal = TBL(x0,y0,z0,0)*w0 + TBL(x0,y0,z1,0)*w1 + TBL(x0,y1,z1,0)*w2 + TBL(x1,y1,z1,0)*dx;
                    gVal = TBL(x0,y0,z0,1)*w0 + TBL(x0,y0,z1,1)*w1 + TBL(x0,y1,z1,1)*w2 + TBL(x1,y1,z1,1)*dx;
                    bVal = TBL(x0,y0,z0,2)*w0 + TBL(x0,y0,z1,2)*w1 + TBL(x0,y1,z1,2)*w2 + TBL(x1,y1,z1,2)*dx;
                }
                else if (dz >= dx) {
                    float w0 = 1.0f - dy;
                    float w1 = dy - dz;
                    float w2 = dz - dx;
                    rVal = TBL(x0,y0,z0,0)*w0 + TBL(x0,y1,z0,0)*w1 + TBL(x0,y1,z1,0)*w2 + TBL(x1,y1,z1,0)*dx;
                    gVal = TBL(x0,y0,z0,1)*w0 + TBL(x0,y1,z0,1)*w1 + TBL(x0,y1,z1,1)*w2 + TBL(x1,y1,z1,1)*dx;
                    bVal = TBL(x0,y0,z0,2)*w0 + TBL(x0,y1,z0,2)*w1 + TBL(x0,y1,z1,2)*w2 + TBL(x1,y1,z1,2)*dx;
                }
                else {
                    float w0 = 1.0f - dy;
                    float w1 = dy - dx;
                    float w2 = dx - dz;
                    rVal = TBL(x0,y0,z0,0)*w0 + TBL(x0,y1,z0,0)*w1 + TBL(x1,y1,z0,0)*w2 + TBL(x1,y1,z1,0)*dz;
                    gVal = TBL(x0,y0,z0,1)*w0 + TBL(x0,y1,z0,1)*w1 + TBL(x1,y1,z0,1)*w2 + TBL(x1,y1,z1,1)*dz;
                    bVal = TBL(x0,y0,z0,2)*w0 + TBL(x0,y1,z0,2)*w1 + TBL(x1,y1,z0,2)*w2 + TBL(x1,y1,z1,2)*dz;
                }
            }

            #undef TBL

            r = rVal;
            g = gVal;
            b = bVal;
        }

        p[0] = r;
        p[1] = g;
        p[2] = b;
    }
}

} // namespace rawalchemy
