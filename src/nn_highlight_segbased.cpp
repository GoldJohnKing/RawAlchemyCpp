// SPDX-License-Identifier: AGPL-3.0-or-later
// Segmentation-based highlight reconstruction — see nn_highlight_segbased.h.
// Slimmed port: only the full-clip recovery path (darktable segbased.c:621-700).
// Partial-clip candidate reconstruction (step g) removed — opposed handles that.

#include "nn_highlight_segbased.h"

#include "nn_image_primitives.h"
#include "nn_logging.h"
#include "nn_segmentation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace rawalchemy {
namespace {

constexpr float kStrength = 1.0f;
constexpr float kNoiseLevel = 0.0f;
constexpr int kRecoveryClose = 2;

constexpr int HL_RGB_PLANES = 3;
constexpr int HL_FLOAT_PLANES = 8;
constexpr int HL_BORDER = 8;

inline size_t roundUp2(int x) { return (size_t)((x + 1) & ~1); }

inline size_t rawToPlane(int pwidth, int row, int col) {
    return ((size_t)HL_BORDER + (row / 3)) * pwidth + (col / 3) + HL_BORDER;
}

// Initial boundary gradient (Scharr) at distance 0..2 from the all-clipped edge.
// darktable segbased.c:224-242. Writes into `recout`.
void initialGradients(int w, int height, const float* luminance,
                      const float* distance, float* recout) {
    for (int row = HL_BORDER + 2; row < height - HL_BORDER - 2; ++row) {
        for (int col = HL_BORDER + 2; col < w - HL_BORDER - 2; ++col) {
            const size_t v = (size_t)row * w + col;
            float g = 0.0f;
            if (distance[v] > 0.0f && distance[v] < 2.0f)
                g = 4.0f * scharrGradient(&luminance[v], w);
            recout[v] = g;
        }
    }
}

float segmentMaxDistance(const float* distance, const Segmentation& seg, uint32_t id) {
    const int xmin = std::max(seg.xmin[id] - 2, seg.border);
    const int xmax = std::min(seg.xmax[id] + 3, seg.width - seg.border);
    const int ymin = std::max(seg.ymin[id] - 2, seg.border);
    const int ymax = std::min(seg.ymax[id] + 3, seg.height - seg.border);
    float maxD = 0.0f;
    for (int row = ymin; row < ymax; ++row)
        for (int col = xmin; col < xmax; ++col) {
            const size_t v = (size_t)row * seg.width + col;
            if ((uint32_t)seg.data[v] == id) maxD = std::max(maxD, distance[v]);
        }
    return maxD;
}

// Adaptive attenuation from segment size. darktable segbased.c:267-277 (ADAPT mode).
float segmentAttenuation(const Segmentation& seg, uint32_t id) {
    const float maxdist = std::max(1.0f, seg.val1[id]);
    return std::min(1.7f, 0.9f + 3.0f / maxdist);
}

// Propagate gradients inward by distance rings. darktable segbased.c:288-328.
// Operates on `recout` (named gradient in darktable's signature).
void calcDistanceRing(int xmin, int xmax, int ymin, int ymax,
                      float* recout, const float* distance,
                      float attenuate, float dist, const Segmentation& seg, uint32_t id) {
    for (int row = ymin; row < ymax; ++row) {
        for (int col = xmin; col < xmax; ++col) {
            const size_t v = (size_t)row * seg.width + col;
            const float dv = distance[v];
            if (dv < dist || dv >= dist + 1.5f || (uint32_t)seg.data[v] != id) continue;
            float grd = 0.0f, cnt = 0.0f;
            for (int y = -2; y < 3; ++y)
                for (int x = -2; x < 3; ++x) {
                    const size_t p = v + (size_t)x + (size_t)seg.width * y;
                    const float dd = distance[p];
                    if (dd >= dist - 1.5f && dd < dist) { cnt += 1.0f; grd += recout[p]; }
                }
            if (cnt > 0.0f)
                recout[v] = std::min(1.5f, (grd / cnt) * (1.0f + 1.0f / std::pow(distance[v], attenuate)));
        }
    }
}

// Full gradient propagation for one segment. darktable segbased.c:330-384.
// Propagates in `recout`, uses `boxTmp` for the ridge-suppression box blur.
void segmentGradients(const float* distance, float* recout, float* boxTmp,
                      const Segmentation& seg, uint32_t id) {
    const int xmin = std::max(seg.xmin[id] - 1, seg.border);
    const int xmax = std::min(seg.xmax[id] + 2, seg.width - seg.border);
    const int ymin = std::max(seg.ymin[id] - 1, seg.border);
    const int ymax = std::min(seg.ymax[id] + 2, seg.height - seg.border);
    const float attenuate = segmentAttenuation(seg, id);
    const float strength = attenuate - 0.1f * (float)kRecoveryClose;

    float maxdist = 1.5f;
    while (maxdist < seg.val1[id]) {
        calcDistanceRing(xmin, xmax, ymin, ymax, recout, distance, attenuate, maxdist, seg, id);
        maxdist += 1.5f;
    }
    if (maxdist > 4.0f) {
        const int sw = xmax - xmin;
        const int sh = ymax - ymin;
        for (int row = 0; row < sh; ++row)
            for (int col = 0; col < sw; ++col)
                boxTmp[(size_t)row * sw + col] = recout[(size_t)(row + ymin) * seg.width + (col + xmin)];
        boxMean1ch(boxTmp, sw, sh, std::min((int)maxdist, 15), 2);
        for (int row = 0; row < sh; ++row)
            for (int col = 0; col < sw; ++col) {
                const size_t v = (size_t)(row + ymin) * seg.width + (col + xmin);
                if ((uint32_t)seg.data[v] == id) recout[v] = boxTmp[(size_t)row * sw + col];
            }
    }
    for (int row = ymin; row < ymax; ++row)
        for (int col = xmin; col < xmax; ++col) {
            const size_t v = (size_t)row * seg.width + col;
            if ((uint32_t)seg.data[v] == id) recout[v] *= strength;
        }
}

void addPoissonNoise(float* lum, const Segmentation& seg, uint32_t id, float noiseLevel) {
    const int xmin = std::max(seg.xmin[id], seg.border);
    const int xmax = std::min(seg.xmax[id] + 1, seg.width - seg.border);
    const int ymin = std::max(seg.ymin[id], seg.border);
    const int ymax = std::min(seg.ymax[id] + 1, seg.height - seg.border);
    RngState state = {splitmix32((uint64_t)ymin), splitmix32((uint64_t)xmin),
                      splitmix32(1337), splitmix32(666)};
    xoshiro128plus(state); xoshiro128plus(state); xoshiro128plus(state); xoshiro128plus(state);
    for (int row = ymin; row < ymax; ++row)
        for (int col = xmin; col < xmax; ++col) {
            const size_t v = (size_t)row * seg.width + col;
            if ((uint32_t)seg.data[v] == id)
                lum[v] += poissonNoise(lum[v] * noiseLevel, noiseLevel, col & 1, state);
        }
}

void masksExtendBorder(float* mask, int width, int height, int border) {
    if (border <= 0) return;
    for (int row = border; row < height - border; ++row) {
        const size_t idx = (size_t)row * width;
        for (int i = 0; i < border; ++i) {
            mask[idx + i] = mask[idx + border];
            mask[idx + width - i - 1] = mask[idx + width - border - 1];
        }
    }
    for (int col = 0; col < width; ++col) {
        const float top = mask[(size_t)border * width + std::min(width - border - 1, std::max(col, border))];
        const float bot = mask[(size_t)(height - border - 1) * width + std::min(width - border - 1, std::max(col, border))];
        for (int i = 0; i < border; ++i) {
            mask[col + (size_t)i * width] = top;
            mask[col + (size_t)(height - i - 1) * width] = bot;
        }
    }
}

} // namespace

bool reconstructHighlightsSegmentBased(float* cfa, int W, int H,
                                       const CfaPhase& phase, float clipFactor,
                                       const float wbRgb[3]) {
    std::vector<uint8_t> fc((size_t)W * H);
    for (int y = 0; y < H; ++y) {
        const size_t rb = (size_t)y * W;
        for (int x = 0; x < W; ++x)
            fc[rb + x] = (uint8_t)canonicalCfaColor(y + phase.dy, x + phase.dx, phase);
    }

    const float cubeClipval[3] = {
        std::cbrt(clipFactor * wbRgb[0]), std::cbrt(clipFactor * wbRgb[1]), std::cbrt(clipFactor * wbRgb[2])
    };
    const int xshifter = (!phase.isXtrans && fc[0] == 1) ? 1 : 2;

    const int pwidth = (int)roundUp2(W / 3) + 2 * HL_BORDER;
    const int pheight = (int)roundUp2(H / 3) + 2 * HL_BORDER;
    const size_t pSize = (size_t)pwidth * pheight;

    // 8 float planes: [0..2]=color means (cbrt), [3]=distance, [4]=gradient,
    // [5]=luminance, [6]=recout (gradient propagation), [7]=tmp (luminance+box scratch).
    std::vector<float> fbuffer(HL_FLOAT_PLANES * pSize);
    float* plane[HL_FLOAT_PLANES];
    for (int i = 0; i < HL_FLOAT_PLANES; ++i) plane[i] = fbuffer.data() + (size_t)i * pSize;
    float* distance  = plane[3];
    float* gradient  = plane[4];
    float* luminance = plane[5];
    float* recout    = plane[6];
    float* planeTmp  = plane[7];

    Segmentation segall;
    const int segLimit = std::max(256, W * H / 4000);
    if (segmentationInit(segall, pwidth, pheight, HL_BORDER + 1, segLimit)) {
        nnlog::info("[NN] segbased: segmentation init failed, skipping");
        return false;
    }

    // --- Build color planes (cbrt means) + detect all-clipped superpixels. ---
    long anyclipped = 0;
    bool hasAllClipped = false;
    for (int row = 1; row < H - 1; ++row) {
        if (row % 3 != 1) continue;
        for (int col = 1; col < W - 1; ++col) {
            if (col % 3 != xshifter) continue;
            float mean[3] = {0, 0, 0}, cnt[3] = {0, 0, 0};
            for (int dy = row - 1; dy < row + 2; ++dy) {
                const size_t rb = (size_t)dy * W;
                for (int dx = col - 1; dx < col + 2; ++dx) {
                    const int c = fc[rb + dx];
                    mean[c] += std::max(0.0f, cfa[rb + dx]);
                    cnt[c] += 1.0f;
                }
            }
            int allclipped = 0;
            const size_t o = rawToPlane(pwidth, row, col);
            for (int c = 0; c < 3; ++c) {
                mean[c] = (cnt[c] > 0.0f) ? std::cbrt(mean[c] / cnt[c]) : 0.0f;
                plane[c][o] = mean[c];
                if (mean[c] > cubeClipval[c]) ++allclipped;
            }
            segall.data[o] = (allclipped == 3) ? 1u : 0u;
            if (allclipped == 3) hasAllClipped = true;
            anyclipped += allclipped;
        }
    }

    if (anyclipped < 20) return false;

    // --- Full-clip recovery (darktable segbased.c:621-700). ---
    if (!hasAllClipped || kStrength <= 0.0f) return false;

    for (int i = 0; i < 3; ++i) masksExtendBorder(plane[i], pwidth, pheight, HL_BORDER);
    segmentsCombine(segall, kRecoveryClose);
    std::fill(gradient, gradient + pSize, std::min(1.0f, 5.0f * kStrength));
    std::fill(distance, distance + pSize, 0.0f);
    for (int row = segall.border; row < pheight - segall.border; ++row) {
        for (int col = segall.border; col < pwidth - segall.border; ++col) {
            const size_t i = (size_t)row * pwidth + col;
            planeTmp[i] = (plane[0][i] * wbRgb[0] + plane[1][i] * wbRgb[1] + plane[2][i] * wbRgb[2]) / 3.0f;
            distance[i] = (segall.data[i] == 1u) ? kDtMax : 0.0f;
        }
    }
    masksExtendBorder(planeTmp, pwidth, pheight, segall.border);
    gaussianBlur1ch(planeTmp, luminance, pwidth, pheight, 1.2f);

    const float maxDistance = distanceTransform(distance, pwidth, pheight);
    if (maxDistance <= 3.0f) return true;  // all-clipped regions too small for recovery

    segmentizePlane(segall);
    initialGradients(pwidth, pheight, luminance, distance, recout);
    masksExtendBorder(recout, pwidth, pheight, segall.border);

    for (uint32_t id = 2; id < (uint32_t)segall.nr; ++id) {
        segall.val1[id] = segmentMaxDistance(distance, segall, id);
        if (segall.val1[id] > 2.0f)
            segmentGradients(distance, recout, planeTmp, segall, id);
    }
    gaussianBlur1ch(recout, gradient, pwidth, pheight, 1.2f);

    if (kNoiseLevel > 0.0f)
        for (uint32_t id = 2; id < (uint32_t)segall.nr; ++id)
            if (segall.val1[id] > 3.0f) addPoissonNoise(gradient, segall, id, kNoiseLevel);

    // Commit recovery: add propagated texture to clipped sensels, in-place.
    const float dshift = 2.0f + (float)kRecoveryClose;
    const float clips[3] = { clipFactor * wbRgb[0], clipFactor * wbRgb[1], clipFactor * wbRgb[2] };
    for (int row = 1; row < H - 1; ++row) {
        const size_t rb = (size_t)row * W;
        for (int col = 1; col < W - 1; ++col) {
            const size_t idx = rb + col;
            const int c = fc[idx];
            if (std::max(0.0f, cfa[idx]) <= clips[c]) continue;
            const size_t o = rawToPlane(pwidth, row, col);
            const float effect = kStrength / (1.0f + std::exp(-(distance[o] - dshift)));
            cfa[idx] += std::max(0.0f, gradient[o] * effect);
        }
    }

    nnlog::info("[NN] segbased: recovered all-clipped regions (segments=%d, maxDist=%.1f)",
                segall.nr - 2, maxDistance);
    return true;
}

} // namespace rawalchemy
