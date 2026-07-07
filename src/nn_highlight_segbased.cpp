// SPDX-License-Identifier: AGPL-3.0-or-later
// Segmentation-based highlight reconstruction — see nn_highlight_segbased.h.
// Faithful port matching the x-veon reference: reads from originalCfa (pre-opposed
// snapshot) for all threshold/refavg/candidate math; writes to cfa. Step g (candidate
// reconstruction) overrides opposed where a good candidate exists; opposed survives
// as fallback where it doesn't. Step h adds gradient-based texture recovery for
// fully-clipped regions.

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
constexpr int kCombine = 0;
constexpr float kCandidating = 0.0f;

constexpr int HL_RGB_PLANES = 3;
constexpr int HL_SEGMENT_PLANES = 4;
constexpr int HL_FLOAT_PLANES = 8;
constexpr int HL_BORDER = 8;

inline float fcube(float a) { return a * a * a; }

inline size_t roundUp2(int x) { return (size_t)((x + 1) & ~1); }

inline size_t rawToPlane(int pwidth, int row, int col) {
    return ((size_t)HL_BORDER + (row / 3)) * pwidth + (col / 3) + HL_BORDER;
}

// Opposed estimate from a 3×3 CFA neighborhood. Reads from `in` (originalCfa).
// darktable segbased.c:186-222, linear=FALSE (cube-root space for step g).
inline float calcRefavgNonlinear(const float* in, int W, int H, int row, int col,
                                 int color, const uint8_t* fc) {
    float mean[3] = {0, 0, 0}, cnt[3] = {0, 0, 0};
    const int y0 = std::max(0, row - 1);
    const int x0 = std::max(0, col - 1);
    const int y1 = std::min(H - 1, row + 1) + 1;
    const int x1 = std::min(W - 1, col + 1) + 1;
    for (int dy = y0; dy < y1; ++dy) {
        const size_t rb = (size_t)dy * W;
        for (int dx = x0; dx < x1; ++dx) {
            const int c = fc[rb + dx];
            mean[c] += std::max(0.0f, in[rb + dx]);
            cnt[c] += 1.0f;
        }
    }
    float cm[3];
    for (int c = 0; c < 3; ++c)
        cm[c] = (cnt[c] > 0.0f) ? std::cbrt(mean[c] / cnt[c]) : 0.0f;
    float ref;
    switch (color) {
        case 0:  ref = 0.5f * (cm[1] + cm[2]); break;
        case 1:  ref = 0.5f * (cm[0] + cm[2]); break;
        default: ref = 0.5f * (cm[0] + cm[1]); break;
    }
    return ref;  // nonlinear (cube-root) mode — step g cubes it after adding offsets
}

float localStdDeviation(const float* p, int w) {
    const int w2 = 2 * w;
    const float av =
        (p[-w2 - 1] + p[-w2] + p[-w2 + 1] +
         p[-w - 2] + p[-w - 1] + p[-w] + p[-w + 1] + p[-w + 2] +
         p[-2] + p[-1] + p[0] + p[1] + p[2] +
         p[w - 2] + p[w - 1] + p[w] + p[w + 1] + p[w + 2] +
         p[w2 - 1] + p[w2] + p[w2 + 1]) / 21.0f;
    auto sq = [](float x) { return x * x; };
    return std::sqrt(
        (sq(p[-w2 - 1] - av) + sq(p[-w2] - av) + sq(p[-w2 + 1] - av) +
         sq(p[-w - 2] - av) + sq(p[-w - 1] - av) + sq(p[-w] - av) + sq(p[-w + 1] - av) + sq(p[-w + 2] - av) +
         sq(p[-2] - av) + sq(p[-1] - av) + sq(p[0] - av) + sq(p[1] - av) + sq(p[2] - av) +
         sq(p[w - 2] - av) + sq(p[w - 1] - av) + sq(p[w] - av) + sq(p[w + 1] - av) + sq(p[w + 2] - av) +
         sq(p[w2 - 1] - av) + sq(p[w2] - av) + sq(p[w2 + 1] - av)) / 21.0f);
}

float calcWeight(const float* s, size_t loc, int w, float clipval) {
    const float smoothness = std::max(0.0f, 1.0f - 10.0f * std::sqrt(localStdDeviation(&s[loc], w)));
    float val = 0.0f;
    for (int y = -1; y < 2; ++y)
        for (int x = -1; x < 2; ++x)
            val += s[loc + (size_t)y * w + x] / 9.0f;
    const float sval = std::max(1.0f, std::pow(std::min(clipval, val) / clipval, 2.0f));
    return sval * smoothness;
}

// Find best candidate per segment. seg.val1[id]=candidate luminance, seg.val2[id]=refavg.
void calcPlaneCandidates(const float* plane, const float* refavg,
                         Segmentation& seg, float cubeClipval, float badlevel) {
    for (uint32_t id = 2; id < (uint32_t)seg.nr; ++id) {
        seg.val1[id] = 0.0f;
        seg.val2[id] = 0.0f;
        if (seg.ymax[id] - seg.ymin[id] <= 2 || seg.xmax[id] - seg.xmin[id] <= 2) continue;
        size_t testref = 0;
        float testweight = 0.0f;
        const int rowLo = std::max(seg.border + 2, seg.ymin[id] - 2);
        const int rowHi = std::min(seg.height - seg.border - 2, seg.ymax[id] + 3);
        const int colLo = std::max(seg.border + 2, seg.xmin[id] - 2);
        const int colHi = std::min(seg.width - seg.border - 2, seg.xmax[id] + 3);
        for (int row = rowLo; row < rowHi; ++row)
            for (int col = colLo; col < colHi; ++col) {
                const size_t pos = (size_t)row * seg.width + col;
                if (getSegmentId(seg, pos) != id) continue;
                if (plane[pos] >= cubeClipval) continue;
                const float borderBonus = (seg.data[pos] & kSegIdMask) ? 1.0f : 0.75f;
                const float wht = calcWeight(plane, pos, seg.width, cubeClipval) * borderBonus;
                if (wht > testweight) { testweight = wht; testref = pos; }
            }
        if (testref != 0 && testweight > 1.0f - badlevel) {
            static const float weights[5][5] = {
                {1, 4, 6, 4, 1}, {4, 16, 24, 16, 4}, {6, 24, 36, 24, 6},
                {4, 16, 24, 16, 4}, {1, 4, 6, 4, 1}};
            float sum = 0.0f, pix = 0.0f;
            for (int y = -2; y < 3; ++y)
                for (int x = -2; x < 3; ++x) {
                    const size_t pos = testref + (size_t)y * seg.width + x;
                    if (plane[pos] < cubeClipval) {
                        sum += plane[pos] * weights[y + 2][x + 2];
                        pix += weights[y + 2][x + 2];
                    }
                }
            const float av = sum / std::max(1.0f, pix);
            if (av > 0.125f * cubeClipval) {
                seg.val1[id] = std::min(cubeClipval, av);
                seg.val2[id] = refavg[testref];
            }
        }
    }
}

void initialGradients(int w, int height, const float* luminance,
                      const float* distance, float* recout) {
    for (int row = HL_BORDER + 2; row < height - HL_BORDER - 2; ++row)
        for (int col = HL_BORDER + 2; col < w - HL_BORDER - 2; ++col) {
            const size_t v = (size_t)row * w + col;
            float g = 0.0f;
            if (distance[v] > 0.0f && distance[v] < 2.0f)
                g = 4.0f * scharrGradient(&luminance[v], w);
            recout[v] = g;
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

float segmentAttenuation(const Segmentation& seg, uint32_t id) {
    return std::min(1.7f, 0.9f + 3.0f / std::max(1.0f, seg.val1[id]));
}

void calcDistanceRing(int xmin, int xmax, int ymin, int ymax,
                      float* recout, const float* distance,
                      float attenuate, float dist, const Segmentation& seg, uint32_t id) {
    for (int row = ymin; row < ymax; ++row)
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
        const int sw = xmax - xmin, sh = ymax - ymin;
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

bool reconstructHighlightsSegmentBased(float* cfa, const float* originalCfa,
                                       int W, int H, const CfaPhase& phase,
                                       float clipFactor, const float wbRgb[3]) {
    std::vector<uint8_t> fc((size_t)W * H);
    for (int y = 0; y < H; ++y) {
        const size_t rb = (size_t)y * W;
        for (int x = 0; x < W; ++x)
            fc[rb + x] = (uint8_t)canonicalCfaColor(y + phase.dy, x + phase.dx, phase);
    }

    const float cubeClipval[3] = {
        std::cbrt(clipFactor * wbRgb[0]), std::cbrt(clipFactor * wbRgb[1]), std::cbrt(clipFactor * wbRgb[2])
    };
    const float clips[3] = { clipFactor * wbRgb[0], clipFactor * wbRgb[1], clipFactor * wbRgb[2] };
    const int xshifter = (!phase.isXtrans && fc[0] == 1) ? 1 : 2;

    const int pwidth = (int)roundUp2(W / 3) + 2 * HL_BORDER;
    const int pheight = (int)roundUp2(H / 3) + 2 * HL_BORDER;
    const size_t pSize = (size_t)pwidth * pheight;

    std::vector<float> fbuffer(HL_FLOAT_PLANES * pSize);
    float* plane[HL_FLOAT_PLANES];
    for (int i = 0; i < HL_FLOAT_PLANES; ++i) plane[i] = fbuffer.data() + (size_t)i * pSize;
    float* refavgP[3] = { plane[4], plane[5], plane[6] };

    Segmentation isegments[HL_SEGMENT_PLANES];
    const int segLimit = std::max(256, W * H / 4000);
    for (int i = 0; i < HL_SEGMENT_PLANES; ++i)
        segmentationInit(isegments[i], pwidth, pheight, HL_BORDER + 1, segLimit);

    // --- Build planes + refavg from originalCfa (pre-opposed), detect clips. ---
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
                    mean[c] += std::max(0.0f, originalCfa[rb + dx]);
                    cnt[c] += 1.0f;
                }
            }
            for (int c = 0; c < 3; ++c)
                mean[c] = (cnt[c] > 0.0f) ? std::cbrt(mean[c] / cnt[c]) : 0.0f;
            const float cubeRefavg[3] = {
                0.5f * (mean[1] + mean[2]), 0.5f * (mean[0] + mean[2]), 0.5f * (mean[0] + mean[1])};
            const size_t o = rawToPlane(pwidth, row, col);
            int allclipped = 0;
            for (int c = 0; c < 3; ++c) {
                plane[c][o] = mean[c];
                refavgP[c][o] = cubeRefavg[c];
                if (mean[c] > cubeClipval[c]) { allclipped++; isegments[c].data[o] = 1; }
            }
            isegments[3].data[o] = (allclipped == 3) ? 1u : 0u;
            if (allclipped == 3) hasAllClipped = true;
            anyclipped += allclipped;
        }
    }

    if (anyclipped < 20) return false;

    for (int i = 0; i < 3; ++i) masksExtendBorder(plane[i], pwidth, pheight, HL_BORDER);
    for (int p = 0; p < 3; ++p) segmentsCombine(isegments[p], kCombine);
    for (int p = 0; p < 3; ++p) segmentizePlane(isegments[p]);
    for (int p = 0; p < 3; ++p) calcPlaneCandidates(plane[p], refavgP[p], isegments[p], cubeClipval[p], kCandidating);

    // --- Step g: candidate-based reconstruction. Reads originalCfa, writes cfa. ---
    // Where a segment has a good candidate, cfa[idx] = max(original, candidate_estimate),
    // overriding opposed. Where no candidate, opposed's result (already in cfa) survives.
    for (int row = 1; row < H - 1; ++row) {
        const size_t rb = (size_t)row * W;
        for (int col = 1; col < W - 1; ++col) {
            const size_t idx = rb + col;
            const float inval = std::max(0.0f, originalCfa[idx]);
            const int color = fc[idx];
            if (inval <= clips[color]) continue;
            const size_t o = rawToPlane(pwidth, row, col);
            const uint32_t pid = getSegmentId(isegments[color], o);
            if (pid <= 1 || pid >= (uint32_t)isegments[color].nr) continue;
            const float candidate = isegments[color].val1[pid];
            if (candidate == 0.0f) continue;
            const float candRef = isegments[color].val2[pid];
            const float refHere = calcRefavgNonlinear(originalCfa, W, H, row, col, color, fc.data());
            const float oval = fcube(refHere + candidate - candRef);
            cfa[idx] = std::max(inval, oval);
            plane[color][o] = oval;
        }
    }

    // --- Step h: full-clip gradient recovery. Planes from originalCfa; adds to cfa. ---
    Segmentation& segall = isegments[3];
    if (!hasAllClipped || kStrength <= 0.0f) {
        nnlog::info("[NN] segbased: step g done, no full-clip recovery needed");
        return true;
    }

    float* distance  = plane[3];
    float* gradient  = plane[4];
    float* luminance = plane[5];
    float* recout    = plane[6];
    float* planeTmp  = plane[7];

    segmentsCombine(segall, kRecoveryClose);
    std::fill(gradient, gradient + pSize, std::min(1.0f, 5.0f * kStrength));
    std::fill(distance, distance + pSize, 0.0f);
    for (int row = segall.border; row < pheight - segall.border; ++row)
        for (int col = segall.border; col < pwidth - segall.border; ++col) {
            const size_t i = (size_t)row * pwidth + col;
            planeTmp[i] = (plane[0][i] * wbRgb[0] + plane[1][i] * wbRgb[1] + plane[2][i] * wbRgb[2]) / 3.0f;
            distance[i] = (segall.data[i] == 1u) ? kDtMax : 0.0f;
        }
    masksExtendBorder(planeTmp, pwidth, pheight, segall.border);
    gaussianBlur1ch(planeTmp, luminance, pwidth, pheight, 1.2f);

    const float maxDistance = distanceTransform(distance, pwidth, pheight);
    if (maxDistance > 3.0f) {
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

        const float dshift = 2.0f + (float)kRecoveryClose;
        for (int row = 1; row < H - 1; ++row) {
            const size_t rb = (size_t)row * W;
            for (int col = 1; col < W - 1; ++col) {
                const size_t idx = rb + col;
                const int c = fc[idx];
                if (std::max(0.0f, originalCfa[idx]) <= clips[c]) continue;
                const size_t o = rawToPlane(pwidth, row, col);
                const float effect = kStrength / (1.0f + std::exp(-(distance[o] - dshift)));
                cfa[idx] += std::max(0.0f, gradient[o] * effect);
            }
        }
    }

    nnlog::info("[NN] segbased: step g + h done (segments R=%d G=%d B=%d all=%d, maxDist=%.1f)",
                isegments[0].nr - 2, isegments[1].nr - 2, isegments[2].nr - 2, isegments[3].nr - 2, maxDistance);
    return true;
}

} // namespace rawalchemy
