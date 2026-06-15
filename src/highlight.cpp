/**
 * @file highlight.cpp
 * @brief Phase 2 highlight reconstruction — segmentation-based "inpaint-opposed".
 *
 * Direct port of Python reference `raw_alchemy.core.highlight_inpaint_opposed`
 * (core.py:53-136) + `raw_alchemy.math_ops._compute_hl_refavg_kernel`
 * (math_ops.py:1182-1230).
 *
 * All morphology / connected-component / max-filter primitives are hand-rolled
 * (no OpenCV) to match the project's minimal-deps stance.
 *
 * Critical oracle-faithful detail: the 7x7 morphological close uses
 * BORDER_CONSTANT=0 for BOTH the dilate and erode passes (erode border pixels
 * with OOB in their 7x7 footprint evaluate to 0). This matches
 * scipy.ndimage.binary_closing(border_value=0) exactly and propagates into
 * the per-segment chroma estimates.
 */

#include "highlight.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#ifdef RA_USE_OPENMP
#include <omp.h>
#endif

namespace rawalchemy {

// CFA color helper (Bayer FC macro / X-Trans 6x6 lookup).
// Returns raw color index (Bayer: 0/1/2/3; X-Trans: 0/1/2).
//
// NOTE: the FC shift must be computed with explicit parens — `<<` binds
// tighter than `|` in C, so the naive transcription
// `(f >> (((r<<1)&14) | (c&1)) << 1) & 3` parses as
// `((f >> (...)) << 1) & 3`, which is wrong. We compute the shift amount
// separately to avoid the precedence trap.
static inline int cfaColorAt(const RawMosaic& m, int r, int c) {
    if (m.filters == 9) {
        return static_cast<int>(m.xtrans[((r % 6) + 6) % 6][((c % 6) + 6) % 6]);
    }
    const unsigned f = m.filters;
    const int shift = ((((r << 1) & 14) | (c & 1)) << 1);
    return static_cast<int>((f >> shift) & 3);
}

// ============================================================
//                  Hand-rolled primitives
// ============================================================

// 7x7 flat-rect dilation, BORDER_CONSTANT=0.
// out[p]=1 iff ANY in-bounds pixel in the 7x7 footprint centered at p is 1.
// (OOB samples are 0; OR with 0 is a no-op, so we simply iterate in-bounds.)
static std::vector<uint8_t> dilate7x7_const0(const uint8_t* in, int W, int H) {
    std::vector<uint8_t> out(static_cast<size_t>(W) * H, 0);
    for (int y = 0; y < H; ++y) {
        const int yLo = std::max(0, y - 3);
        const int yHi = std::min(H - 1, y + 3);
        for (int x = 0; x < W; ++x) {
            const int xLo = std::max(0, x - 3);
            const int xHi = std::min(W - 1, x + 3);
            uint8_t v = 0;
            for (int ny = yLo; ny <= yHi && !v; ++ny) {
                const uint8_t* row = in + static_cast<size_t>(ny) * W;
                for (int nx = xLo; nx <= xHi; ++nx) {
                    if (row[nx]) { v = 1; break; }
                }
            }
            out[static_cast<size_t>(y) * W + x] = v;
        }
    }
    return out;
}

// 7x7 flat-rect erosion, BORDER_CONSTANT=0.
// out[p]=1 ONLY IF the entire 7x7 footprint centered at p is in-bounds AND all
// samples are 1. (OOB samples are 0; AND with 0 forces the output to 0, so
// pixels within 3 of any image edge erode to 0. This is the critical
// cv2 / scipy.binary_closing(border_value=0) edge trap.)
static std::vector<uint8_t> erode7x7_const0(const uint8_t* in, int W, int H) {
    std::vector<uint8_t> out(static_cast<size_t>(W) * H, 0);
    for (int y = 0; y < H; ++y) {
        // Footprint touches OOB -> output 0 (skip the inner scan).
        if (y - 3 < 0 || y + 3 >= H) continue;
        for (int x = 0; x < W; ++x) {
            if (x - 3 < 0 || x + 3 >= W) continue;
            uint8_t v = 1;
            for (int dy = -3; dy <= 3 && v; ++dy) {
                const uint8_t* row = in + static_cast<size_t>(y + dy) * W;
                for (int dx = -3; dx <= 3; ++dx) {
                    if (!row[x + dx]) { v = 0; break; }
                }
            }
            out[static_cast<size_t>(y) * W + x] = v;
        }
    }
    return out;
}

// 7x7 binary morphological close (dilate then erode), BORDER_CONSTANT=0 for
// BOTH passes. See erode7x7_const0 / dilate7x7_const0 above.
static std::vector<uint8_t> morphClose7x7(const uint8_t* in, int W, int H) {
    auto dilated = dilate7x7_const0(in, W, H);
    return erode7x7_const0(dilated.data(), W, H);
}

// 8-connectivity connected components via flood-fill BFS.
// Background (0) stays 0; segments labeled 1..num_seg in raster-scan order.
// Label numbering is order-dependent but downstream stats are order-invariant.
static std::vector<int> connectedComponents8(const uint8_t* bin, int W, int H,
                                              int& num_seg) {
    const size_t N = static_cast<size_t>(W) * H;
    std::vector<int> labels(N, 0);
    num_seg = 0;

    // Reusable BFS queue.
    std::vector<int> queue;
    queue.reserve(N);

    for (int y0 = 0; y0 < H; ++y0) {
        for (int x0 = 0; x0 < W; ++x0) {
            const size_t seed = static_cast<size_t>(y0) * W + x0;
            if (!bin[seed] || labels[seed]) continue;

            ++num_seg;
            const int lbl = num_seg;
            queue.clear();
            queue.push_back(static_cast<int>(seed));
            labels[seed] = lbl;

            size_t head = 0;
            while (head < queue.size()) {
                const int cur = queue[head++];
                const int cy = cur / W;
                const int cx = cur - cy * W;
                const int yLo = std::max(0, cy - 1);
                const int yHi = std::min(H - 1, cy + 1);
                const int xLo = std::max(0, cx - 1);
                const int xHi = std::min(W - 1, cx + 1);
                for (int ny = yLo; ny <= yHi; ++ny) {
                    const size_t rowBase = static_cast<size_t>(ny) * W;
                    for (int nx = xLo; nx <= xHi; ++nx) {
                        const size_t nidx = rowBase + nx;
                        if (!bin[nidx] || labels[nidx]) continue;
                        labels[nidx] = lbl;
                        queue.push_back(static_cast<int>(nidx));
                    }
                }
            }
        }
    }
    return labels;
}

// 7x7 grey dilation (max-filter), OOB treated as 0 (never raises the max).
// OpenMP over rows.
static std::vector<float> greyDilate7x7(const float* in, int W, int H) {
    std::vector<float> out(static_cast<size_t>(W) * H, 0.0f);

    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int y = 0; y < H; ++y) {
        const int yLo = std::max(0, y - 3);
        const int yHi = std::min(H - 1, y + 3);
        float* outRow = out.data() + static_cast<size_t>(y) * W;
        for (int x = 0; x < W; ++x) {
            const int xLo = std::max(0, x - 3);
            const int xHi = std::min(W - 1, x + 3);
            float m = 0.0f;
            for (int ny = yLo; ny <= yHi; ++ny) {
                const float* row = in + static_cast<size_t>(ny) * W;
                for (int nx = xLo; nx <= xHi; ++nx) {
                    if (row[nx] > m) m = row[nx];
                }
            }
            outRow[x] = m;
        }
    }
    return out;
}

// ============================================================
//                  computeHlRefavg (math_ops.py:1182-1230)
// ============================================================
HlRefavg computeHlRefavg(const RawMosaic& m, const uint8_t* color_map,
                          const float wb_gains[3], const float raw_clips[3]) {
    const int H = m.height;
    const int W = m.width;
    const size_t N = static_cast<size_t>(H) * W;

    HlRefavg out;
    out.refavg.assign(N, 0.0f);
    out.clipped.assign(N, 0);

    const float* raw = m.data.data();

    #ifdef RA_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t idx = static_cast<size_t>(y) * W + x;
            const int color = color_map[idx];
            const float val = raw[idx];
            const uint8_t is_clipped = (val >= raw_clips[color]) ? 1 : 0;

            // 3x3 neighborhood, BORDER_REPLICATE (clamp).
            float mean[3] = {0.0f, 0.0f, 0.0f};
            float cnt[3]  = {0.0f, 0.0f, 0.0f};
            for (int dy = -1; dy <= 1; ++dy) {
                int ny = y + dy;
                if (ny < 0) ny = 0; else if (ny >= H) ny = H - 1;
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = x + dx;
                    if (nx < 0) nx = 0; else if (nx >= W) nx = W - 1;
                    const size_t nidx = static_cast<size_t>(ny) * W + nx;
                    float nv = raw[nidx];
                    if (nv < 0.0f) nv = 0.0f;
                    const int nc = color_map[nidx];
                    mean[nc] += nv;
                    cnt[nc]  += 1.0f;
                }
            }

            float cbrt_mean[3] = {0.0f, 0.0f, 0.0f};
            for (int c = 0; c < 3; ++c) {
                if (cnt[c] > 0.0f) {
                    const float arg = wb_gains[c] * mean[c] / cnt[c];
                    cbrt_mean[c] = (arg > 0.0f)
                        ? powf(arg, 1.0f / 3.0f)
                        : 0.0f;
                }
            }

            // opp_cbrt[color] = 0.5 * (sum of cbrt_mean over the other two colors)
            const float opp_cbrt_color =
                0.5f * (cbrt_mean[0] + cbrt_mean[1] + cbrt_mean[2]
                        - cbrt_mean[color]);

            float ref = opp_cbrt_color * opp_cbrt_color * opp_cbrt_color;
            if (wb_gains[color] > 1e-6f) ref /= wb_gains[color];

            out.refavg[idx] = ref;
            out.clipped[idx] = is_clipped;
        }
    }
    return out;
}

// ============================================================
//          highlightInpaintOpposed (core.py:53-136)
// ============================================================
void highlightInpaintOpposed(RawMosaic& m) {
    const int H = m.height;
    const int W = m.width;
    const size_t N = static_cast<size_t>(H) * W;

    // WB gains normalized by green channel.
    const float g = std::max(m.cam_mul[1], 1e-6f);
    const float wb_gains[3] = {
        m.cam_mul[0] / g,
        1.0f,
        m.cam_mul[2] / g,
    };

    const float CLIP = 0.987f;
    float raw_clips[3];
    for (int c = 0; c < 3; ++c) {
        raw_clips[c] = CLIP / std::max(wb_gains[c], 1e-6f);
    }

    // Build color_map: Bayer FC / X-Trans lookup, then collapse >=3 -> 1
    // (X-Trans secondary greens / Bayer G2 -> G). Values in {0,1,2}.
    std::vector<uint8_t> color_map(N);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int c = cfaColorAt(m, y, x);
            if (c >= 3) c = 1;
            color_map[static_cast<size_t>(y) * W + x] = static_cast<uint8_t>(c);
        }
    }

    // Per-pixel opposing-channel reference average + clip mask.
    auto hl = computeHlRefavg(m, color_map.data(), wb_gains, raw_clips);

    // No-op if nothing is clipped.
    bool any_clipped = false;
    for (size_t i = 0; i < N; ++i) {
        if (hl.clipped[i]) { any_clipped = true; break; }
    }
    if (!any_clipped) return;

    // diff = raw - refavg
    std::vector<float> diff(N);
    for (size_t i = 0; i < N; ++i) {
        diff[i] = m.data[i] - hl.refavg[i];
    }

    // Per-color plane segmentation + chroma reconstruction.
    for (int c = 0; c < 3; ++c) {
        // clipped_c = clipped & (color_map == c)
        std::vector<uint8_t> clipped_c(N, 0);
        bool any_c = false;
        for (size_t i = 0; i < N; ++i) {
            if (hl.clipped[i] && color_map[i] == c) {
                clipped_c[i] = 1;
                any_c = true;
            }
        }
        if (!any_c) continue;

        // Morphological close, BORDER_CONSTANT=0 for both passes.
        auto closed = morphClose7x7(clipped_c.data(), W, H);

        // 8-connectivity connected components.
        int num_seg = 0;
        auto labels = connectedComponents8(closed.data(), W, H, num_seg);
        if (num_seg == 0) continue;

        // 7x7 grey dilation (max-filter) on the label image (as float).
        // OOB treated as 0 so border never contributes a higher label.
        std::vector<float> labels_f(N);
        for (size_t i = 0; i < N; ++i) {
            labels_f[i] = static_cast<float>(labels[i]);
        }
        auto expanded = greyDilate7x7(labels_f.data(), W, H);

        // Per-segment chroma accumulation from unclipped border pixels.
        const float lo = raw_clips[c] * 0.2f;
        std::vector<float> seg_sum(static_cast<size_t>(num_seg) + 1, 0.0f);
        std::vector<int>   seg_cnt(static_cast<size_t>(num_seg) + 1, 0);

        for (int y = 0; y < H; ++y) {
            const size_t rowBase = static_cast<size_t>(y) * W;
            for (int x = 0; x < W; ++x) {
                const size_t i = rowBase + x;
                // border = (expanded > 0) && (labels == 0) && unclipped_valid
                if (expanded[i] <= 0.0f) continue;
                if (labels[i] != 0) continue;
                if (color_map[i] != c) continue;
                if (hl.clipped[i]) continue;
                if (!(m.data[i] > lo)) continue;

                const int L = static_cast<int>(expanded[i]);
                if (L < 1 || L > num_seg) continue;
                seg_sum[L] += diff[i];
                seg_cnt[L] += 1;
            }
        }

        // Global chroma fallback.
        int total_cnt = 0;
        float total_sum = 0.0f;
        for (int L = 1; L <= num_seg; ++L) {
            total_cnt += seg_cnt[L];
            total_sum += seg_sum[L];
        }
        const float global_chroma =
            (total_cnt > 100) ? total_sum / static_cast<float>(total_cnt) : 0.0f;

        // Per-segment chroma, falling back to global for under-populated segments.
        std::vector<float> seg_chroma(static_cast<size_t>(num_seg) + 1, 0.0f);
        for (int L = 1; L <= num_seg; ++L) {
            seg_chroma[L] = (seg_cnt[L] > 10)
                ? seg_sum[L] / static_cast<float>(std::max(seg_cnt[L], 1))
                : global_chroma;
        }

        // Replace clipped pixels with max(orig, refavg + seg_chroma[label]).
        for (size_t i = 0; i < N; ++i) {
            if (!clipped_c[i]) continue;
            const int L = labels[i];
            if (L <= 0) continue;
            const float new_val = hl.refavg[i] + seg_chroma[L];
            if (new_val > m.data[i]) m.data[i] = new_val;
        }
    }
}

} // namespace rawalchemy
