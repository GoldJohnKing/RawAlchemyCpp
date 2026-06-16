/**
 * @file raw_preprocess.cpp
 * @brief Phase 1 RAW preprocessing — black-level subtraction + hot-pixel fix.
 *
 * Direct ports of Python reference `raw_alchemy.core`:
 *   - subtract_black_level()  (core.py:20-31)
 *   - fix_hot_pixels()        (core.py:34-46)
 *
 * The hot-pixel median is hand-rolled (no OpenCV dependency). The plane loop
 * is OpenMP-parallelized to match the existing project style (src/stylize.cpp).
 */

#include "raw_preprocess.h"

#include <algorithm>
#include <cmath>

#ifdef RA_USE_OPENMP
#include <omp.h>
#endif

namespace rawalchemy {

// ---- subtractBlackLevel ----
// Port of Python `subtract_black_level` (core.py:20-31):
//   pat_size = cfa_pattern.shape[0]
//   for r in range(pat_size):
//     for c in range(pat_size):
//       color = cfa_pattern[r, c]
//       bl_c  = float(bl[min(color, len(bl) - 1)])
//       result[r::pat_size, c::pat_size] =
//           np.maximum(sensor_raw[r::pat_size, c::pat_size] - bl_c, 0) /
//           (wl - bl_c)
void subtractBlackLevel(RawMosaic& m) {
    const int patSize = (m.filters == 9) ? 6 : 2;
    const int W = m.width;
    const int H = m.height;
    const float wl = m.maximum;

    // cblack holds 4 collapsed per-channel values. Clamp the color index to
    // 3 (== len(bl) - 1 in the Python oracle where bl has 4 entries).
    for (int r = 0; r < patSize; ++r) {
        for (int c = 0; c < patSize; ++c) {
            const int color = cfaColor(m, r, c);
            const int idx = std::min(color, 3);
            const float bl_c = m.cblack[idx];
            const float denom = wl - bl_c;

            // Guard against degenerate white==black.
            if (denom <= 0.0f) {
                continue;  // leave values untouched; nothing to normalize.
            }

            // result[r::patSize, c::patSize] = max(val - bl_c, 0) / denom
            for (int y = r; y < H; y += patSize) {
                float* row = m.data.data() + static_cast<size_t>(y) * W;
                for (int x = c; x < W; x += patSize) {
                    const float v = row[x] - bl_c;
                    row[x] = (v > 0.0f ? v : 0.0f) / denom;
                }
            }
        }
    }
}

// ---- Hand-rolled 3x3 median (BORDER_REPLICATE) ----
// Returns the median of up to 9 floats; uses std::nth_element. The scratch
// buffer is caller-supplied so the parallel per-pixel loop can keep one on
// each thread's stack.
static inline float median3x3(const float* plane, int W, int H, int y, int x,
                                float* scratch) {
    int n = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        int yy = y + dy;
        if (yy < 0) yy = 0; else if (yy >= H) yy = H - 1;
        for (int dx = -1; dx <= 1; ++dx) {
            int xx = x + dx;
            if (xx < 0) xx = 0; else if (xx >= W) xx = W - 1;
            scratch[n++] = plane[static_cast<size_t>(yy) * W + xx];
        }
    }
    // Median of n values: position n/2 after partial sort.
    std::nth_element(scratch, scratch + n / 2, scratch + n);
    return scratch[n / 2];
}

// ---- fixHotPixels ----
// Port of Python `fix_hot_pixels` (core.py:34-46):
//   pat_size = cfa_pattern.shape[0]
//   for r in range(pat_size):
//     for c in range(pat_size):
//       plane  = raw_norm[r::pat_size, c::pat_size]
//       med    = cv2.medianBlur(plane, 3)         # 3x3 median (oracle: scipy)
//       diff   = np.abs(plane - med)
//       std    = max(np.std(diff), 1e-6)
//       hot    = diff > threshold * std
//       plane[hot] = med[hot]
//
// Parallelization: each CFA plane is processed with the per-pixel median +
// diff/std + hot-replace as three separate #pragma omp parallel for loops.
// Bayer has only 4 CFA offsets (patSize=2 -> 2x2) and X-Trans has 36
// (patSize=6 -> 6x6); parallelizing across the CFA offset grid would
// underutilize threads on Bayer, so the pixel dimension (the median hot
// spot) is the parallel axis instead. The median is per-pixel independent;
// the diff reduction uses an OpenMP reduction clause; the hot-replace is
// per-pixel independent.
//
// Scratch reuse (L15): `plane`/`med` are declared once outside the CFA
// offset loop and resized per plane (no realloc after the first/largest
// plane), avoiding the 2*patSize*patSize allocations per call.
void fixHotPixels(RawMosaic& m, float threshold) {
    const int patSize = (m.filters == 9) ? 6 : 2;
    const int W = m.width;
    const int H = m.height;

    // For each plane (br, bc) in the patSize x patSize CFA offset grid:
    //   planeDims: planeH = ceil((H - br) / patSize), planeW = ceil((W - bc) / patSize)
    // Operate on the mosaic in-place by indexing through the strided subsample.

    // Reusable scratch buffers — resized per plane, never reallocated once the
    // high-water mark is reached.
    std::vector<float> plane, med;

    for (int br = 0; br < patSize; ++br) {
        for (int bc = 0; bc < patSize; ++bc) {
            // Plane dimensions (subsampled grid).
            const int planeH = (H - br + patSize - 1) / patSize;
            const int planeW = (W - bc + patSize - 1) / patSize;
            if (planeH <= 0 || planeW <= 0) continue;

            const size_t planeN = static_cast<size_t>(planeH) * planeW;
            const int total = static_cast<int>(planeN);
            plane.resize(planeN);
            med.resize(planeN);

            // Pull the plane into a contiguous buffer for cache-friendly
            // median filtering, then write back. This matches the Python
            // semantic of `plane = raw_norm[r::pat_size, c::pat_size]`.
            for (int py = 0; py < planeH; ++py) {
                const int y = br + py * patSize;
                const float* srcRow = m.data.data() + static_cast<size_t>(y) * W;
                float* dstRow = plane.data() + static_cast<size_t>(py) * planeW;
                for (int px = 0; px < planeW; ++px) {
                    const int x = bc + px * patSize;
                    dstRow[px] = srcRow[x];
                }
            }

            // 3x3 median per pixel, BORDER_REPLICATE. Each pixel is
            // independent -> parallelize over the plane (the hot spot).
            // scratch lives on each thread's stack.
            const float* planeData = plane.data();
            float* medData = med.data();
            #pragma omp parallel for schedule(static)
            for (int idx = 0; idx < total; ++idx) {
                const int py = idx / planeW;
                const int px = idx - py * planeW;
                float scratch[9];
                medData[idx] = median3x3(planeData, planeW, planeH, py, px, scratch);
            }

            // diff = abs(plane - med); std = max(stddev(diff), 1e-6).
            // Python uses np.std(diff) which is population std (ddof=0).
            // Reuse plane[] to store |diff| (avoids another alloc). The two
            // accumulators are reduced across threads.
            float* diffData = plane.data();
            double sum = 0.0, sumSq = 0.0;
            #pragma omp parallel for schedule(static) reduction(+:sum,sumSq)
            for (int idx = 0; idx < total; ++idx) {
                const float d = std::fabs(diffData[idx] - medData[idx]);
                diffData[idx] = d;
                sum += static_cast<double>(d);
                sumSq += static_cast<double>(d) * d;
            }
            const double mean = sum / static_cast<double>(planeN);
            double variance = sumSq / static_cast<double>(planeN) - mean * mean;
            if (variance < 0.0) variance = 0.0;  // guard against FP rounding
            float std_v = static_cast<float>(std::sqrt(variance));
            if (std_v < 1e-6f) std_v = 1e-6f;
            const float cutoff = threshold * std_v;

            // hot = diff > cutoff; replace hot pixels in the mosaic with median.
            // Non-hot pixels keep their existing mosaic value (untouched in
            // m.data). plane[] currently holds |diff|; med[] holds the median.
            #pragma omp parallel for schedule(static)
            for (int py = 0; py < planeH; ++py) {
                const int y = br + py * patSize;
                float* dstRow = m.data.data() + static_cast<size_t>(y) * W;
                const float* medRow  = medData + static_cast<size_t>(py) * planeW;
                const float* diffRow = diffData + static_cast<size_t>(py) * planeW;
                for (int px = 0; px < planeW; ++px) {
                    if (diffRow[px] > cutoff) {
                        dstRow[bc + px * patSize] = medRow[px];
                    }
                }
            }
        }
    }
}

} // namespace rawalchemy
