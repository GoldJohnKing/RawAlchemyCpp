// SPDX-License-Identifier: AGPL-3.0-or-later
//
// SINGLE-CHANNEL À TROUS WAVELET DENOISE.
//
// Ported from darktable src/common/dwt.c (Copyright (C) 2017-2026 darktable
// developers; GNU GPL v3). Original code licensed under GPL-3.0-or-later;
// AGPL-3.0-or-later is one-way compatible per the FSF matrix.
//
// The canonical darktable source lives at .reference/darktable/dwt.c. Only the
// single-channel denoise path is ported (dwt_denoise + the _1ch helpers +
// dwt_interleave_rows); the 4-channel general decomposer and the OpenCL path
// are out of scope.
//
// Helper substitutions:
//   dt_alloc_align_float(n)  -> AlignedVector<float, 64>(n)
//   dt_free_align(p)         -> (RAII destructor)
//   dt_iop_image_fill(b,0,w,h,1) -> std::fill(b, b+w*h, 0.0f)
//   DT_OMP_FOR()             -> #pragma omp for
//   DT_OMP_SIMD()            -> #pragma omp simd
//   MIN(a,b)                 -> std::min(a,b)
//   dwt_interleave_rows      -> interleaveRows (anon namespace)

#include "dwt_denoise.h"
#include "aligned_allocator.h"

#include <algorithm>
#include <cstddef>

#ifdef RA_USE_OPENMP
#include <omp.h>
#endif

namespace rawalchemy {
namespace {

// Cache-friendly row reordering so the vertical pass touches rows already in L2.
// Verbatim port of dwt_interleave_rows (darktable dwt.h).
inline int interleaveRows(int rowid, int height, int stride) {
    if (height <= stride) return rowid;
    const int perPass = (height + stride - 1) / stride;
    const int longPasses = height % stride;
    if (longPasses == 0 || rowid < longPasses * perPass)
        return (rowid / perPass) + stride * (rowid % perPass);
    const int rowid2 = rowid - longPasses * perPass;
    return longPasses + (rowid2 / (perPass - 1)) + stride * (rowid2 % (perPass - 1));
}

// First ("vertical") pass: weighted sum of each row with rows `scale` above and
// below, reflecting at the edges. Reads `in`, writes `out`. Port of
// dwt_denoise_vert_1ch.
void denoiseVert1ch(float* __restrict out, const float* __restrict in,
                    int height, int width, int lev) {
    const int vscale = std::min(1 << lev, height);
#ifdef RA_USE_OPENMP
    #pragma omp for
#endif
    for (int rowid = 0; rowid < height; ++rowid) {
        const int row = interleaveRows(rowid, height, vscale);
        const size_t rowstart = static_cast<size_t>(row) * width;
        const int aboveRow = std::abs(row - vscale);
        const int belowRow = (row + vscale < height) ? (row + vscale)
                                                     : 2 * (height - 1) - (row + vscale);
        const float* __restrict center = in + rowstart;
        const float* __restrict above  = in + static_cast<size_t>(aboveRow) * width;
        const float* __restrict below  = in + static_cast<size_t>(belowRow) * width;
        float* __restrict outrow = out + rowstart;
#ifdef RA_USE_OPENMP
        #pragma omp simd
#endif
        for (int col = 0; col < width; ++col)
            outrow[col] = 2.0f * center[col] + above[col] + below[col];
    }
}

// Second ("horizontal") pass: average each pixel of `coarse` (the vertical-pass
// output buffer, read-only) with its left/right neighbors to form `hat`, then
// split `io` (the image buffer) into the new coarse (written back into `io`)
// and the detail coefficient (soft-thresholded and accumulated into `accum`).
// On the last scale, fold the accumulated surviving detail back into `io` to
// form the denoised result. Port of dwt_denoise_horiz_1ch.
//
// Buffer roles mirror darktable exactly: darktable's signature is
// (out, in, accum) called as (interm, img, details_buf); inside, `coarse`
// aliases `out` (=interm, the vert output) and `details` aliases `in` (=img).
// We rename by role (coarse / io / accum) for clarity but keep the math 1:1.
void denoiseHoriz1ch(const float* __restrict coarse, float* __restrict io,
                     float* __restrict accum,
                     int height, int width, int lev, float thold, bool last) {
    const int hscale = std::min(1 << lev, width);
#ifdef RA_USE_OPENMP
    #pragma omp for
#endif
    for (int row = 0; row < height; ++row) {
        const size_t rowindex = static_cast<size_t>(row) * width;
        const float* __restrict cRow = coarse + rowindex;  // vert output (read for hat)
        float* __restrict ioRow = io + rowindex;           // original->coarse (read+write)
        float* __restrict accumRow = accum + rowindex;

        // Left edge: reflect the left neighbor. coarse[col-hscale] -> coarse[hscale-col].
#ifdef RA_USE_OPENMP
        #pragma omp simd
#endif
        for (int col = 0; col < hscale && col < width; ++col) {
            const float left = cRow[hscale - col];
            const float right = (col + hscale < width) ? cRow[col + hscale]
                                                       : cRow[2 * width - 2 - (col + hscale)];
            const float hat = (2.0f * cRow[col] + left + right) / 16.0f;
            const float diff = ioRow[col] - hat;
            ioRow[col] = hat;
            accumRow[col] += std::max(diff - thold, 0.0f) + std::min(diff + thold, 0.0f);
        }
        // Interior.
#ifdef RA_USE_OPENMP
        #pragma omp simd
#endif
        for (int col = hscale; col < width - hscale; ++col) {
            const float hat = (2.0f * cRow[col] + cRow[col - hscale] + cRow[col + hscale]) / 16.0f;
            const float diff = ioRow[col] - hat;
            ioRow[col] = hat;
            accumRow[col] += std::max(diff - thold, 0.0f) + std::min(diff + thold, 0.0f);
        }
        // Right edge: reflect the right neighbor. coarse[col+hscale] -> coarse[2*width-2-(col+hscale)].
#ifdef RA_USE_OPENMP
        #pragma omp simd
#endif
        for (int col = std::max(hscale, width - hscale); col < width; ++col) {
            const float right = cRow[2 * width - 2 - (col + hscale)];
            const float hat = (2.0f * cRow[col] + cRow[col - hscale] + right) / 16.0f;
            const float diff = ioRow[col] - hat;
            ioRow[col] = hat;
            accumRow[col] += std::max(diff - thold, 0.0f) + std::min(diff + thold, 0.0f);
        }

        if (last) {
#ifdef RA_USE_OPENMP
            #pragma omp simd
#endif
            for (int col = 0; col < width; ++col)
                ioRow[col] += accumRow[col];
        }
    }
}

} // namespace

void dwt_denoise(float* img, int width, int height, int bands, const float* noise) {
    if (width <= 0 || height <= 0 || bands <= 0) return;
    const size_t n = static_cast<size_t>(width) * height;
    // Two scratch planes in one allocation: [0,n) = detail accumulator,
    // [n,2n) = vertical-pass intermediate. Mirrors darktable's single alloc.
    AlignedVector<float, 64> scratch(2 * n);
    float* accum = scratch.data();
    float* interm = scratch.data() + n;
    std::fill(accum, accum + n, 0.0f);

#ifdef RA_USE_OPENMP
    #pragma omp parallel
#endif
    for (int lev = 0; lev < bands; ++lev) {
        const bool last = (lev + 1) == bands;
        denoiseVert1ch(interm, img, height, width, lev);             // img -> interm
        denoiseHoriz1ch(interm, img, accum, height, width, lev, noise[lev], last);  // interm(read hat), img(read orig+write coarse), accum
    }
    // `img` now holds the denoised result; `scratch` freed by RAII.
}

} // namespace rawalchemy
