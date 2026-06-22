// SPDX-License-Identifier: AGPL-3.0-or-later
// Temporary bilinear stubs — real RCD and Markesteijn ports land in Tasks 9 and 13.

#include "demosaic_rcd.h"
#include "demosaic_markesteijn.h"

#include <cstddef>
#include <stdexcept>

namespace rawalchemy {

void rcd_demosaic(const float* in, float* out, int w, int h, unsigned filters) {
    (void)filters;  // stub ignores CFA geometry — produces a flat RGB copy
    if (!in || !out || w <= 0 || h <= 0) {
        throw std::runtime_error("rcd_demosaic stub: invalid parameters");
    }
    const size_t n = static_cast<size_t>(w) * h;
    for (size_t i = 0; i < n; ++i) {
        out[i]             = in[i];  // R = mosaic value
        out[i + n]         = in[i];  // G = mosaic value
        out[i + 2 * n]     = in[i];  // B = mosaic value
    }
}

void markesteijn_demosaic(const float* in, float* out, int w, int h,
                           const unsigned char xtrans[6][6]) {
    (void)xtrans;  // stub ignores CFA geometry — produces a flat RGB copy
    if (!in || !out || w <= 0 || h <= 0) {
        throw std::runtime_error("markesteijn_demosaic stub: invalid parameters");
    }
    const size_t n = static_cast<size_t>(w) * h;
    for (size_t i = 0; i < n; ++i) {
        out[i]             = in[i];
        out[i + n]         = in[i];
        out[i + 2 * n]     = in[i];
    }
}

} // namespace rawalchemy
