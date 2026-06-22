// SPDX-License-Identifier: AGPL-3.0-or-later
// Dispatch shims. Real RCD lives in demosaic_rcd.cpp (Task 9).
// markesteijn_demosaic stays a bilinear stub until Task 13.

#include "demosaic_rcd.h"
#include "demosaic_markesteijn.h"

#include <cstddef>
#include <stdexcept>

namespace rawalchemy {

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
