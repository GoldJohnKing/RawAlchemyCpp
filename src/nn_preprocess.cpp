// SPDX-License-Identifier: AGPL-3.0-or-later
#include "nn_preprocess.h"

namespace rawalchemy {

CfaPhase detectCfaPhase(unsigned filters) {
    CfaPhase p;
    if (isXtrans(filters)) {
        p.isXtrans = true;
        p.period = 6;
        p.dy = 0;
        p.dx = 0;
        return p;
    }
    // Bayer: find which corner holds R. Canonical RGGB has R at (0,0).
    // bayerColor returns 0 for R, 1 for G in row 0, 2 for B, 3 for G in row 1.
    p.period = 2;
    // Check the 2x2 origin colors
    int c00 = bayerColor(0, 0, filters);  // expected 0 (R) for RGGB
    if (c00 == 0) { p.dy = 0; p.dx = 0; }        // RGGB
    else if (bayerColor(0, 1, filters) == 0) { p.dy = 0; p.dx = 1; }  // GRBG
    else if (bayerColor(1, 0, filters) == 0) { p.dy = 1; p.dx = 0; }  // GBRG
    else { p.dy = 1; p.dx = 1; }                                        // BGGR
    return p;
}

void normalizeCfaInPlace(float* cfa, size_t count, float blackLevel, float whiteLevel) {
    const float range = whiteLevel - blackLevel;
    if (range <= 0.0f) {
        // Degenerate; zero the buffer to avoid div-by-zero.
        for (size_t i = 0; i < count; ++i) cfa[i] = 0.0f;
        return;
    }
    const float invRange = 1.0f / range;
    for (size_t i = 0; i < count; ++i) {
        cfa[i] = (cfa[i] - blackLevel) * invRange;
    }
}

} // namespace rawalchemy
