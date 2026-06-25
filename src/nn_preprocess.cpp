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

// Canonical X-Trans 6x6 pattern (Fujifilm standard). 0=R, 1=G, 2=B.
static const int XTRANS_CANONICAL[6][6] = {
    {1, 2, 1, 1, 0, 1},
    {0, 1, 0, 2, 1, 2},
    {1, 2, 1, 1, 0, 1},
    {1, 0, 1, 1, 2, 1},
    {2, 1, 2, 0, 1, 0},
    {1, 0, 1, 1, 2, 1}
};

// Canonical RGGB 2x2. 0=R, 1=G, 2=B.
static const int RGGB_2x2[2][2] = {{0, 1}, {1, 2}};

int canonicalCfaColor(int y, int x, const CfaPhase& phase) {
    if (phase.isXtrans) {
        // C++ % may be negative for negative operands; normalize to [0, period).
        const int py = ((y % 6) + 6) % 6;
        const int px = ((x % 6) + 6) % 6;
        return XTRANS_CANONICAL[py][px];
    }
    const int py = ((y % 2) + 2) % 2;
    const int px = ((x % 2) + 2) % 2;
    return RGGB_2x2[py][px];
}

void makeCanonicalMasks(float* outMaskR, float* outMaskG, float* outMaskB,
                        const CfaPhase& phase) {
    for (int y = 0; y < NN_PATCH_SIZE; ++y) {
        for (int x = 0; x < NN_PATCH_SIZE; ++x) {
            int idx = y * NN_PATCH_SIZE + x;
            const int ch = canonicalCfaColor(y, x, phase);
            outMaskR[idx] = (ch == 0) ? 1.0f : 0.0f;
            outMaskG[idx] = (ch == 1) ? 1.0f : 0.0f;
            outMaskB[idx] = (ch == 2) ? 1.0f : 0.0f;
        }
    }
}

void packTileInput(float* outTile4ch,
                   const float* cfaTile,
                   const float* maskR,
                   const float* maskG,
                   const float* maskB) {
    const int N = NN_PATCH_SIZE * NN_PATCH_SIZE;
    // Channel 0: CFA
    for (int i = 0; i < N; ++i) outTile4ch[i] = cfaTile[i];
    // Channel 1: R mask
    for (int i = 0; i < N; ++i) outTile4ch[N + i] = maskR[i];
    // Channel 2: G mask
    for (int i = 0; i < N; ++i) outTile4ch[2 * N + i] = maskG[i];
    // Channel 3: B mask
    for (int i = 0; i < N; ++i) outTile4ch[3 * N + i] = maskB[i];
}

} // namespace rawalchemy
