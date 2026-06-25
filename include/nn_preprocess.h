// SPDX-License-Identifier: AGPL-3.0-or-later
// Pure preprocessing primitives for x-veon NN demosaic.
// Algorithms: see docs/nn-demosaic-design.md §2.3.
#pragma once
#include <cstddef>
#include "cfa_lookup.h"  // for unsigned filters helpers

namespace rawalchemy {

/** Detected CFA phase relative to canonical RGGB (Bayer) or canonical X-Trans 6x6.
 *  `dy`,`dx` is the top-left offset to mirror-pad so the image origin aligns
 *  to the canonical pattern (R at (0,0) for Bayer). */
struct CfaPhase {
    int dy = 0;
    int dx = 0;
    int period = 2;       // 2 for Bayer, 6 for X-Trans
    bool isXtrans = false;
};

/** Detect the CFA family and phase. For Bayer orientations other than RGGB,
 *  returns the offset needed to align to RGGB origin. For X-Trans returns
 *  period=6, dy=dx=0 (LibRaw delivers canonically-aligned X-Trans). */
CfaPhase detectCfaPhase(unsigned filters);

/** In-place CFA normalization: out = (raw - black) / (white - black).
 *  Single black/white level for all sites (per x-veon training, NOT per-site).
 *  Values may exceed 1.0 for HDR highlights (no upper clamp).
 *  If `whiteLevel <= blackLevel`, the buffer is zeroed (avoids NaN/Inf propagation). */
void normalizeCfaInPlace(float* cfa, size_t count, float blackLevel, float whiteLevel);

/** Tile constants — fixed by the static ONNX export (design §2.5). */
static constexpr int NN_PATCH_SIZE = 288;
static constexpr int NN_OVERLAP = 48;
static constexpr int NN_STRIDE = NN_PATCH_SIZE - NN_OVERLAP;  // 240

/** Fill three NN_PATCH_SIZE × NN_PATCH_SIZE planes with one-hot masks of the
 *  CANONICAL CFA pattern (RGGB for Bayer, the standard 6×6 for X-Trans).
 *  The image is assumed already phase-aligned via mirror-pad using CfaPhase. */
void makeCanonicalMasks(float* outMaskR, float* outMaskG, float* outMaskB,
                        const CfaPhase& phase);

/** Assemble the [1,4,288,288] planar NCHW tile input from a CFA tile + 3 masks.
 *  Channel order: [CFA gray, R mask, G mask, B mask]. */
void packTileInput(float* outTile4ch,
                   const float* cfaTile,
                   const float* maskR,
                   const float* maskG,
                   const float* maskB);

} // namespace rawalchemy
