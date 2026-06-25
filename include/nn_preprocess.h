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
 *  Values may exceed 1.0 for HDR highlights (no upper clamp). */
void normalizeCfaInPlace(float* cfa, size_t count, float blackLevel, float whiteLevel);

} // namespace rawalchemy
