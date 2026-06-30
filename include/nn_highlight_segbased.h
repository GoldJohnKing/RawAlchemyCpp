// SPDX-License-Identifier: AGPL-3.0-or-later
// Segmentation-based highlight reconstruction for fully-clipped CFA regions.
// Slimmed port of darktable's segbased.c — keeps ONLY the full-clip recovery path
// (step h). The partial-clip candidate reconstruction (step g) was removed because
// inpaint-opposed already handles partial clips; running both was redundant.
//
// What remains: build 1/3-res color planes → detect all-clipped superpixels →
// segment them → distance-transform → propagate boundary gradients inward →
// box-blur ridge suppression → add recovered texture to clipped sensels.
// Faithful to darktable's _process_segmentation recovery path (segbased.c:621-700).
#pragma once

#include "nn_preprocess.h"  // CfaPhase

namespace rawalchemy {

/** Recover texture in fully-clipped CFA regions via gradient propagation.
 *  In-place on `cfa`. No-op (returns false) when there aren't enough all-clipped
 *  sensels to bother. Only handles regions where ALL 3 channels are clipped —
 *  partial clips are left to inpaint-opposed. */
bool reconstructHighlightsSegmentBased(float* cfa, int W, int H,
                                       const CfaPhase& phase, float clipFactor);

} // namespace rawalchemy
