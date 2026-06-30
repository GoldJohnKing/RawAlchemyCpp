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

/** Reconstruct clipped CFA sensels via segmentation + candidate selection + gradient
 *  propagation. Matches the x-veon reference: reads threshold/refavg/candidates from
 *  `originalCfa` (the pre-opposed snapshot), writes results to `cfa`.
 *
 *  Step g (candidate-based) overrides opposed's reconstruction where a good candidate
 *  exists; opposed's result survives in cfa where it doesn't. Step h adds gradient-
 *  based texture recovery for fully-clipped regions.
 *
 *  @param cfa          [W*H] WB-applied, post-opposed. Modified in place (step g
 *                      overrides + step h recovery add).
 *  @param originalCfa  [W*H] read-only snapshot taken BEFORE opposed ran (post-WB).
 *                      Used for all threshold/refavg/candidate math.
 *  @returns false on early-exit (insufficient clipped sensels). */
bool reconstructHighlightsSegmentBased(float* cfa, const float* originalCfa,
                                       int W, int H, const CfaPhase& phase,
                                       float clipFactor, const float wbRgb[3]);

} // namespace rawalchemy
