// SPDX-License-Identifier: AGPL-3.0-or-later
// Segmentation-based highlight reconstruction for fully-clipped CFA regions.
// Faithful port of darktable src/iop/hlreconstruct/segbased.c (_process_segmentation,
// by Hanno Schwalm, Iain, garagecoder). Runs AFTER inpaint-opposed on the same CFA:
// opposed handles partially-clipped sensels; this reconstructs texture/color in
// regions where ALL channels are clipped (sky blowouts, specular blobs) via
// segmentation + candidate selection + gradient propagation.
//
// Defaults (darktable-equivalent, tunable later): strength=1.0, adaptive recovery,
// no noise injection. Runs single-threaded (one-time per image; tile inference
// dominates). Pre-WB (correction={1,1,1}, uniform clips) — same pipeline position
// as inpaint-opposed.
#pragma once

#include "nn_preprocess.h"  // CfaPhase

namespace rawalchemy {

/** Reconstruct fully-clipped CFA sensels via segmentation-based recovery.
 *  In-place on `cfa` ([W*H] float, normalized [0,1], pre-WB). Returns true if any
 *  reconstruction was performed, false on early-exit (insufficient clipped sensels). */
bool reconstructHighlightsSegmentBased(float* cfa, int W, int H,
                                       const CfaPhase& phase, float clipFactor);

} // namespace rawalchemy
