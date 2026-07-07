// SPDX-License-Identifier: AGPL-3.0-or-later
// Inpaint-opposed highlight reconstruction for the x-veon NN demosaic pipeline.
//
// This is the step the design (docs/nn-demosaic-design.md §2.3 step 4) specified
// but was originally skipped — see the former deviation comment in
// demosaic_nn_xveon.cpp. It runs on the CFA mosaic in linear sensor space, BEFORE
// white balance and BEFORE mirror-pad, reconstructing clipped sensels so the NN
// model never sees the out-of-distribution (wb_R, 1, wb_B) magenta input that
// fully-clipped highlights would otherwise produce after per-channel WB.
//
// Algorithm: faithful port of darktable's _process_opposed
// (src/iop/hlreconstruct/opposed.c) + _calc_refavg
// (src/iop/hlreconstruct/segbased.c:186-222). Origin: G'MIC team (@garagecoder,
// @Iain) + darktable (@jenshannoschwalm); merged in darktable PR #12578 (2022-10),
// default since darktable 4.2. No published paper; original discussion:
// https://discuss.pixls.us/t/highlight-recovery-teaser/17670
#pragma once

#include "nn_preprocess.h"  // CfaPhase

namespace rawalchemy {

// Design §2.3 step 4 clip factor. darktable's default is 0.987 (highlights_clip_magics
// [OPPOSED]); this pipeline uses 0.93 to catch the reconstruction halo more aggressively.
// Deliberate tuning — verify no midtone desaturation against Test/Sample.NEF; bump
// toward 0.987 if needed. Exposed so the call site names the magic instead of inlining.
constexpr float kNnHighlightClipFactor = 0.93f;

/** Reconstruct clipped CFA sensels in place using the inpaint-opposed algorithm.
 *
 *  Runs POST white-balance and PRE mirror-pad — matches darktable's pipeline order
 *  (rawprepare → WB → highlights → demosaic). The sensor color of original pixel
 *  (y,x) is read via canonicalCfaColor(y + phase.dy, x + phase.dx, phase).
 *
 *  @param cfa         [W*H] float, WB-applied, normalized. Clipped sensels may read
 *                     well above 1.0 (WB amplifies R/B). Reconstructed in place.
 *  @param W,H         active CFA dimensions.
 *  @param phase       CFA phase (Bayer 2×2 or X-Trans 6×6) from detectCfaPhase().
 *  @param clipFactor  base fraction of nominal white (scaled per-channel by wbRgb).
 *  @param wbRgb       per-channel WB multipliers; clips[c] = clipFactor × wbRgb[c].
 *
 *  No-op (early exit) when no sensel reaches the clip threshold, so well-exposed
 *  images pay only the cost of the mask-build scan. Single-threaded; tile inference
 *  dominates end-to-end time and the early-exit keeps the common case cheap. */
void reconstructHighlightsOpposed(float* cfa, int W, int H,
                                 const CfaPhase& phase, float clipFactor,
                                 const float wbRgb[3]);

} // namespace rawalchemy
