// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file demosaic_markesteijn.h
 * @brief 3-pass Markesteijn demosaicing for X-Trans CFA sensors.
 *
 * darktable-derived implementation. The real port lands in Task 13; today the
 * definition lives in demosaic_dispatch.cpp as a bilinear stub.
 */

namespace rawalchemy {

/// 3-pass Markesteijn demosaic for X-Trans CFA input.
///
/// @param in      single-channel CFA mosaic, float[w*h] in [0,1], preprocessed
/// @param out     planar RGB output, float[3*w*h], layout [R plane | G plane | B plane]
/// @param w, h    image dimensions
/// @param xtrans  6x6 X-Trans pattern (from LibRaw imgdata.idata.xtrans)
void markesteijn_demosaic(const float* in, float* out, int w, int h,
                           const unsigned char xtrans[6][6]);

} // namespace rawalchemy
