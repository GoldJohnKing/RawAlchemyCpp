// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file demosaic_rcd.h
 * @brief Ratio Corrected Demosaicing (RCD) for Bayer CFA sensors.
 *
 * darktable-derived implementation. The real port lands in Task 9; today the
 * definition lives in demosaic_dispatch.cpp as a bilinear stub.
 */

namespace rawalchemy {

/// RCD demosaic for Bayer CFA input.
///
/// @param in       single-channel CFA mosaic, float[w*h] in [0,1], preprocessed
/// @param out      planar RGB output, float[3*w*h], layout [R plane | G plane | B plane]
/// @param w, h     image dimensions
/// @param filters  LibRaw CFA bitmask (any value other than 9)
void rcd_demosaic(const float* in, float* out, int w, int h, unsigned filters);

} // namespace rawalchemy
