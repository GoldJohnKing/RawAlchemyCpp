// SPDX-License-Identifier: AGPL-3.0-or-later
// Primaries adaptation: linear camRGB -> linear ProPhoto RGB (D50, ROMM).
// Uses the camera's cam_xyz matrix (same one LibRaw uses in the classical path)
// so the NN path is bit-identical into vLog/LUT. NO sRGB intermediate (avoids
// gamut clipping of wide-gamut saturated colors ProPhoto preserves).
#pragma once
#include <cstddef>

namespace rawalchemy {

/** Convert linear camRGB -> linear ProPhoto RGB using the camera's cam_xyz matrix.
 *  Computes M = PROPHOTO_FROM_XYZ @ inv(normalizeRows(camXyz)) once, applies per-pixel.
 *  `camXyz` is LibRaw's imgdata.color.cam_xyz[0..2][0..2], row-major 3x3 (XYZ->cam).
 *  `dst` may equal `camRgb` (in-place safe). NO clamping (preserve HDR + negatives,
 *  matching the classical LibRaw path). */
void camRgbToProPhotoLinear(float* dst, const float* camRgb, size_t pixelCount,
                            const float camXyz[9]);

} // namespace rawalchemy
