// SPDX-License-Identifier: AGPL-3.0-or-later
// Postprocessing primitives for x-veon NN demosaic.
#pragma once
#include <cstddef>
#include "nn_preprocess.h"  // for NN_PATCH_SIZE, NN_OVERLAP

namespace rawalchemy {

/** Fill a NN_PATCH_SIZE × NN_PATCH_SIZE buffer with the 2D trapezoidal blend
 *  weight window: w[y,x] = wy[y] * wx[x], where w1d ramps 0->1 over NN_OVERLAP
 *  px, is flat 1.0 in the center, and ramps 1->0 symmetrically. */
void makeTrapezoidWeights(float* outWeights2d);

/** Compute the camRGB -> sRGB 3x3 matrix from a camera's xyzToCam matrix.
 *  M = inv(xyzToCam @ inv(XYZ_TO_SRGB)), with xyzToCam rows normalized to sum=1.
 *  Output is row-major [3x3]. */
void computeCamRgbToSrgb(float outMatrix[9], const float xyzToCam[9]);

/** Apply a 3x3 color matrix (row-major) to interleaved RGB pixel buffer in place.
 *  Low-side clamps to 0.0f; high side NOT clamped (preserves HDR highlights). */
void applyColorMatrixInPlace(float* rgbInterleaved, size_t pixelCount, const float matrix[9]);

} // namespace rawalchemy
