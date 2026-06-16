#pragma once
/**
 * @file raw_postprocess.h
 * @brief Phase 5 RAW postprocessing — white-balance multiply + orientation
 *        flip + camera->working color matrix application.
 *
 * Direct ports of Python reference `raw_alchemy.core`:
 *   - applyWhiteBalance()   (core.py:214-216)
 *   - applyFlip()           (core.py:212 -> onnx/denoiser.py:412-431 _apply_flip)
 *
 * Plus the camera->ProPhoto color-matrix application site that consumes the
 * matrix derived in colorspace_matrices.h. These operate on a demosaiced
 * camera-RGB ImageBuffer (H x W x 3), in-place.
 */

#include "common.h"

#include <array>

namespace rawalchemy {

/**
 * @brief Apply camera white balance to a demosaiced RGB image (in-place).
 *
 * Port of Python core.py:214-216. Green channel is the anchor (untouched);
 * R and B are scaled by the green-normalized WB coefficients:
 *   g = cam_mul[1] > 0 ? cam_mul[1] : 1.0
 *   rgb[:, :, 0] *= cam_mul[0] / g
 *   rgb[:, :, 2] *= cam_mul[2] / g
 *
 * This is the WB MULTIPLY on demosaiced RGB — separate from the matrix's
 * daylight-WB row-normalize (which is baked into cameraToProphotoMatrix).
 *
 * @param rgb      Demosaiced camera-RGB image (modified in-place).
 * @param cam_mul  Camera WB coefficients (length 4; only [0],[1],[2] used).
 */
void applyWhiteBalance(ImageBuffer& rgb, const float cam_mul[4]);

/**
 * @brief Apply LibRaw/dcraw orientation flip to a demosaiced RGB image
 *        (in-place).
 *
 * Port of Python onnx/denoiser.py:412-431 ``_apply_flip``. Flip codes match
 * LibRaw/rawpy ``sizes.flip``:
 *   0 = no rotation
 *   3 = 180°
 *   5 = 90° CCW (270° CW)
 *   6 = 90° CW
 *
 * 90° rotations swap H/W. Unknown codes are a no-op with a stderr warning
 * (matches the reference).
 *
 * @param rgb   Demosaiced RGB image (modified in-place; H/W may swap).
 * @param flip  LibRaw flip code.
 */
void applyFlip(ImageBuffer& rgb, int flip);

/**
 * @brief Apply a 3x3 color matrix to every pixel of an RGB image (in-place).
 *
 * Standard row-major 3x3 multiply: out[i] = sum_j M[i][j] * in[j]. Used to
 * apply the camera -> ProPhoto RGB matrix derived by
 * cameraToProphotoMatrix() (which returns std::array<std::array<float,3>,3>).
 * OpenMP-parallelized across pixels.
 *
 * @param rgb  Image buffer (modified in-place).
 * @param M    Row-major 3x3 matrix (M[i][j]).
 */
void applyColorMatrix(ImageBuffer& rgb, const std::array<std::array<float, 3>, 3>& M);

} // namespace rawalchemy
