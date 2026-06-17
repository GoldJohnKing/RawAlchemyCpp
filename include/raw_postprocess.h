#pragma once
/**
 * @file raw_postprocess.h
 * @brief Phase 5 RAW postprocessing — white-balance multiply + orientation
 *        flip + camera->working color matrix application.
 *
 * Direct ports of Python reference `raw_alchemy.core`:
 *   - applyFlip()           (core.py:212 -> onnx/denoiser.py:412-431 _apply_flip)
 *
 * Plus applyWhiteBalanceMosaic (v2: pre-demosaic float-mosaic WB, darktable
 * temperature(3.0) equivalent) and the camera->ProPhoto color-matrix
 * application site. WB+matrix operate on a demosaiced camera-RGB ImageBuffer;
 * applyWhiteBalanceMosaic operates on the pre-demosaic float CFA mosaic.
 */

#include "common.h"
#include "raw_mosaic.h"

#include <array>

namespace rawalchemy {

/**
 * @brief Apply camera white balance to a CFA mosaic IN-PLACE (pre-demosaic).
 *
 * New in v2 refactor: clean per-channel gain multiplication on the float
 * mosaic, modeled on darktable's `temperature` iop (position 3.0). This
 * REPLACES the prior design where WB was deferred to post-demosaic via
 * `applyWhiteBalance(ImageBuffer&, ...)` to avoid LibRaw `scale_colors`'s
 * max-anchored magenta cast. Running WB on the float mosaic (not ushort
 * imgdata.image) avoids both the cast AND ushort R/B saturation.
 *
 * Green-anchored: g = cam_mul[1] (or 1.0 if <=0); each photosite's value is
 * scaled by cam_mul[cfaColor]/g where cfaColor is the photosite's CFA color.
 * This is a uniform per-channel multiply — single-channel mosaic, no chroma
 * cross-talk. After this call the mosaic is WHITE-BALANCED (downstream
 * highlights/hotpixels/demosaic all see post-WB data, matching darktable).
 *
 * @param m  Mosaic (modified in-place). Uses m.cam_mul + m.filters/xtrans to
 *           resolve per-photosite channel. cam_mul is NOT modified by this
 *           call (callers that need cam_mul zeroed afterward, e.g. before
 *           highlightInpaintOpposed's virtual-WB path, must do it themselves).
 */
void applyWhiteBalanceMosaic(RawMosaic& m);

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
