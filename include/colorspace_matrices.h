#pragma once
/**
 * @file colorspace_matrices.h
 * @brief Phase 5 Camera -> working RGB matrix derivation (analytical, no
 *        rawpy.postprocess needed).
 *
 * Direct C++/OpenMP port of Python reference
 * ``raw_alchemy.colorspace_matrices`` (colorspace_matrices.py:1-147), which is
 * itself a port of darktable's ``dt_colorspaces_conversion_matrices_rgb()``
 * from ``src/common/colorspaces.c`` (itself a port of dcraw's ``cam_xyz_coeff()``).
 *
 * All matrix math is done in float64 (``double``). The final Camera -> ProPhoto
 * matrix is cast to float only at the application site (see
 * cameraToProphotoMatrix()).
 */

#include "raw_mosaic.h"

#include <array>
#include <stdexcept>

namespace rawalchemy {

// ============================================================
//                  Reference constants
//  (exact, from colorspace_matrices.py:29-53)
// ============================================================

/// sRGB(D65) -> XYZ(D65) — exactly the matrix darktable uses (constants from
/// colorspaces.c so we match its numerical behaviour bit-for-bit).
/// (colorspace_matrices.py:29-33)
extern const double RGB_TO_XYZ_D65[3][3];

/// ProPhoto-RGB(D50) -> XYZ(D50). Bruce Lindbloom canonical constants.
/// (colorspace_matrices.py:36-40)
extern const double PRO_TO_XYZ_D50[3][3];

/// Bradford chromatic adaptation D65 -> D50.
/// (colorspace_matrices.py:43-47)
extern const double D65_TO_D50_BRADFORD[3][3];

/**
 * @brief ProPhoto(D65) -> XYZ(D65), pre-computed.
 *
 * rawpy's ``rgb_xyz_matrix`` is XYZ_D65 -> cam, so the working-space input
 * also needs to live in D65. Computed as
 *   ``PRO_TO_XYZ_D65 = inverse(D65_TO_D50_BRADFORD) @ PRO_TO_XYZ_D50``
 * (colorspace_matrices.py:51-52). Computed once at first call and cached.
 */
const double (&proToXyzD65())[3][3];

/**
 * @brief Port of darktable's ``dt_colorspaces_pseudoinverse()``
 *        (colorspaces.c:2277).
 *
 * Takes a (size, 3) row-major matrix and returns its (size, 3) pseudoinverse.
 * Functionally equivalent to ``np.linalg.pinv(M).T`` but uses the exact
 * Gauss-Jordan procedure from colorspaces.c:2277 so we match darktable
 * bit-for-bit. NOT a library inverse.
 *
 * @param M       Row-major (size x 3) input matrix (M[k][j], k in [0,size)).
 * @param size    Number of rows of M.
 * @param out     Row-major (size x 3) output buffer (out[k][j]).
 */
void pseudoinverse(const double* const* M, int size, double* out);

// Convenience overload taking a flat (size*3) row-major buffer.
void pseudoinverse(const double* M, int size, double* out);

/**
 * @brief Compute the CAM -> working RGB matrix the dcraw / darktable way.
 *
 * Port of ``cam_to_working_matrix`` (colorspace_matrices.py:87-138).
 *
 * Steps:
 *   1. Drop all-zero rows of ``xyz_to_cam_4x3`` (3-color sensors have a zero
 *      4th row).
 *   2. ``RGB_to_CAM = XYZ_to_CAM @ working_rgb_to_xyz_d65``
 *   3. Row-normalize: ``RGB_to_CAM /= row_sum``; ``daylight_mul = 1/row_sum``
 *      (the daylight-WB anchor — critical).
 *   4. ``cam_to_rgb = pseudoinverse(RGB_to_CAM).T``
 *   5. 4-color fold: ``M[:,1] += cam_to_rgb[:,3]``; return ``cam_to_rgb[:,:3]``.
 *
 * @param xyz_to_cam_4x3            rawpy's rgb_xyz_matrix (4x3, row-major).
 * @param colors                    Sensor color count (3 typical, 4 for RGBE).
 * @param working_rgb_to_xyz_d65    (3x3) working RGB(D65) -> XYZ(D65).
 * @param daylight_mul_out          [out] length-3 daylight WB multipliers.
 * @return  Row-major (3x3) double matrix mapping camera RGB -> working RGB.
 */
std::array<std::array<double, 3>, 3> camToWorkingMatrix(
    const double xyz_to_cam_4x3[4][3],
    int colors,
    const double working_rgb_to_xyz_d65[3][3],
    double daylight_mul_out[3]);

/**
 * @brief Convenience: derive the Camera -> ProPhoto RGB (D65) matrix from a
 *        decoded RawMosaic.
 *
 * Equivalent to ``cam_to_working_matrix(m.cam_xyz, PRO_TO_XYZ_D65)``. The
 * final matrix is cast to float for direct application to an ImageBuffer.
 *
 * @param m  Decoded mosaic (reads m.cam_xyz[4][3] and m.colors).
 * @return   Row-major (3x3) float matrix.
 */
std::array<std::array<float, 3>, 3> cameraToProphotoMatrix(const RawMosaic& m);

} // namespace rawalchemy
