/**
 * @file colorspace_matrices.cpp
 * @brief Phase 5 Camera -> working RGB matrix derivation — C++ port of
 *        colorspace_matrices.py (verbatim math, float64 throughout).
 *
 * All matrix math runs in float64 (``double``). Only cameraToProphotoMatrix()
 * casts the final result to float for application to an ImageBuffer.
 */

#include "colorspace_matrices.h"

namespace rawalchemy {

// ============================================================
//                  Reference constants
//  (exact, from colorspace_matrices.py:29-53)
// ============================================================

const double RGB_TO_XYZ_D65[3][3] = {
    {0.412453, 0.357580, 0.180423},
    {0.212671, 0.715160, 0.072169},
    {0.019334, 0.119193, 0.950227},
};

const double PRO_TO_XYZ_D50[3][3] = {
    {0.7976749, 0.1351917, 0.0313534},
    {0.2880402, 0.7118741, 0.0000857},
    {0.0000000, 0.0000000, 0.8252100},
};

const double D65_TO_D50_BRADFORD[3][3] = {
    { 1.0478112,  0.0228866, -0.0501270},
    { 0.0295424,  0.9904844, -0.0170491},
    {-0.0092345,  0.0150436,  0.7521316},
};

namespace {

/// Cached derived matrices (computed once on first access).
struct DerivedMatrices {
    double d50_to_d65[3][3];
    double pro_to_xyz_d65[3][3];
};

/// Analytical 3x3 matrix inverse (for the Bradford D65<->D50 adaptation).
/// Matches numpy.linalg.inv for 3x3 (cofactor / determinant method).
void inverse3x3(const double M[3][3], double out[3][3]) {
    const double a = M[0][0], b = M[0][1], c = M[0][2];
    const double d = M[1][0], e = M[1][1], f = M[1][2];
    const double g = M[2][0], h = M[2][1], i = M[2][2];

    // Cofactors (transposed adjugate).
    const double A =  (e * i - f * h);
    const double B = -(d * i - f * g);
    const double C =  (d * h - e * g);
    const double D = -(b * i - c * h);
    const double E =  (a * i - c * g);
    const double F = -(a * h - b * g);
    const double G =  (b * f - c * e);
    const double H = -(a * f - c * d);
    const double I =  (a * e - b * d);

    const double det = a * A + b * B + c * C;
    if (det == 0.0) {
        throw std::runtime_error(
            "[colorspace_matrices] inverse3x3: singular matrix");
    }
    const double invDet = 1.0 / det;

    out[0][0] = A * invDet; out[0][1] = D * invDet; out[0][2] = G * invDet;
    out[1][0] = B * invDet; out[1][1] = E * invDet; out[1][2] = H * invDet;
    out[2][0] = C * invDet; out[2][1] = F * invDet; out[2][2] = I * invDet;
}

/// 3x3 matrix multiply: out = A @ B.
void matmul3x3(const double A[3][3], const double B[3][3], double out[3][3]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += A[i][k] * B[k][j];
            out[i][j] = s;
        }
    }
}

/// Lazily compute and cache the derived D50_TO_D65 and PRO_TO_XYZ_D65.
const DerivedMatrices& derivedMatrices() {
    static DerivedMatrices dm = []() {
        DerivedMatrices d;
        double d50_to_d65[3][3];
        inverse3x3(D65_TO_D50_BRADFORD, d50_to_d65);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                d.d50_to_d65[i][j] = d50_to_d65[i][j];
        // PRO_TO_XYZ_D65 = D50_TO_D65 @ PRO_TO_XYZ_D50
        matmul3x3(d50_to_d65, PRO_TO_XYZ_D50, d.pro_to_xyz_d65);
        return d;
    }();
    return dm;
}

} // namespace

const double (&proToXyzD65())[3][3] {
    return derivedMatrices().pro_to_xyz_d65;
}

// ============================================================
//                  Pseudoinverse (Gauss-Jordan)
//  Port of _pseudoinverse (colorspace_matrices.py:56-84) verbatim.
// ============================================================
//
// Takes a (size, 3) row-major matrix and returns its (size, 3) pseudoinverse.
// Uses the exact Gauss-Jordan procedure from colorspaces.c:2277 so we match
// darktable bit-for-bit. NOT np.linalg.pinv.

// Pointer-of-pointer overload (M[k] points to row k of length 3).
void pseudoinverse(const double* const* M, int size, double* out) {
    // work is a (3, 6) scratch buffer (row-major).
    double work[3][6] = {{0}};
    for (int i = 0; i < 3; ++i) {
        work[i][i + 3] = 1.0;
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < size; ++k) {
                work[i][j] += M[k][i] * M[k][j];
            }
        }
    }
    for (int i = 0; i < 3; ++i) {
        double num = work[i][i];
        for (int j = 0; j < 6; ++j) work[i][j] /= num;
        for (int k = 0; k < 3; ++k) {
            if (k == i) continue;
            num = work[k][i];
            for (int j = 0; j < 6; ++j) work[k][j] -= work[i][j] * num;
        }
    }
    // out is (size, 3) row-major: out[i*size_stride + j] indexed as out[i*3+j]
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) {
                s += work[j][k + 3] * M[i][k];
            }
            out[i * 3 + j] = s;
        }
    }
}

// Flat row-major (size*3) overload.
void pseudoinverse(const double* M, int size, double* out) {
    // Build an array of row pointers for the pointer-of-pointer overload.
    const double* rows[4];
    for (int k = 0; k < size; ++k) rows[k] = M + k * 3;
    pseudoinverse(rows, size, out);
}

// ============================================================
//                cam_to_working_matrix
//  Port of cam_to_working_matrix (colorspace_matrices.py:87-138) verbatim.
// ============================================================
std::array<std::array<double, 3>, 3> camToWorkingMatrix(
    const double xyz_to_cam_4x3[4][3],
    int colors,
    const double working_rgb_to_xyz_d65[3][3],
    double daylight_mul_out[3]) {

    (void)colors;  // The Python uses any(axis=1) to drop zero rows, not `colors`.

    // Step 0: drop all-zero rows (3-color cameras have a zero 4th row).
    // `valid[k]` is true iff row k has any non-zero element.
    bool valid[4] = {false, false, false, false};
    int n = 0;
    for (int k = 0; k < 4; ++k) {
        bool any = false;
        for (int j = 0; j < 3; ++j) {
            if (xyz_to_cam_4x3[k][j] != 0.0) { any = true; break; }
        }
        valid[k] = any;
        if (any) ++n;
    }
    if (n < 3) {
        throw std::runtime_error(
            "[colorspace_matrices] XYZ_to_CAM has only " +
            std::to_string(n) + " valid rows; need at least 3.");
    }

    // XYZ_to_CAM is the (n, 3) valid-row submatrix.
    double XYZ_to_CAM[4][3];
    int rowIdx = 0;
    for (int k = 0; k < 4; ++k) {
        if (!valid[k]) continue;
        for (int j = 0; j < 3; ++j) XYZ_to_CAM[rowIdx][j] = xyz_to_cam_4x3[k][j];
        ++rowIdx;
    }

    // Step 1: RGB(D65) -> CAM = XYZ_to_CAM @ working_rgb_to_xyz_d65.
    double RGB_to_CAM[4][3];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) {
                s += XYZ_to_CAM[i][k] * working_rgb_to_xyz_d65[k][j];
            }
            RGB_to_CAM[i][j] = s;
        }
    }

    // Step 2: row-normalize so RGB_to_CAM @ (1,1,1) == (1,...,1).
    // daylight_mul = 1 / row_sum (the daylight WB anchor).
    double daylight_mul[4];
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < 3; ++j) s += RGB_to_CAM[i][j];
        RGB_to_CAM[i][0] /= s;
        RGB_to_CAM[i][1] /= s;
        RGB_to_CAM[i][2] /= s;
        daylight_mul[i] = 1.0 / s;
    }

    // Step 3: pseudoinverse then transpose (per colorspaces.c:2417).
    // inv has shape (n, 3); cam_to_rgb = inv.T has shape (3, n).
    double inv[4][3];
    pseudoinverse(&RGB_to_CAM[0][0], n, &inv[0][0]);

    // cam_to_rgb[i][j] = inv[j][i]  (transpose)
    double cam_to_rgb[3][4];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < n; ++j)
            cam_to_rgb[i][j] = inv[j][i];

    std::array<std::array<double, 3>, 3> result{};
    if (n == 4) {
        // 4-color sensor (RGBG): fold the second green into G.
        for (int i = 0; i < 3; ++i) {
            result[i][0] = cam_to_rgb[i][0];
            result[i][1] = cam_to_rgb[i][1] + cam_to_rgb[i][3];
            result[i][2] = cam_to_rgb[i][2];
        }
        daylight_mul_out[0] = daylight_mul[0];
        daylight_mul_out[1] = daylight_mul[1];
        daylight_mul_out[2] = daylight_mul[2];
    } else {
        for (int i = 0; i < 3; ++i) {
            result[i][0] = cam_to_rgb[i][0];
            result[i][1] = cam_to_rgb[i][1];
            result[i][2] = cam_to_rgb[i][2];
        }
        daylight_mul_out[0] = daylight_mul[0];
        daylight_mul_out[1] = daylight_mul[1];
        daylight_mul_out[2] = daylight_mul[2];
    }

    return result;
}

// ============================================================
//                cameraToProphotoMatrix
//  Convenience: Camera -> ProPhoto RGB (D65) for a decoded RawMosaic.
// ============================================================
std::array<std::array<float, 3>, 3> cameraToProphotoMatrix(const RawMosaic& m) {
    double daylight_mul[3];
    auto Md = camToWorkingMatrix(m.cam_xyz, m.colors, proToXyzD65(), daylight_mul);
    std::array<std::array<float, 3>, 3> result{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            result[i][j] = static_cast<float>(Md[i][j]);
    return result;
}

} // namespace rawalchemy
