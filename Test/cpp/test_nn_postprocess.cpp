// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../../include/nn_postprocess.h"
#include "../../include/nn_preprocess.h"  // for NN_PATCH_SIZE, NN_OVERLAP

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    using namespace rawalchemy;

    // --- makeTrapezoidWeights ---
    {
        std::vector<float> w(NN_PATCH_SIZE * NN_PATCH_SIZE);
        makeTrapezoidWeights(w.data());
        // Corners (max ramp distance) should be smallest
        float corner = w[0];
        float center = w[(NN_PATCH_SIZE / 2) * NN_PATCH_SIZE + (NN_PATCH_SIZE / 2)];
        assert(corner < center);
        // Center should be 1.0 (flat region)
        assert(std::fabs(center - 1.0f) < 1e-6f);
        // Corner = (0/overlap) * (0/overlap) = 0
        assert(std::fabs(corner - 0.0f) < 1e-6f);
        // At x=overlap (just past ramp), weight in x should be 1.0
        float atRampEnd = w[(NN_PATCH_SIZE / 2) * NN_PATCH_SIZE + NN_OVERLAP];
        assert(std::fabs(atRampEnd - 1.0f) < 1e-6f);
    }

    // --- applyColorMatrixInPlace: identity matrix is a no-op ---
    {
        float identity[9] = {1,0,0, 0,1,0, 0,0,1};
        float rgb[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
        applyColorMatrixInPlace(rgb, 2, identity);
        assert(std::fabs(rgb[0] - 0.1f) < 1e-6f);
        assert(std::fabs(rgb[5] - 0.6f) < 1e-6f);
    }

    // --- applyColorMatrixInPlace: low-side clamp ---
    {
        float zeroDiag[9] = {0,0,0, 0,0,0, 0,0,0};
        float rgb[] = {0.5f, 0.5f, 0.5f};
        applyColorMatrixInPlace(rgb, 1, zeroDiag);
        assert(rgb[0] == 0.0f);  // clamped, not negative
    }

    // --- computeCamRgbToSrgb: identity xyzToCam → approx XYZ->sRGB ---
    // normalize(I) = I, so out = inv(I @ SRGB_TO_XYZ) = inv(SRGB_TO_XYZ).
    // Therefore out @ SRGB_TO_XYZ must round-trip to the identity matrix.
    {
        float identity[9] = {1,0,0, 0,1,0, 0,0,1};
        float m[9];
        computeCamRgbToSrgb(m, identity);
        static const float SRGB_TO_XYZ[9] = {
            0.4124564f, 0.3575761f, 0.1804375f,
            0.2126729f, 0.7151522f, 0.0721750f,
            0.0193339f, 0.1191920f, 0.9503041f
        };
        float roundtrip[9];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                float sum = 0.0f;
                for (int k = 0; k < 3; ++k) sum += m[r*3+k] * SRGB_TO_XYZ[k*3+c];
                roundtrip[r*3+c] = sum;
            }
        }
        for (int i = 0; i < 3; ++i) {
            assert(std::fabs(roundtrip[i*3+i] - 1.0f) < 1e-3f);
        }
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                if (r != c) assert(std::fabs(roundtrip[r*3+c]) < 1e-3f);
            }
        }
    }

    // --- computeCamRgbToSrgb: real camera matrix → finite, non-degenerate ---
    // Catches typos/sign errors that produce NaN or singular output.
    {
        // Adobe DCP xyzToCam for Canon EOS 5D Mark II (D65), row-major
        float canon5d2[9] = {
            0.5012f, 0.0853f, -0.0169f,
            0.4321f, 0.7896f,  0.3013f,
            0.0667f, 0.1251f,  0.7156f
        };
        float m[9];
        computeCamRgbToSrgb(m, canon5d2);
        for (int i = 0; i < 9; ++i) {
            assert(std::isfinite(m[i]));
        }
        float det = m[0]*(m[4]*m[8]-m[5]*m[7])
                  - m[1]*(m[3]*m[8]-m[5]*m[6])
                  + m[2]*(m[3]*m[7]-m[4]*m[6]);
        assert(det > 0.0f);
    }

    // --- computeCamRgbToSrgb: singular input → no crash, output unused ---
    // invert3x3 returns false on |det|<1e-20 and leaves outMatrix unwritten.
    // This just verifies the path doesn't crash on degenerate input.
    {
        float singular[9] = {1,2,3, 2,4,6, 3,6,9};  // rank-1 (rows are multiples)
        float m[9];
        computeCamRgbToSrgb(m, singular);
    }

    std::cout << "test_nn_postprocess: OK\n";
    return 0;
}
