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

    std::cout << "test_nn_postprocess: OK\n";
    return 0;
}
