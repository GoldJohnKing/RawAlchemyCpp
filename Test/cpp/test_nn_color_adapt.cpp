// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../../include/nn_color_adapt.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

int main() {
    using namespace rawalchemy;

    // Canonical XYZ -> linear ProPhoto RGB (ROMM RGB, D50, Bruce Lindbloom).
    // Used here only to derive the expected first column for the identity case.
    static const float PROPHOTO_FROM_XYZ[9] = {
         2.34187440f, -1.02029350f, -0.26219110f,
        -0.98060158f,  1.95377353f,  0.02682407f,
         0.02595534f, -0.09186585f,  1.26542330f
    };

    // Identity cam_xyz (cam==XYZ) -> M = PROPHOTO_FROM_XYZ @ inv(I) = PROPHOTO_FROM_XYZ.
    // camRGB=(1,0,0) [pure X] -> ProPhoto = first column of PROPHOTO_FROM_XYZ
    // = (2.34187440, -0.98060158, 0.02595534). No clamping: out-of-gamut X yields a
    // negative ProPhoto channel, which must be preserved (matches the classical path).
    {
        float identityCamXyz[9] = {1,0,0, 0,1,0, 0,0,1};
        float camRgb[] = {1.0f, 0.0f, 0.0f};
        float out[3];
        camRgbToProPhotoLinear(out, camRgb, 1, identityCamXyz);
        assert(std::fabs(out[0] - PROPHOTO_FROM_XYZ[0]) < 1e-3f);
        assert(std::fabs(out[1] - PROPHOTO_FROM_XYZ[3]) < 1e-3f);
        assert(std::fabs(out[2] - PROPHOTO_FROM_XYZ[6]) < 1e-3f);
    }

    // Real camera matrix -> finite, non-degenerate output.
    // Catches typos/sign errors in the matrix derivation or inversion that would
    // produce NaN or singular output for a realistic cam_xyz.
    {
        float canonCamXyz[9] = {  // Canon 5D2 D65 cam_xyz (3x3 row-major)
            0.5012f, 0.0853f, -0.0169f,
            0.4321f, 0.7896f,  0.3013f,
            0.0667f, 0.1251f,  0.7156f
        };
        float camRgb[] = {0.3f, 0.5f, 0.2f};
        float proPhoto[3];
        camRgbToProPhotoLinear(proPhoto, camRgb, 1, canonCamXyz);
        for (int i = 0; i < 3; ++i) assert(std::isfinite(proPhoto[i]));
    }

    // In-place safe: dst == camRgb must yield the same result as a separate dst.
    // (Tests the documented property directly. Note: a generic gray input does NOT
    // map to equal ProPhoto channels because XYZ->ProPhoto has unequal row sums,
    // so a gray-symmetry assertion would be incorrect.)
    {
        float canonCamXyz[9] = {
            0.5012f, 0.0853f, -0.0169f,
            0.4321f, 0.7896f,  0.3013f,
            0.0667f, 0.1251f,  0.7156f
        };
        float src[] = {0.3f, 0.7f, -0.2f, 0.9f, 0.1f, 0.4f};  // incl. a negative (HDR-like)
        float separate[6];
        camRgbToProPhotoLinear(separate, src, 2, canonCamXyz);
        float inPlace[6];
        std::memcpy(inPlace, src, sizeof(src));
        camRgbToProPhotoLinear(inPlace, inPlace, 2, canonCamXyz);
        for (int i = 0; i < 6; ++i) assert(std::fabs(inPlace[i] - separate[i]) < 1e-6f);
    }

    // Zero is zero (holds for any linear matrix).
    {
        float identityCamXyz[9] = {1,0,0, 0,1,0, 0,0,1};
        float in[] = {0.0f, 0.0f, 0.0f};
        float out[3];
        camRgbToProPhotoLinear(out, in, 1, identityCamXyz);
        assert(out[0] == 0.0f && out[1] == 0.0f && out[2] == 0.0f);
    }

    std::cout << "test_nn_color_adapt: OK\n";
    return 0;
}
