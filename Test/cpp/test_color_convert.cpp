// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../../include/color_convert.h"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    using namespace rawalchemy;

    // Test 1: identity matrix leaves data unchanged (2-pixel planar buffer)
    {
        const float identity[3][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0}};
        // Layout: [R0,R1 | G0,G1 | B0,B1]
        float rgb[6] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
        applyCameraToProPhoto(rgb, 2, 1, identity);
        assert(rgb[0] == 0.1f && rgb[1] == 0.2f);
        assert(rgb[2] == 0.3f && rgb[3] == 0.4f);
    }

    // Test 2: known matrix multiply (pure red input)
    {
        const float m[3][4] = {{0.5f, 0.25f, 0.25f, 0},
                                {0.25f, 0.5f, 0.25f, 0},
                                {0.25f, 0.25f, 0.5f, 0}};
        float rgb[3] = {1.0f, 0.0f, 0.0f};
        applyCameraToProPhoto(rgb, 1, 1, m);
        // Expected: R=0.5, G=0.25, B=0.25
        assert(std::abs(rgb[0] - 0.5f) < 1e-6f);
        assert(std::abs(rgb[1] - 0.25f) < 1e-6f);
        assert(std::abs(rgb[2] - 0.25f) < 1e-6f);
    }

    // Test 3: planar → interleaved (1x1 case)
    {
        float p1[3] = {0.1f, 0.2f, 0.3f};
        auto buf = interleavePlanarRgb(p1, 1, 1);
        assert(buf.data[0] == 0.1f);
        assert(buf.data[1] == 0.2f);
        assert(buf.data[2] == 0.3f);
    }

    // Test 4: extractCfaFromImage for RGGB Bayer
    // bayerColor() channel indices for RGGB: (0,0)=R→0, (0,1)=Gr→1,
    // (1,0)=Gb→3, (1,1)=B→2. Each pixel's value must live at its channel index
    // in the 4-channel LibRaw image[pixel][4] layout.
    {
        unsigned rggb = 0x94949494u;
        float image[4][4] = {
            {10,  0,  0,  0},   // (0,0)=R  → ch0
            {0,   20, 0,  0},   // (0,1)=Gr → ch1
            {0,   0,  0,  30},  // (1,0)=Gb → ch3
            {0,   0,  40, 0},   // (1,1)=B  → ch2
        };
        float out[4];
        extractCfaFromImage(image, 2, 2, rggb, nullptr, out);
        assert(out[0] == 10.0f);
        assert(out[1] == 20.0f);
        assert(out[2] == 30.0f);
        assert(out[3] == 40.0f);
    }

    std::cout << "test_color_convert: PASS\n";
    return 0;
}
