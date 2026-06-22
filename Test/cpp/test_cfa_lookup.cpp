// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../../include/cfa_lookup.h"

#include <cassert>
#include <iostream>

int main() {
    using namespace rawalchemy;

    // Test 1: Standard RGGB Bayer pattern (filters = 0x94949494)
    // Row 0: R G R G ...  Row 1: G B G B ...
    {
        unsigned rggb = 0x94949494u;
        assert(bayerColor(0, 0, rggb) == 0);  // R
        assert(bayerColor(0, 1, rggb) == 1);  // G
        assert(bayerColor(1, 0, rggb) == 3);  // G (other green)
        assert(bayerColor(1, 1, rggb) == 2);  // B
        // Pattern repeats
        assert(bayerColor(2, 0, rggb) == 0);  // R
        assert(bayerColor(100, 100, rggb) == bayerColor(0, 0, rggb));
    }

    // Test 2: X-Trans detection
    {
        assert(isXtrans(9) == true);
        assert(isXtrans(0x94949494u) == false);
    }

    // Test 3: X-Trans color lookup (standard Fujifilm pattern)
    {
        // Standard X-Trans ILC pattern (simplified)
        unsigned char xtrans[6][6] = {
            { 1, 2, 1, 1, 0, 1 },
            { 0, 1, 0, 2, 1, 2 },
            { 1, 2, 1, 1, 0, 1 },
            { 1, 0, 1, 1, 2, 1 },
            { 2, 1, 2, 0, 1, 0 },
            { 1, 0, 1, 1, 2, 1 }
        };
        assert(xtransColor(0, 1, xtrans) == 2);  // B
        assert(xtransColor(1, 0, xtrans) == 0);  // R
        assert(xtransColor(0, 0, xtrans) == 1);  // G
        // Wraps correctly
        assert(xtransColor(6, 6, xtrans) == xtransColor(0, 0, xtrans));
        assert(xtransColor(-1, -1, xtrans) == xtransColor(5, 5, xtrans));
    }

    std::cout << "test_cfa_lookup: PASS\n";
    return 0;
}
