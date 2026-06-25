// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../../include/nn_preprocess.h"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    using namespace rawalchemy;

    // --- normalizeCfaInPlace ---
    {
        float cfa[] = {100.0f, 200.0f, 300.0f, 400.0f};
        normalizeCfaInPlace(cfa, 4, /*black=*/100.0f, /*white=*/300.0f);
        // (raw-100)/(300-100) = (raw-100)/200
        assert(std::fabs(cfa[0] - 0.0f) < 1e-6f);
        assert(std::fabs(cfa[1] - 0.5f) < 1e-6f);
        assert(std::fabs(cfa[2] - 1.0f) < 1e-6f);
        assert(std::fabs(cfa[3] - 1.5f) < 1e-6f);  // HDR pass-through above 1.0
    }

    // --- detectCfaPhase: RGGB ---
    {
        CfaPhase p = detectCfaPhase(0x94949494u);
        assert(p.isXtrans == false);
        assert(p.period == 2);
        assert(p.dy == 0 && p.dx == 0);  // RGGB already canonical
    }

    // --- detectCfaPhase: BGGR (phase-shifted RGGB) ---
    {
        // BGGR = 0x16161616; R is at (1,1), so canonical origin offset is (1,1)
        CfaPhase p = detectCfaPhase(0x16161616u);
        assert(p.isXtrans == false);
        assert(p.period == 2);
        assert(p.dy == 1 && p.dx == 1);
    }

    // --- detectCfaPhase: X-Trans ---
    {
        CfaPhase p = detectCfaPhase(9);  // X-Trans marker
        assert(p.isXtrans == true);
        assert(p.period == 6);
    }

    std::cout << "test_nn_preprocess: OK\n";
    return 0;
}
