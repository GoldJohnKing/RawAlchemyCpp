// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../../include/nn_preprocess.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

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

    // --- detectCfaPhase: GRBG ---
    {
        CfaPhase p = detectCfaPhase(0x61616161u);
        assert(p.isXtrans == false && p.period == 2);
        assert(p.dy == 0 && p.dx == 1);
    }

    // --- detectCfaPhase: GBRG ---
    {
        CfaPhase p = detectCfaPhase(0x49494949u);
        assert(p.isXtrans == false && p.period == 2);
        assert(p.dy == 1 && p.dx == 0);
    }

    // --- detectCfaPhase: X-Trans ---
    {
        CfaPhase p = detectCfaPhase(9);  // X-Trans marker
        assert(p.isXtrans == true);
        assert(p.period == 6);
    }

    // --- makeCanonicalMasks: Bayer RGGB ---
    {
        constexpr int N = NN_PATCH_SIZE * NN_PATCH_SIZE;
        std::vector<float> mr(N), mg(N), mb(N);
        CfaPhase rggb; rggb.period = 2; rggb.isXtrans = false;
        makeCanonicalMasks(mr.data(), mg.data(), mb.data(), rggb);
        // RGGB: (0,0)=R, (0,1)=G, (1,0)=G, (1,1)=B
        int idx00 = 0 * NN_PATCH_SIZE + 0;
        int idx01 = 0 * NN_PATCH_SIZE + 1;
        int idx10 = 1 * NN_PATCH_SIZE + 0;
        int idx11 = 1 * NN_PATCH_SIZE + 1;
        assert(mr[idx00] == 1.0f && mg[idx00] == 0.0f && mb[idx00] == 0.0f);
        assert(mr[idx01] == 0.0f && mg[idx01] == 1.0f && mb[idx01] == 0.0f);
        assert(mr[idx10] == 0.0f && mg[idx10] == 1.0f && mb[idx10] == 0.0f);
        assert(mr[idx11] == 0.0f && mg[idx11] == 0.0f && mb[idx11] == 1.0f);
        // Pattern repeats every 2
        assert(mr[(2 * NN_PATCH_SIZE + 0)] == 1.0f);  // row 2 == row 0
    }

    // --- packTileInput: channel order is [CFA, R, G, B] ---
    {
        constexpr int N = NN_PATCH_SIZE * NN_PATCH_SIZE;
        std::vector<float> cfa(N, 0.42f);
        std::vector<float> mr(N, 0.0f), mg(N, 1.0f), mb(N, 0.0f);
        std::vector<float> out(4 * N);
        packTileInput(out.data(), cfa.data(), mr.data(), mg.data(), mb.data());
        // Channel 0 = CFA
        assert(std::fabs(out[0] - 0.42f) < 1e-6f);
        // Channel 1 = R mask (all 0)
        assert(out[N] == 0.0f);
        // Channel 2 = G mask (all 1)
        assert(out[2 * N] == 1.0f);
        // Channel 3 = B mask (all 0)
        assert(out[3 * N] == 0.0f);
    }

    std::cout << "test_nn_preprocess: OK\n";
    return 0;
}
