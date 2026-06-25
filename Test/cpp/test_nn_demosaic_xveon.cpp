// SPDX-License-Identifier: AGPL-3.0-or-later
// Smoke test for the x-veon NN demosaic dispatch entry point.
//
// The NnDemosaicSession is NOT initialized in this process, so nnDemosaic()
// must short-circuit at the session-readiness gate. This still proves the
// dispatch plumbing, param validation and the entry-point signature compile
// and link against the real ORT + all prior NN modules. A runtime demosaic
// test belongs to Task 8 (needs the real models from Task 9).
#include "../../include/demosaic_nn_xveon.h"

#include <cassert>
#include <iostream>
#include <vector>

int main() {
    using namespace rawalchemy;

    // Build a tiny synthetic RGGB CFA. Dimensions are deliberately below one
    // tile (288x288) — irrelevant, since the session gate fires first.
    const int W = 64;
    const int H = 64;
    std::vector<float> cfa(static_cast<size_t>(W) * H, 0.5f);
    std::vector<float> out(static_cast<size_t>(W) * H * 3, 0.0f);

    NnDemosaicInput in;
    in.width = W;
    in.height = H;
    in.filters = 0x94949494u;  // RGGB
    in.cfa = cfa.data();
    in.blackLevel = 0.0f;
    in.whiteLevel = 1.0f;
    in.wb[0] = in.wb[1] = in.wb[2] = 1.0f;
    // Identity xyzToCam.
    in.xyzToCam[0] = in.xyzToCam[4] = in.xyzToCam[8] = 1.0f;

    // --- SessionNotReady: valid params, but no init() was ever called. ---
    assert(nnDemosaic(in, out.data()) == NnDemosaicStatus::SessionNotReady);

    // --- InvalidInput: null output. ---
    assert(nnDemosaic(in, nullptr) == NnDemosaicStatus::InvalidInput);

    // --- InvalidInput: null CFA. ---
    const float* savedCfa = in.cfa;
    in.cfa = nullptr;
    assert(nnDemosaic(in, out.data()) == NnDemosaicStatus::InvalidInput);
    in.cfa = savedCfa;

    // --- InvalidInput: zero width. ---
    const int savedW = in.width;
    in.width = 0;
    assert(nnDemosaic(in, out.data()) == NnDemosaicStatus::InvalidInput);
    in.width = savedW;

    // --- InvalidInput: negative height. ---
    const int savedH = in.height;
    in.height = -4;
    assert(nnDemosaic(in, out.data()) == NnDemosaicStatus::InvalidInput);
    in.height = savedH;

    std::cout << "test_nn_demosaic_xveon: OK\n";
    return 0;
}
