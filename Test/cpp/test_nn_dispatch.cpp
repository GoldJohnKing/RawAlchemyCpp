// SPDX-License-Identifier: AGPL-3.0-or-later
// Integration test for demosaicDispatch() — the central router between the
// classical (RCD/Markesteijn) and neural (x-veon) demosaic paths.
//
// Sub-tests:
//   1. Neural without session init -> SessionNotReady (no models needed).
//   2. Null / zero-dimension input -> InvalidParam (router + NN param gate).
//      Also covers the Classical Plan A stub (InvalidParam regardless of input).
//   3. Gated golden-image demosaic: only runs when env
//      RA_NN_INTEGRATION_TEST=1 AND RA_NN_BAYER_ONNX=<path>. Loads a synthetic
//      64x64 RGGB fixture (Test/data/bayer_test_cfa.bin, produced by
//      generate_fixture.py), initializes the session, demosaics, and asserts
//      Ok + output dims + finite (non-NaN/Inf) output. Skipped otherwise,
//      since the real bayer.onnx is not vendored until Task 9.
//
// Sub-tests 1+2 are runnable on Linux without any models. Sub-test 3 is the
// env-gated golden path. The fixture path is resolved relative to this source
// file via __FILE__ so the build-directory location does not matter.
#include "../../include/demosaic_dispatch.h"
#include "../../include/nn_session.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

// Resolve the Test/data fixture relative to this source file so the test is
// independent of the working directory it is launched from. test_nn_dispatch.cpp
// lives in Test/cpp/, so the fixture is at ../../Test/data/.
std::string fixturePath() {
    const std::string file = __FILE__;
    const size_t slash = file.find_last_of('/');
    const std::string dir = (slash == std::string::npos) ? "." : file.substr(0, slash);
    return dir + "/../../Test/data/bayer_test_cfa.bin";
}

bool envFlag(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '1';
}

} // namespace

int main() {
    using namespace rawalchemy;

    // Tiny synthetic RGGB CFA. Dimensions are deliberately below one NN tile
    // (288x288) — irrelevant for sub-tests 1+2 (the session/param gate fires
    // first), and the fixture used by sub-test 3 is loaded separately.
    const int W = 64;
    const int H = 64;
    std::vector<float> cfa(static_cast<size_t>(W) * H, 0.5f);

    NnDemosaicInput in;
    in.width = W;
    in.height = H;
    in.filters = 0x94949494u;  // RGGB
    in.cfaMosaic = cfa.data();
    in.blackLevel = 0.0f;
    in.whiteLevel = 1.0f;
    in.wbRgb[0] = in.wbRgb[1] = in.wbRgb[2] = 1.0f;
    in.xyzToCam[0] = in.xyzToCam[4] = in.xyzToCam[8] = 1.0f;

    NnDemosaicOutput out;

    // --- Sub-test 1: Neural path without init -> SessionNotReady. ---
    // NnDemosaicSession is never initialized in this process (except in
    // sub-test 3), so the router must surface SessionNotReady verbatim.
    assert(demosaicDispatch(in, out, DemosaicPath::Neural) ==
           NnDemosaicStatus::SessionNotReady);

    // --- Sub-test 2: InvalidParam for null / zero-dimension input. ---
    // Param validation runs before the session gate inside nnDemosaic, so a
    // null CFA returns InvalidParam even though the session is not ready.
    const float* savedCfa = in.cfaMosaic;
    in.cfaMosaic = nullptr;
    assert(demosaicDispatch(in, out, DemosaicPath::Neural) ==
           NnDemosaicStatus::InvalidParam);
    in.cfaMosaic = savedCfa;

    const int savedW = in.width;
    in.width = 0;
    assert(demosaicDispatch(in, out, DemosaicPath::Neural) ==
           NnDemosaicStatus::InvalidParam);
    in.width = savedW;

    const int savedH = in.height;
    in.height = 0;
    assert(demosaicDispatch(in, out, DemosaicPath::Neural) ==
           NnDemosaicStatus::InvalidParam);
    in.height = savedH;

    // Classical path Plan A stub returns InvalidParam regardless of input.
    assert(demosaicDispatch(in, out, DemosaicPath::Classical) ==
           NnDemosaicStatus::InvalidParam);

    // --- Sub-test 3: Gated golden-image demosaic (skipped without models). ---
    if (!envFlag("RA_NN_INTEGRATION_TEST")) {
        std::cout << "test_nn_dispatch: OK (golden path skipped — "
                     "set RA_NN_INTEGRATION_TEST=1 + RA_NN_BAYER_ONNX=path)\n";
        return 0;
    }

    const char* onnxPath = std::getenv("RA_NN_BAYER_ONNX");
    if (onnxPath == nullptr || onnxPath[0] == '\0') {
        std::cerr << "test_nn_dispatch: RA_NN_INTEGRATION_TEST=1 but "
                     "RA_NN_BAYER_ONNX is unset; cannot run golden path\n";
        return 1;
    }

    // Load the synthetic 64x64 RGGB fixture produced by generate_fixture.py.
    const std::string fixture = fixturePath();
    std::ifstream f(fixture, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "test_nn_dispatch: fixture missing: " << fixture
                  << " (run Test/data/generate_fixture.py)\n";
        return 1;
    }
    const std::streamoff bytes = f.tellg();
    constexpr std::streamoff expected = static_cast<std::streamoff>(W) * H * sizeof(float);
    if (bytes != expected) {
        std::cerr << "test_nn_dispatch: fixture size mismatch (" << bytes
                  << " bytes, expected " << expected << ")\n";
        return 1;
    }
    std::vector<float> fixtureCfa(static_cast<size_t>(W) * H);
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(fixtureCfa.data()), bytes);

    // Initialize the session with the bayer model (Linux dev-loop: CPU EP).
    NnSessionConfig cfg;
    cfg.bayerModelPath = onnxPath;
    if (!NnDemosaicSession::instance().init(cfg)) {
        std::cerr << "test_nn_dispatch: NnDemosaicSession::init failed for "
                  << onnxPath << "\n";
        return 1;
    }
    assert(NnDemosaicSession::instance().isReady());

    // Drive the demosaic through the dispatch router (Neural path).
    in.cfaMosaic = fixtureCfa.data();
    const NnDemosaicStatus status =
        demosaicDispatch(in, out, DemosaicPath::Neural);
    assert(status == NnDemosaicStatus::Ok);
    assert(out.width == W);
    assert(out.height == H);
    assert(out.rgbInterleaved.size() == static_cast<size_t>(W) * H * 3);

    // Belt-and-suspenders finiteness check. The pipeline's own NaN guard
    // (nn_nan_guard.cpp) already rejects NaN/Inf output via NaNOutput, so by
    // the time we see Ok the output must be finite. Verify it anyway.
    for (float v : out.rgbInterleaved) {
        assert(std::isfinite(v));
    }

    std::cout << "test_nn_dispatch: OK (golden path exercised)\n";
    return 0;
}
