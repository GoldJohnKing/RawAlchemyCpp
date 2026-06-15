/**
 * @file highlight_test.cpp
 * @brief Standalone deterministic harness for highlight reconstruction
 *        cross-validation. Builds a synthetic 64x80 RGGB Bayer mosaic,
 *        runs Phase 1 + Phase 2, and writes the resulting float mosaic as
 *        uint16 LE to argv[1] (default /tmp/cpp_hl.bin).
 *
 * Used by Test/cross_validate_highlight.py — the synthetic mosaic GUARANTEES
 * the highlight reconstruction path is exercised even when Sample.NEF has
 * no natural clipping.
 *
 * The construction (gradient formula, clipped block, cam_mul) MUST match
 * Test/cross_validate_highlight.py exactly — both sides build the same input.
 */

#include "highlight.h"
#include "raw_mosaic.h"
#include "raw_preprocess.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static rawalchemy::RawMosaic buildSyntheticMosaic() {
    // Same construction as Test/cross_validate_highlight.py:Test A.
    rawalchemy::RawMosaic m;
    m.width   = 80;
    m.height  = 64;
    m.filters = 0x94949494u;  // LibRaw RGGB Bayer
    m.colors  = 3;
    m.maximum = 1.0f;
    m.cam_mul[0] = 2.0f;      // R
    m.cam_mul[1] = 1.0f;      // G (reference)
    m.cam_mul[2] = 1.5f;      // B
    m.cam_mul[3] = 1.0f;      // (unused G2 — Bayer collapses 3->1)
    for (int c = 0; c < 4; ++c) m.cblack[c] = 0.0f;
    // X-Trans pattern left zeroed (this is Bayer).

    const int W = m.width;
    const int H = m.height;
    m.data.resize(static_cast<size_t>(H) * W);

    // Smooth linear-ish gradient in [0.05, 0.30]. Below the smallest raw_clip
    // (0.4935 for cam_mul[0]=2.0), so only the explicit 1.0 patches below are
    // flagged as clipped.
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t idx = static_cast<size_t>(y) * W + x;
            const float t = static_cast<float>(y * W + x) /
                            static_cast<float>(H * W);
            const float wave = 0.04f * std::sin(0.15f * static_cast<float>(y)) *
                                      std::cos(0.12f * static_cast<float>(x));
            m.data[idx] = 0.05f + 0.25f * t + wave;
        }
    }

    // 12x12 clipped block (rows 20..31, cols 30..41) at 1.0.
    for (int y = 20; y <= 31; ++y) {
        for (int x = 30; x <= 41; ++x) {
            m.data[static_cast<size_t>(y) * W + x] = 1.0f;
        }
    }

    // A few scattered clipped pixels elsewhere (some may be removed by
    // fixHotPixels; both C++ and Python reimplementations agree on that).
    m.data[static_cast<size_t>(5)  * W + 7]  = 1.0f;
    m.data[static_cast<size_t>(50) * W + 60] = 1.0f;
    m.data[static_cast<size_t>(10) * W + 70] = 1.0f;
    m.data[static_cast<size_t>(55) * W + 15] = 1.0f;

    return m;
}

int main(int argc, char* argv[]) {
    const char* out_path = (argc >= 2) ? argv[1] : "/tmp/cpp_hl.bin";

    auto m = buildSyntheticMosaic();

    rawalchemy::subtractBlackLevel(m);   // no-op (cblack=0, maximum=1)
    rawalchemy::fixHotPixels(m);
    rawalchemy::highlightInpaintOpposed(m);

    // Write the float mosaic directly (NOT clamped uint16). Highlight
    // reconstruction intentionally produces super-white values (the recovered
    // detail sits ABOVE the clip threshold — that's the whole point). The
    // uint16 LE + clamp convention used by --dump-mosaic would erase this,
    // so we write float32 here for faithful cross-validation.
    FILE* fp = std::fopen(out_path, "wb");
    if (!fp) {
        std::fprintf(stderr, "Error: cannot write to '%s'\n", out_path);
        return 1;
    }
    const size_t want = m.data.size() * sizeof(float);
    if (std::fwrite(m.data.data(), 1, want, fp) != want) {
        std::fclose(fp);
        std::fprintf(stderr, "Error: short write to '%s'\n", out_path);
        return 1;
    }
    std::fclose(fp);

    // Print "H W" for the cross-validation script (matches --dump-mosaic style).
    std::printf("%d %d\n", m.height, m.width);
    return 0;
}
