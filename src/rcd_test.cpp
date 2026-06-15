/**
 * @file rcd_test.cpp
 * @brief Standalone RCD demosaic cross-validation harness.
 *
 * Builds one of several known 256x256 RGGB Bayer mosaics in memory, runs
 * rcdDemosaic, and writes the resulting camera-RGB ImageBuffer as flat
 * float32 binary (row-major RGB interleaved) to the output path given on the
 * command line. Prints "256 256" to stdout.
 *
 * Usage (two equivalent forms — the second is backward-compatible with the
 * original single-arg Test A invocation):
 *   raw_alchemy_rcd_test <pattern> <out.bin>   # explicit pattern (Test B)
 *   raw_alchemy_rcd_test <out.bin>             # legacy form, pattern="smooth"
 *
 * Patterns (all 256x256, RGGB filters=0x94949494):
 *   - smooth : R=col/255, G=row/255, B=(row+col)/510. Used by
 *              Test/cross_validate_demosaic.py:Test A (known-truth gate). The
 *              construction MUST match the Python oracle EXACTLY.
 *   - edge-h : vertical step edge at col=128 (R=G: 0.1/0.9, B: 0.05/0.5).
 *              Tests horizontal-direction demosaic (V/H selection in RCD).
 *   - edge-v : horizontal step edge at row=128 (same levels). Tests
 *              vertical-direction demosaic.
 *   - edge-d : 45-degree diagonal step edge at (row+col)=256 (all channels
 *              0.1/0.9). Tests P/Q diagonal discrimination in RCD.
 *
 * The smooth pattern is degenerate for RCD's direction-selection logic
 * (VH_Disc and PQ_Disc are all ~0.5 on smooth data, so V/H or P/Q swap bugs
 * are invisible). The edge patterns are required to exercise that logic —
 * see Test/cross_validate_demosaic.py:Test B.
 */

#include "demosaic.h"
#include "raw_mosaic.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static rawalchemy::RawMosaic buildSyntheticRggbMosaic(const char* pattern) {
    rawalchemy::RawMosaic m;
    m.width   = 256;
    m.height  = 256;
    m.filters = 0x94949494u;  // RGGB Bayer
    m.colors  = 3;
    m.maximum = 1.0f;
    for (int c = 0; c < 4; ++c) m.cblack[c] = 0.0f;
    m.xtrans[0][0] = 0;  // unused (Bayer); left zeroed explicitly.

    const int W = m.width;
    const int H = m.height;
    m.data.resize(static_cast<size_t>(H) * W, 0.0f);

    // Per-pattern per-pixel (R,G,B) construction (all in [0,1]). Subsample to
    // RGGB: keep only the CFA color's value at each pixel, set others to 0
    // (rcdPopulate does max(.,0) so 0-fill is fine).
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            float R = 0.0f, G = 0.0f, B = 0.0f;
            if (std::strcmp(pattern, "smooth") == 0) {
                // Known-truth gradient (matches Test A's Python oracle EXACTLY).
                R = static_cast<float>(col) / 255.0f;
                G = static_cast<float>(row) / 255.0f;
                B = static_cast<float>(row + col) / 510.0f;
            } else if (std::strcmp(pattern, "edge-h") == 0) {
                // Vertical edge line at col=128 (column discontinuity) —
                // exercises horizontal-direction demosaic.
                const bool hi = (col >= 128);
                R = hi ? 0.9f : 0.1f;
                G = hi ? 0.9f : 0.1f;
                B = hi ? 0.5f : 0.05f;
            } else if (std::strcmp(pattern, "edge-v") == 0) {
                // Horizontal edge line at row=128 (row discontinuity) —
                // exercises vertical-direction demosaic.
                const bool hi = (row >= 128);
                R = hi ? 0.9f : 0.1f;
                G = hi ? 0.9f : 0.1f;
                B = hi ? 0.5f : 0.05f;
            } else if (std::strcmp(pattern, "edge-d") == 0) {
                // 45-degree diagonal edge at (row+col)=256 — exercises P/Q
                // diagonal discrimination.
                const bool hi = ((row + col) >= 256);
                const float v = hi ? 0.9f : 0.1f;
                R = G = B = v;
            } else {
                std::fprintf(stderr, "Error: unknown pattern '%s' "
                            "(expected: smooth|edge-h|edge-v|edge-d)\n",
                            pattern);
                std::exit(2);
            }

            const int color = rawalchemy::cfaColor(m, row, col);  // 0=R,1=G,2=B,3=G2
            float v = 0.0f;
            if (color == 0)                   v = R;
            else if (color == 1 || color == 3) v = G;  // G1 and G2
            else                              v = B;  // color == 2
            m.data[static_cast<size_t>(row) * W + col] = v;
        }
    }
    return m;
}

int main(int argc, char* argv[]) {
    // Two invocation forms (backward compatible):
    //   raw_alchemy_rcd_test <pattern> <out.bin>   (new form)
    //   raw_alchemy_rcd_test <out.bin>             (legacy form -> "smooth")
    const char* pattern  = "smooth";
    const char* out_path = "/tmp/cpp_rcd.bin";
    if (argc >= 3) {
        pattern  = argv[1];
        out_path = argv[2];
    } else if (argc == 2) {
        out_path = argv[1];
    } else {
        std::fprintf(stderr, "Usage: %s [<pattern>] <out.bin>\n"
                            "  pattern: smooth | edge-h | edge-v | edge-d\n",
                            argv[0]);
        return 1;
    }

    auto m = buildSyntheticRggbMosaic(pattern);

    // rcdDemosaic expects normalized [0,1] input; our synthetic mosaic already
    // is (cblack=0, maximum=1 -> subtractBlackLevel is a no-op, but we don't
    // even need to call it). Run demosaic directly.
    auto img = rawalchemy::rcdDemosaic(m);

    // Write float32 RGB interleaved (H*W*3 floats, row-major).
    FILE* fp = std::fopen(out_path, "wb");
    if (!fp) {
        std::fprintf(stderr, "Error: cannot write to '%s'\n", out_path);
        return 1;
    }
    const size_t want = img.data.size() * sizeof(float);
    if (std::fwrite(img.data.data(), 1, want, fp) != want) {
        std::fclose(fp);
        std::fprintf(stderr, "Error: short write to '%s'\n", out_path);
        return 1;
    }
    std::fclose(fp);

    // Print "H W" for the cross-validation script.
    std::printf("%d %d\n", img.height, img.width);
    return 0;
}
