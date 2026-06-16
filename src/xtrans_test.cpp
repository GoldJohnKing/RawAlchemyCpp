/**
 * @file xtrans_test.cpp
 * @brief Standalone X-Trans Markesteijn demosaic cross-validation harness.
 *
 * Three modes (selected by argv[1]):
 *
 *   1. <path>            Demosaic a synthetic X-Trans mosaic and write the
 *                        resulting camera-RGB ImageBuffer (H*W*3 float32,
 *                        RGB interleaved) to <path>. Prints "H W" to stdout.
 *                        Pattern type via argv[2]: "smooth" (default),
 *                        "edge-h", "edge-v", "edge-d".
 *                        Size via argv[3] (default 300). Used by Test A/B.
 *
 *   2. --allhex          Print the allhex_dr[72] / allhex_dc[72] / sgrow /
 *                        sgcol for the standard X-Trans pattern, one int per
 *                        line, prefixed. Used by Test C (exact oracle).
 *
 *   3. --smoke <path> [size]
 *                        Large synthetic run (default 1200x1200) for smoke
 *                        timing/memory. Writes float32 RGB to <path>.
 *
 * The X-Trans pattern used is the CANONICAL Fujifilm 6x6 pattern (the
 * Raw-Alchemy reference's XTRANS_PATTERN, 8R/20G/8B per 6x6). NOTE: the
 * Phase-4 spec sheet listed a different 6x6 pattern that turned out to be
 * malformed (7R/23G/6B — not a valid X-Trans); the canonical pattern from
 * the actual Python reference (raw_alchemy.rawspeed.XTRANS_PATTERN) is used
 * instead. This is the pattern the real xtrans_markesteijn_demosaic runs
 * against, so the C++ port must match it.
 *
 * Used by Test/cross_validate_demosaic_xtrans.py (Tests A/B/C + smoke).
 */

#include "demosaic.h"
#include "raw_mosaic.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Canonical Fujifilm X-Trans 6x6 CFA pattern (Raw-Alchemy XTRANS_PATTERN).
// 0=R, 1=G, 2=B. 8R / 20G / 8B per 6x6 — the correct X-Trans distribution.
constexpr char kXtransStandard[6][6] = {
    {1, 1, 0, 1, 1, 2},
    {1, 1, 2, 1, 1, 0},
    {2, 0, 1, 0, 2, 1},
    {1, 1, 2, 1, 1, 0},
    {1, 1, 0, 1, 1, 2},
    {0, 2, 1, 2, 0, 1},
};

rawalchemy::RawMosaic makeXtransMosaic(int W, int H) {
    rawalchemy::RawMosaic m;
    m.width   = W;
    m.height  = H;
    m.filters = 9;  // X-Trans sentinel
    m.colors  = 3;
    m.maximum = 1.0f;
    for (int c = 0; c < 4; ++c) m.cblack[c] = 0.0f;
    std::memcpy(m.xtrans, kXtransStandard, sizeof(kXtransStandard));
    m.data.assign(static_cast<size_t>(H) * W, 0.0f);
    return m;
}

// FC helper matching the C++ port's FCxt (the mosaic isn't built yet during
// mosaic construction, but m.xtrans is filled before data, so this works).
int fcAt(const rawalchemy::RawMosaic& m, int r, int c) {
    const auto& xt = m.xtrans;
    return static_cast<int>(xt[((r % 6) + 6) % 6][((c % 6) + 6) % 6]);
}

// ------------------------------------------------------------------
// Synthetic RGB -> X-Trans mosaic subsampling.
// truth is filled with the full RGB image (H*W*3, row-major RGB interleaved).
// ------------------------------------------------------------------

// Pattern "smooth": R = col/(W-1), G = row/(H-1), B = (R+G)/2. All in [0,1].
void buildSmooth(const rawalchemy::RawMosaic& m, float* truth) {
    const int W = m.width, H = m.height;
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            const float R = static_cast<float>(col) / static_cast<float>(W - 1);
            const float G = static_cast<float>(row) / static_cast<float>(H - 1);
            const float B = (R + G) * 0.5f;
            float* px = truth + (static_cast<size_t>(row) * W + col) * 3;
            px[0] = R; px[1] = G; px[2] = B;
        }
    }
}

// Horizontal step edge: left half dark, right half bright. Edge runs
// vertically at col=W/2. All channels carry the same grey edge so color
// doesn't confound the structural checks. Used for zipper / edge-position /
// channel-agreement tests.
void buildEdgeH(const rawalchemy::RawMosaic& m, float* truth) {
    const int W = m.width, H = m.height;
    const float dark = 0.1f, bright = 0.9f;
    const int half = W / 2;
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            const float v = (col < half) ? dark : bright;
            float* px = truth + (static_cast<size_t>(row) * W + col) * 3;
            px[0] = v; px[1] = v; px[2] = v;
        }
    }
}

// Vertical step edge: top half dark, bottom half bright. Edge runs
// horizontally at row=H/2.
void buildEdgeV(const rawalchemy::RawMosaic& m, float* truth) {
    const int W = m.width, H = m.height;
    const float dark = 0.1f, bright = 0.9f;
    const int half = H / 2;
    for (int row = 0; row < H; ++row) {
        const float v = (row < half) ? dark : bright;
        for (int col = 0; col < W; ++col) {
            float* px = truth + (static_cast<size_t>(row) * W + col) * 3;
            px[0] = v; px[1] = v; px[2] = v;
        }
    }
}

// 45-degree diagonal step edge: bright where (row+col) < (H+W)/2, else dark.
void buildEdgeD(const rawalchemy::RawMosaic& m, float* truth) {
    const int W = m.width, H = m.height;
    const float dark = 0.1f, bright = 0.9f;
    const int thresh = (H + W) / 2;
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            const float v = ((row + col) < thresh) ? bright : dark;
            float* px = truth + (static_cast<size_t>(row) * W + col) * 3;
            px[0] = v; px[1] = v; px[2] = v;
        }
    }
}

// Subsample a full-RGB truth image into the X-Trans mosaic (keep only the
// CFA color's value at each pixel; m.data is single-channel).
void subsampleToMosaic(rawalchemy::RawMosaic& m, const float* truth) {
    const int W = m.width, H = m.height;
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            const int c = fcAt(m, row, col);
            const float v = truth[(static_cast<size_t>(row) * W + col) * 3 + c];
            m.data[static_cast<size_t>(row) * W + col] = v;
        }
    }
}

bool writeFloat32(const char* path, const float* data, size_t n) {
    FILE* fp = std::fopen(path, "wb");
    if (!fp) {
        std::fprintf(stderr, "Error: cannot write to '%s'\n", path);
        return false;
    }
    const size_t wrote = std::fwrite(data, sizeof(float), n, fp);
    std::fclose(fp);
    if (wrote != n) {
        std::fprintf(stderr, "Error: short write to '%s'\n", path);
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    // Mode 2: --allhex — dump the allhex precompute for Test C.
    if (argc >= 2 && std::strcmp(argv[1], "--allhex") == 0) {
        // Expose buildAllhex by reconstructing it here would duplicate logic;
        // instead we demosaic a tiny mosaic and... no — we need the allhex
        // itself. Re-implement the build inline (must match demosaic_xtrans.cpp
        // EXACTLY; both port xtrans_demosaic.py:64-107 verbatim).
        static const int orth[12] = {1, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 1};
        static const int patt[2][16] = {
            {0, 1, 0, -1, 2, 0, -1, 0, 1, 1, 1, -1, 0, 0, 0, 0},
            {0, 1, 0, -2, 1, 0, -2, 0, 1, 1, -2, -2, 1, -1, -1, 1},
        };
        char xt[6][6];
        std::memcpy(xt, kXtransStandard, sizeof(xt));
        auto fc = [&](int r, int c) {
            return static_cast<int>(xt[((r % 6) + 6) % 6][((c % 6) + 6) % 6]);
        };
        int allhex_dr[72] = {0};
        int allhex_dc[72] = {0};
        int sgrow = 0, sgcol = 0;
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                int ng = 0;
                for (int d_idx = 0; d_idx <= 8; d_idx += 2) {
                    int g = (fc(row, col) == 1) ? 1 : 0;
                    if (fc(row + orth[d_idx], col + orth[d_idx + 2]) == 1) ng = 0;
                    else ++ng;
                    if (ng == 4) { sgrow = row; sgcol = col; }
                    if (ng == g + 1) {
                        for (int c = 0; c < 8; ++c) {
                            int v = orth[d_idx]     * patt[g][c * 2]
                                  + orth[d_idx + 1] * patt[g][c * 2 + 1];
                            int h = orth[d_idx + 2] * patt[g][c * 2]
                                  + orth[d_idx + 3] * patt[g][c * 2 + 1];
                            int idx = c ^ (g * 2 & d_idx);
                            int flat = row * 24 + col * 8 + idx;
                            allhex_dr[flat] = v;
                            allhex_dc[flat] = h;
                        }
                    }
                }
            }
        }
        // Print: SGROW, SGCOL, then DR0..DR71, then DC0..DC71 — one per line.
        std::printf("SGROW %d\n", sgrow);
        std::printf("SGCOL %d\n", sgcol);
        for (int i = 0; i < 72; ++i) std::printf("DR %d\n", allhex_dr[i]);
        for (int i = 0; i < 72; ++i) std::printf("DC %d\n", allhex_dc[i]);
        return 0;
    }

    // Mode 3: --smoke <path> [size]
    bool smoke = (argc >= 2 && std::strcmp(argv[1], "--smoke") == 0);
    if (smoke) {
        const char* path = (argc >= 3) ? argv[2] : "/tmp/cpp_xtrans_smoke.bin";
        int size = 1200;
        if (argc >= 4) size = std::atoi(argv[3]);
        auto m = makeXtransMosaic(size, size);
        std::vector<float> truth(static_cast<size_t>(size) * size * 3);
        buildSmooth(m, truth.data());
        subsampleToMosaic(m, truth.data());
        auto img = rawalchemy::xtransMarkesteijnDemosaic(m);
        if (!writeFloat32(path, img.data.data(), img.data.size())) return 1;
        std::printf("%d %d\n", img.height, img.width);
        return 0;
    }

    // Mode 1: <path> [pattern] [size] — synthetic demosaic dump.
    const char* path = (argc >= 2) ? argv[1] : "/tmp/cpp_xtrans.bin";
    std::string pattern = (argc >= 3) ? argv[2] : "smooth";
    int size = (argc >= 4) ? std::atoi(argv[3]) : 300;

    auto m = makeXtransMosaic(size, size);
    std::vector<float> truth(static_cast<size_t>(size) * size * 3);
    if      (pattern == "smooth")     buildSmooth(m, truth.data());
    else if (pattern == "edge-h")     buildEdgeH(m, truth.data());
    else if (pattern == "edge-v")     buildEdgeV(m, truth.data());
    else if (pattern == "edge-d")     buildEdgeD(m, truth.data());
    else {
        std::fprintf(stderr, "Unknown pattern '%s' (use: smooth, edge-h, "
                             "edge-v, edge-d)\n", pattern.c_str());
        return 1;
    }

    subsampleToMosaic(m, truth.data());
    auto img = rawalchemy::xtransMarkesteijnDemosaic(m);

    if (!writeFloat32(path, img.data.data(), img.data.size())) return 1;
    std::printf("%d %d\n", img.height, img.width);
    return 0;
}
