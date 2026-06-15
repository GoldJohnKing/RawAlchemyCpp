#pragma once
/**
 * @file raw_mosaic.h
 * @brief Phase 1 RAW mosaic container — packed CFA mosaic + sensor metadata.
 *
 * The mosaic is the un-demosaiced single-channel sensor data (one float per
 * photosite), plus all the metadata needed for downstream black-subtraction /
 * hot-pixel / demosaic / highlight / matrix stages.
 *
 * Source: LibRaw `imgdata.rawdata.raw_image` (1-channel ushort CFA buffer).
 * This is distinct from the existing decodeRaw() path which runs dcraw_process
 * into ProPhoto RGB.
 */

#include <vector>

namespace rawalchemy {

/**
 * @brief Decoded RAW mosaic + metadata for the CPU pipeline.
 *
 * Layout of `data`: row-major, 1 float per photosite, visible region only
 * (rows [top_margin, top_margin+height), cols [left_margin, left_margin+width)).
 * Values are raw uint16 sensor readings cast to float — NOT normalized until
 * subtractBlackLevel() runs.
 */
struct RawMosaic {
    std::vector<float> data;        ///< 1-channel packed CFA mosaic (raw uint16 -> float, NOT yet normalized)
    int width = 0;                  ///< Visible mosaic width  (after left_margin crop)
    int height = 0;                 ///< Visible mosaic height (after top_margin crop)
    unsigned filters = 0;           ///< LibRaw CFA code (==9 => X-Trans)
    char xtrans[6][6] = {{0}};      ///< X-Trans pattern (only meaningful if filters==9)
    int colors = 3;                 ///< Sensor color count (3 for typical RGBG Bayer)
    bool is_foveon = false;         ///< True for Foveon / non-CFA sensors (rejected in decode)

    // Per-channel black (collapsed) + white point + WB + matrix.
    // cblack holds the EFFECTIVE per-channel black = (imgdata.color.black +
    // imgdata.color.cblack[c]) for c in 0..3 — i.e. both the scalar base
    // and the per-channel offset collapsed together. This matches rawpy's
    // `black_level_per_channel`.
    float cblack[4] = {0, 0, 0, 0}; ///< Collapsed per-channel black levels
    float maximum = 0.0f;           ///< White point (imgdata.color.maximum)
    float cam_mul[4] = {1, 1, 1, 1};///< Camera WB coefficients
    double cam_xyz[4][3] = {{0}};   ///< XYZ -> camera matrix
    int flip = 0;                   ///< Orientation (applied LATER, post-demosaic; just stored)
    bool has_2d_darkframe = false;  ///< True if cblack[4]/[5] nonzero in source (2D frame NOT applied in Phase 1)
};

/**
 * @brief CFA color index at photosite (row, col) — CANONICAL helper for all phases.
 *
 * Bayer: dcraw/LibRaw FC macro. NOTE: the shift amount MUST be computed with a
 * separate variable / explicit parens. The naive one-liner
 * `(f >> (((r<<1)&14) | (c&1)) << 1) & 3` mis-parses in C++ because `<<` binds
 * tighter than `|`, yielding `((f >> (...)) << 1) & 3` — wrong. This was a latent
 * Phase-1 bug (color-blind stages hid it); Phase 2+ color-sensitive stages exposed it.
 * Returns 0=R, 1=G, 2=B, 3=G2.
 *
 * X-Trans (filters==9): indexes the 6x6 pattern directly (negative-index safe).
 */
inline int cfaColor(const RawMosaic& m, int r, int c) {
    if (m.filters == 9) {
        return static_cast<int>(m.xtrans[((r % 6) + 6) % 6][((c % 6) + 6) % 6]);
    }
    const unsigned f = m.filters;
    const int shift = ((((r << 1) & 14) | (c & 1)) << 1);
    return static_cast<int>((f >> shift) & 3);
}

} // namespace rawalchemy
