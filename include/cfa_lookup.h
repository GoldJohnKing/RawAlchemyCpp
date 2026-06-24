// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file cfa_lookup.h
 * @brief Inline helpers for resolving the color channel at a given CFA pixel.
 *
 * Used by extractCfaFromImage() and the demosaic algorithms (RCD, Markesteijn).
 */

#include <cstddef>
#include <cstdint>

namespace rawalchemy {

/// Color channel indices (matches LibRaw/darktable convention)
/// 0 = Red, 1 = Green (on red row), 2 = Blue, 3 = Green (on blue row)
/// For demosaic purposes, channels 1 and 3 are both "green".

namespace detail {
/// darktable/LibRaw FC macro: extracts a 2-bit color code from the standard
/// CFA filters bitmask. Returns 0=Red, 1=Green, 2=Blue (greens not distinguished).
/// `filters` packs 16 2-bit slots indexing an 8-row x 2-column repeating tile.
inline unsigned fcColor(int row, int col, unsigned filters) {
    return (filters >> ((((row) << 1 & 14) + ((col) & 1)) << 1)) & 3;
}
} // namespace detail

/// Get the color channel index (0=R, 1=G, 2=B, 3=other G) for a Bayer CFA pixel.
/// `filters` is LibRaw's standard CFA bitmask (same encoding as darktable's FC()).
/// Greens are distinguished as 1 (red-row green) and 3 (blue-row green), matching
/// the four-channel layout expected by RCD and dual-green demosaic kernels.
inline unsigned bayerColor(int row, int col, unsigned filters) {
    unsigned color = detail::fcColor(row, col, filters);
    if (color != 1) {
        return color;  // Red (0) or Blue (2) — no ambiguity
    }
    // Green pixel: determine whether it sits on a red row (channel 1)
    // or a blue row (channel 3) by inspecting its horizontal CFA neighbor.
    unsigned neighbor = detail::fcColor(row, col ^ 1, filters);
    return (neighbor == 0) ? 1u : 3u;
}

/// Get the color channel index (0=R, 1=G, 2=B) for an X-Trans CFA pixel.
/// `xtrans` is the 6×6 pattern array (matches LibRaw's char xtrans[6][6] type).
inline unsigned xtransColor(int row, int col, const char xtrans[6][6]) {
    return xtrans[((row % 6) + 6) % 6][((col % 6) + 6) % 6];
}

/// True if the LibRaw filters value indicates an X-Trans sensor.
inline bool isXtrans(unsigned filters) {
    return filters == 9;
}

} // namespace rawalchemy
