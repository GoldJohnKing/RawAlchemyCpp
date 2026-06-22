// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file demosaic.h
 * @brief Public API for RAW demosaicing algorithm selection.
 *
 * darktable RCD (Bayer) and 3-pass Markesteijn (X-Trans) implementations.
 * Replaces LibRaw's built-in demosaic while preserving LibRaw preprocessing.
 */

namespace rawalchemy {

/// Demosaic algorithm selection for decodeRaw().
enum class DemosaicAlgorithm {
    /// Original behavior: delegate to LibRaw's user_qual path (DHT for Bayer,
    /// xtrans_interpolate for X-Trans). Use for fallback/debugging.
    LIBRAW_FALLBACK,

    /// Bayer CFA only: darktable RCD (Ratio Corrected Demosaicing).
    /// Errors if called on X-Trans input.
    RCD,

    /// X-Trans CFA only: darktable 3-pass Markesteijn.
    /// Errors if called on Bayer input.
    MARKESTEIJN_3PASS,

    /// Default: auto-select based on CFA type.
    /// Bayer (filters != 9) → RCD, X-Trans (filters == 9) → MARKESTEIJN_3PASS.
    AUTO,
};

} // namespace rawalchemy
