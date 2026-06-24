// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file dwt_denoise.h
 * @brief Single-channel à trous ("with holes") stationary wavelet soft-threshold denoise.
 *
 * Ported from darktable src/common/dwt.c `dwt_denoise` and its `_1ch` helpers
 * (Copyright (C) 2017-2026 darktable developers; original GPL-3.0-or-later,
 * one-way compatible with AGPL-3.0-or-later). The algorithm decomposes an image
 * into `bands` dyadic wavelet scales using a [1 2 1; 2 4 2; 1 2 1]/16 separable
 * hat transform, applies a soft threshold per scale, and recomposes.
 */

namespace rawalchemy {

/// Number of wavelet scales used by the RAW denoiser. Matches darktable's
/// DT_IOP_RAWDENOISE_BANDS.
inline constexpr int DWT_DENOISE_BANDS = 5;

/**
 * @brief Denoise a single-channel float image in place via à trous wavelets.
 *
 * @param img    Image buffer, float[width*height], modified in place.
 * @param width  Image width in pixels.
 * @param height Image height in pixels.
 * @param bands  Number of wavelet scales (pass DWT_DENOISE_BANDS).
 * @param noise  Per-scale soft-threshold magnitudes, length `bands`. A scale's
 *               detail coefficient is shrunk toward zero by `noise[scale]`.
 *               All-zero `noise` is an identity pass.
 */
void dwt_denoise(float* img, int width, int height, int bands, const float* noise);

} // namespace rawalchemy
