// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file denoise_xtrans.h
 * @brief X-Trans CFA pre-demosaic wavelet denoise (darktable port).
 *
 * Ported from darktable src/iop/rawdenoise.c `wavelet_denoise_xtrans`
 * (Copyright (C) 2011-2026 darktable developers; GPL-3.0-or-later, one-way
 * compatible with AGPL-3.0-or-later). Splits the 6x6 X-Trans mosaic into R/G/B
 * sensel sets, splats each onto a dense grid with a sqrt variance-stabilizing
 * transform, denoises each channel with dwt_denoise, and writes the squared
 * result back to the matching CFA sites.
 */

namespace rawalchemy {

/**
 * @brief Denoise an X-Trans CFA mosaic.
 *
 * @param in        Single-channel X-Trans mosaic, float[w*h]. Read only.
 * @param out       Denoised mosaic, float[w*h]. Must not alias `in` (the
 *                  per-channel write-back would otherwise corrupt channels not
 *                  yet processed). May equal `in` only if w*h == 0.
 * @param w,h       Image dimensions.
 * @param xtrans    6x6 color map from LibRaw `imgdata.idata.xtrans` (0=R,1=G,2=B).
 * @param threshold darktable-domain noise threshold in [0, ~0.1]; 0 is an
 *                  identity pass. Auto mode uses a flat 0.01 (see below).
 */
void denoise_xtrans(const float* in, float* out, int w, int h,
                    const char xtrans[6][6], float threshold);

/**
 * @brief X-Trans denoise threshold: flat 0.01 in auto mode (ISO-independent).
 *
 * denoise_xtrans applies a sqrt variance-stabilizing transform (Anscombe),
 * which makes the shot-noise threshold ISO-independent — so, like darktable's
 * rawdenoise (which has no ISO logic and defaults to 0.01), auto mode returns a
 * single fixed 0.01 across all ISO. An ISO ramp would re-introduce the coupling
 * the VST is there to eliminate.
 *
 * @param iso               EXIF ISO speed (ignored in auto mode; kept for ABI
 *                          stability and a potential future ISO-aware path).
 * @param manualThreshold   <0 = auto (flat 0.01); 0 = off; >0 = literal value.
 * @return                  Threshold in darktable domain; 0 means "do not denoise".
 */
float computeXtransDenoiseThreshold(float iso, float manualThreshold);

} // namespace rawalchemy
