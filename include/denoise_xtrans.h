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
 *                  identity pass. Typical auto range is 0.01..0.05.
 */
void denoise_xtrans(const float* in, float* out, int w, int h,
                    const char xtrans[6][6], float threshold);

/**
 * @brief ISO-adaptive X-Trans denoise threshold, mirroring the Bayer wavelet
 *        ISO curve in raw_decoder.cpp:226-239 but rescaled to darktable's domain.
 *
 * Mirrors Bayer breakpoints: ISO<=100 off, 101..400 light (0.01), >400 a
 * log2-linear ramp to 0.05 at ISO 12800, clamped above.
 *
 * @param iso               EXIF ISO speed (e.g. from imgdata.other.iso_speed).
 * @param manualThreshold   <0 = auto (ISO-adaptive); 0 = off; >0 = literal value.
 * @return                  Threshold in darktable domain; 0 means "do not denoise".
 */
float computeXtransDenoiseThreshold(float iso, float manualThreshold);

} // namespace rawalchemy
