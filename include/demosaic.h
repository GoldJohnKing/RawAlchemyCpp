#pragma once
/**
 * @file demosaic.h
 * @brief Phase 3-4 demosaicing — RCD (Bayer) + Markesteijn 1-pass (X-Trans).
 *
 * Direct C++/OpenMP ports of the Python Taichi references:
 *  - Bayer: `raw_alchemy.demosaic` (demosaic.py:50-540), a Taichi GPU port of
 *    darktable's `demosaic_rcd.cl`. Original algorithm by Luis Sanz Rodríguez.
 *  - X-Trans: `raw_alchemy.xtrans_demosaic` (xtrans_demosaic.py:1-914), a
 *    Taichi GPU port of darktable's `xtrans.c`. Original algorithm by Frank
 *    Markesteijn (1-pass variant).
 *
 * Input:  RawMosaic — black-level subtracted, hot-pixel fixed, highlight-
 *         reconstructed, normalized to [0, 1] (Phase 1+2 output).
 * Output: ImageBuffer (H x W x 3 float RGB), camera-native RGB,
 *         PRE white-balance / PRE color matrix. WB+matrix come in Phase 5.
 *
 * RCD (Bayer): the 8-stage pipeline (populate -> step1..step4_3 ->
 * write_output) runs on the interior; a 4-pixel border is filled by simple
 * bilinear interpolation.
 *
 * Markesteijn (X-Trans): the 9-kernel pipeline (populate -> gminmax ->
 * green_interp -> rb_at_green -> rb_interp -> fill_green22 -> yuv_derivatives
 * -> homo_merge -> border_interpolate) runs on the interior with per-kernel
 * pads {3,3,6,6,8,9,12}; a 12-pixel border is filled by bilinear interpolation.
 */

#include "common.h"
#include "raw_mosaic.h"

#include <cstring>   // memcpy
#include <vector>
#if defined(__aarch64__)
#include <cstdint>   // uint16_t
#endif

namespace rawalchemy {

/**
 * @brief Ratio Corrected Demosaicing of a Bayer CFA mosaic.
 *
 * Port of `raw_alchemy.demosaic.rcd_demosaic` (demosaic.py:459-540).
 *
 * Uses the canonical `cfaColor(const RawMosaic&, int, int)` for all CFA
 * lookups (returns 0=R, 1=G, 2=B, 3=G2). The reference's Taichi FC returns
 * 0/1/2 only (it maps G2->G via the filters bit encoding); since our
 * canonical cfaColor may return 3 for G2 on some non-standard Bayer codes,
 * we treat color==1 || color==3 as green. Standard codes (RGGB/BGGR/GRBG/
 * GBRG) only ever return 0/1/2 so this is a safe superset.
 *
 * Memory: peaks at ~7 x H x W x 4 bytes (intermediates are freed in the
 * same order as the reference's `del` statements — see freeVector() calls).
 *
 * @param m  Input mosaic (read-only). Must be Bayer (m.filters != 9).
 * @return   Demosaiced camera-RGB image (H x W x 3, float, pre-WB/pre-matrix).
 */
ImageBuffer rcdDemosaic(const RawMosaic& m);

/**
 * @brief Markesteijn 1-pass X-Trans demosaicing.
 *
 * Port of `raw_alchemy.xtrans_demosaic.xtrans_markesteijn_demosaic`
 * (xtrans_demosaic.py:806-914).
 *
 * Uses a local FCxt(xt, r, c) helper (negative-index safe) for all CFA
 * lookups, matching the reference's `FCxt(row, col, xtrans)`. The hexagonal
 * neighbor lookup (allhex) is precomputed once on the CPU via buildAllhex().
 *
 * Memory: peaks at ~19 x H x W x 4 bytes (4 dir buffers x 3 ch + 4 drv
 * planes + 1 out). Intermediates are freed to match the reference's `del`
 * ordering (xtrans_demosaic.py:869, 880, 897-898).
 *
 * @param m  Input mosaic (read-only). Must be X-Trans (m.filters == 9).
 * @return   Demosaiced camera-RGB image (H x W x 3, float, pre-WB/pre-matrix).
 */
ImageBuffer xtransMarkesteijnDemosaic(const RawMosaic& m);

// Demosaic intermediate plane. F16 storage on ARM64 (halves memory: ~1.3GB->0.65GB
// Bayer, ~3GB->1.5GB X-Trans), F32 elsewhere. Arithmetic is always F32: reads
// upcast, writes downcast. Mirrors the "F16 storage / F32 compute" pattern of
// log_transform.cpp / lut_applier.cpp.
class DemosaicPlane {
public:
    DemosaicPlane() = default;
    explicit DemosaicPlane(int n) { storage_.resize(static_cast<size_t>(n)); }
    void resize(int n) { storage_.resize(static_cast<size_t>(n)); }
    int size() const { return static_cast<int>(storage_.size()); }
    // READ -- F16->F32 on ARM64, passthrough elsewhere.
    inline float operator[](int idx) const {
#if defined(__aarch64__)
        __fp16 h; std::memcpy(&h, &storage_[idx], 2);
        return static_cast<float>(h);
#else
        return storage_[idx];
#endif
    }
    // WRITE -- F32->F16 on ARM64, passthrough elsewhere.
    inline void set(int idx, float v) {
#if defined(__aarch64__)
        __fp16 h = static_cast<__fp16>(v);
        std::memcpy(&storage_[idx], &h, 2);
#else
        storage_[idx] = v;
#endif
    }
    // Release storage immediately (used by freeVector to mirror Python `del`).
    void clear() { decltype(storage_)().swap(storage_); }
private:
#if defined(__aarch64__)
    std::vector<uint16_t> storage_;   // IEEE-754 binary16 bits
#else
    std::vector<float> storage_;      // F32 (reference path)
#endif
};

} // namespace rawalchemy
