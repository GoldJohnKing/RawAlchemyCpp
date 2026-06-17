/**
 * @file raw_pipeline.cpp
 * @brief Shared custom CPU demosaic pipeline (Phases 1-5) — used by BOTH the
 *        CLI (src/main.cpp) and the C API (src/raw_alchemy_capi.cpp).
 *
 * Single source of truth for the custom RAW decode. v2 pipeline (darktable-
 * aligned order). Reuses LibRaw's compiled green_matching + wavelet_denoise
 * via the DenoiseLibRaw subclass; fixHotPixels is a darktable hotpixels.c port
 * (4-neighbour MAX replacement); demosaic is a RawTherapee rcd_demosaic.cc port.
 *
 * Stage order (cf. darktable iop priority numbers):
 *   open -> unpack -> raw2image_ex(1) [inline black subtract] ->
 *   green_matching -> [set green-anchored pre_mul] ->
 *   [ISO-adaptive threshold -> wavelet_denoise] (rawdenoise(7) equivalent) ->
 *   extract denoised CFA mosaic (normalize by post-wavelet maximum) ->
 *   applyWhiteBalanceMosaic [temperature(3.0), clean per-channel gain on float] ->
 *   zero cam_mul [so highlight's virtual-WB path sees identity gains] ->
 *   highlightInpaintOpposed [highlights(4.0)] ->
 *   fixHotPixels [hotpixels(6.0), darktable port] ->
 *   (rcdDemosaic | xtransMarkesteijnDemosaic) [demosaic(8.0)] ->
 *   median_filter (post-demosaic chroma median, denoiseprofile(9) equivalent) ->
 *   applyFlip -> cameraToProphotoMatrix -> applyColorMatrix -> clamp [0,1].
 *
 * Tradeoff (deepwork plan Path A2): wavelet_denoise runs PRE-WB on ushort
 * imgdata.image to avoid rewriting it in float (keeps LibRaw code, respects
 * "minimize custom" constraint). This is the only deviation from darktable's
 * rawdenoise(7.0) post-WB positioning; the stronger darktable hotpixels port
 * compensates the lost hot-pixel coverage.
 *
 * If any LibRaw denoise stage throws, the function falls back to the proven
 * non-denoised custom pipeline (decodeCustomPipelineNoDenoise, same v2 order
 * minus the LibRaw denoise stages) so callers always get valid output.
 */

#include "raw_pipeline.h"

#include "raw_mosaic.h"
#include "raw_preprocess.h"
#include "highlight.h"
#include "demosaic.h"
#include "raw_postprocess.h"
#include "colorspace_matrices.h"
#include "raw_decoder.h"
#include "denoise_libraw.h"
#include "exif_injector.h"

#include <libraw/libraw.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <stdexcept>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
namespace {
    std::wstring utf8_to_wide(const std::string& utf8) {
        if (utf8.empty()) return std::wstring();
        int size = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
        if (size <= 0) return std::wstring();
        std::wstring wide(static_cast<size_t>(size - 1), 0);
        MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, &wide[0], size);
        return wide;
    }
}
#endif

namespace rawalchemy {

namespace {

// Wide-path open helper (mirrors raw_decoder.cpp / raw_alchemy_capi.cpp).
// Required so non-ASCII paths work via the FFI on Windows.
int openRawFile(LibRaw& p, const std::string& path) {
#ifdef _WIN32
    auto widePath = utf8_to_wide(path);
    return p.open_file(widePath.c_str());
#else
    return p.open_file(path.c_str());
#endif
}

void throwLibRawError(int ret, const char* context) {
    throw std::runtime_error(
        std::string("[CustomPipeline] ") + context + std::string(" failed: ") +
        std::string(libraw_strerror(ret))
    );
}

// ---- LibRaw denoise fallback (proven non-denoised custom pipeline) ----
// Used when any LibRaw denoise stage throws. Re-opens WITHOUT an exif callback
// (the primary open already collected EXIF) so tags are not duplicated, then
// runs the v2 custom path (WB-front, darktable-aligned order). Kept in sync
// with decodeImageWithCustomPipeline's post-extraction stages.
ImageBuffer decodeCustomPipelineNoDenoise(const std::string& inputPath) {
    auto mosaic = decodeRawMosaic(inputPath, nullptr);
    subtractBlackLevel(mosaic);
    // v2: WB applied on float mosaic (pre-demosaic), matching darktable
    // temperature(3.0). Zero cam_mul afterward so highlightInpaintOpposed's
    // virtual-WB path sees identity gains (data already WB'd).
    applyWhiteBalanceMosaic(mosaic);
    for (int c = 0; c < 4; ++c) mosaic.cam_mul[c] = mosaic.cam_mul[1];
    highlightInpaintOpposed(mosaic);
    fixHotPixels(mosaic);
    ImageBuffer img = (mosaic.filters == 9)
        ? xtransMarkesteijnDemosaic(mosaic)
        : rcdDemosaic(mosaic);
    applyFlip(img, mosaic.flip);
    auto M = cameraToProphotoMatrix(mosaic);
    applyColorMatrix(img, M);
    img.clamp();
    return img;
}

} // anonymous namespace

// ============================================================
//             decodeImageWithCustomPipeline
// ============================================================
//
// v2 pipeline (darktable-aligned order). Owns a single DenoiseLibRaw across
// the whole flow so the LibRaw denoise stages (which mutate imgdata.image in
// place) run on the same buffer the mosaic is later extracted from, and the
// post-demosaic median_filter can round-trip through the same imgdata.image.
//
// Order (cf. darktable iop priority): rawprepare(1) -> [green_matching] ->
// wavelet/rawdenoise(7) -> extract mosaic -> temperature/WB(3) ->
// highlights(4) -> hotpixels(6) -> demosaic(8) -> median/denoiseprofile(9).
//
// WB is NOT delegated to LibRaw's scale_colors (max-anchored -> magenta cast).
// Instead applyWhiteBalanceMosaic runs on the float mosaic PRE-demosaic
// (clean per-channel gain, no ushort saturation). cam_mul is zeroed right
// after so highlightInpaintOpposed's virtual-WB path sees identity gains.
//
// Note: wavelet_denoise runs PRE-WB on ushort imgdata.image (Path A2 tradeoff
// in deepwork plan — keeps LibRaw wavelet without rewrite; the only deviation
// from darktable rawdenoise(7) post-WB positioning). hotpixels is the darktable
// port (4-neighbour MAX replacement); demosaic is the RT RCD port.
ImageBuffer decodeImageWithCustomPipeline(
    const std::string& inputPath,
    ExifCollector* exifCollector) {
    try {
        DenoiseLibRaw rawProcessor;

        // EXIF callback before open (same pattern as decodeRaw / decodeRawMosaic).
        // Single collection — JPEG output keeps EXIF. The fallback path re-opens
        // with nullptr so tags are not duplicated.
        if (exifCollector) {
            rawProcessor.set_exifparser_handler(getExifCallback(), exifCollector);
        }

        // --- Open + unpack ---
        int ret = openRawFile(rawProcessor, inputPath);
        if (ret != LIBRAW_SUCCESS) throwLibRawError(ret, "open_file");
        ret = rawProcessor.unpack();
        if (ret != LIBRAW_SUCCESS) throwLibRawError(ret, "unpack");

        // Non-CFA sensors have no raw_image — bail to the caller's routing.
        if (rawProcessor.imgdata.rawdata.raw_image == nullptr) {
            throw std::runtime_error(
                "[CustomPipeline] non-CFA sensor; use dcraw_process path");
        }

        // Step 1: raw2image_ex(1) — populate imgdata.image WITH inline black
        // subtraction. do_subtract_black=1 correctly:
        //   - copies rawdata.raw_image -> imgdata.image (4ch per pixel)
        //   - subtracts black per-channel
        //   - zeroes imgdata.color.black and cblack[]
        //   - adjusts imgdata.color.maximum (maximum -= black)
        // CRITICAL: threshold MUST be 0 here, else IO.shrink=1 -> half res.
        rawProcessor.imgdata.params.threshold = 0;
        rawProcessor.imgdata.params.half_size = 0;
        ret = rawProcessor.raw2image_ex(1);
        if (ret != LIBRAW_SUCCESS) throwLibRawError(ret, "raw2image_ex");

        // Step 2: green_matching (G1/G3 equalization — no WB dependency).
        rawProcessor.green_matching();

        // Step 3: set green-anchored pre_mul. wavelet_denoise uses only the
        // RATIO pre_mul[1]/pre_mul[3] for its G1/G3 pull-together (= 1.0 for
        // standard Bayer where cam_mul[1]==cam_mul[3]); green-anchored is the
        // conceptually correct value and harmless otherwise.
        {
            float* cm = rawProcessor.imgdata.color.cam_mul;
            float g = (cm[1] > 1e-6f) ? cm[1] : 1.0f;
            for (int c = 0; c < 4; c++)
                rawProcessor.imgdata.color.pre_mul[c] = cm[c] / g;
        }

        // Step 4: ISO-adaptive threshold (MUST be after raw2image_ex to avoid
        // the IO.shrink half-resolution bug). Mirrors decodeRaw()'s mapping.
        const float iso = rawProcessor.imgdata.other.iso_speed;
        auto& params = rawProcessor.imgdata.params;
        if (iso <= 100.0f) {
            params.threshold = 0;
        } else if (iso <= 400.0f) {
            params.threshold = 100;
        } else {
            const float t = (std::log2(iso) - 8.644f) / (13.644f - 8.644f);
            params.threshold = 100.0f + std::min(std::max(t, 0.0f), 1.0f) * 900.0f;
        }

        // Step 5: wavelet_denoise (directly — NOT inside scale_colors).
        // Operates on imgdata.image in-place. Side effect: image values AND
        // maximum are left-shifted by `scale` bits (both shift equally, so
        // image/maximum still yields the correct [0,1] normalization).
        if (params.threshold > 0.0f) {
            rawProcessor.wavelet_denoise();
        }

        // Step 6: extract denoised CFA mosaic from imgdata.image.
        // After raw2image_ex(1) + green_matching + wavelet, imgdata.image
        // holds denoised, black-subtracted CFA data. Normalize by the CURRENT
        // maximum (post-wavelet bit-shift, post-black-subtraction).
        // (FBDD is intentionally not used — darktable/RT do not use it either;
        // their pre-demosaic denoise is the wavelet above + hotpixels below.)
        const auto& sizes = rawProcessor.imgdata.sizes;
        const auto& idata = rawProcessor.imgdata.idata;
        const auto& color = rawProcessor.imgdata.color;
        const int W = static_cast<int>(sizes.iwidth);
        const int H = static_cast<int>(sizes.iheight);
        if (W <= 0 || H <= 0) {
            throw std::runtime_error("[CustomPipeline] bad dimensions");
        }

        // Capture maximum AFTER wavelet (it may have been left-shifted).
        const float normMax = static_cast<float>(color.maximum);
        const float normDiv = (normMax > 0.0f) ? normMax : 65535.0f;

        RawMosaic mosaic;
        mosaic.width   = W;
        mosaic.height  = H;
        mosaic.filters = idata.filters;
        mosaic.colors  = idata.colors;
        std::memcpy(mosaic.xtrans, idata.xtrans, sizeof(mosaic.xtrans));
        // LibRaw's color.cam_xyz is float[4][3]; RawMosaic.cam_xyz is
        // double[4][3] — cast element-wise (NOT memcpy), matching decodeRawMosaic.
        for (int c = 0; c < 4; ++c) {
            for (int k = 0; k < 3; ++k) {
                mosaic.cam_xyz[c][k] = static_cast<double>(color.cam_xyz[c][k]);
            }
        }
        mosaic.flip = sizes.flip;
        // Black already subtracted; data normalized during extraction.
        for (int c = 0; c < 4; ++c) mosaic.cblack[c] = 0.0f;
        mosaic.maximum = 1.0f;
        // REAL cam_mul stored for applyWhiteBalanceMosaic downstream (v2: WB
        // applied on the float mosaic pre-demosaic; scale_colors NOT called).
        for (int c = 0; c < 4; ++c) mosaic.cam_mul[c] = color.cam_mul[c];

        mosaic.data.resize(static_cast<size_t>(W) * H);
        ushort(*img4)[4] = rawProcessor.imgdata.image;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const int ch = std::min(cfaColor(mosaic, y, x), 3);
                const ushort val = img4[static_cast<size_t>(y) * W + x][ch];
                mosaic.data[static_cast<size_t>(y) * W + x] =
                    static_cast<float>(val) / normDiv;
            }
        }

        // Step 8: custom pipeline (skip subtractBlackLevel — raw2image_ex(1) did it).
        // v2: WB applied on float mosaic (pre-demosaic), matching darktable
        // temperature(3.0). Zero cam_mul afterward so highlightInpaintOpposed's
        // virtual-WB path sees identity gains (data is already WB'd).
        applyWhiteBalanceMosaic(mosaic);
        for (int c = 0; c < 4; ++c) mosaic.cam_mul[c] = mosaic.cam_mul[1];

        // darktable order: highlights(4.0) before hotpixels(6.0) before demosaic(8.0).
        highlightInpaintOpposed(mosaic);
        fixHotPixels(mosaic);

        ImageBuffer img = (mosaic.filters == 9)
            ? xtransMarkesteijnDemosaic(mosaic)
            : rcdDemosaic(mosaic);
        // WB already applied on mosaic above — no post-demosaic WB.

        // Step 9: post-demosaic chroma median via LibRaw's median_filter.
        // Round-trip img -> imgdata.image -> median_filter -> read back, in
        // sensor-native space (before the camera->ProPhoto matrix). median_filter
        // uses sizes.width/height as stride; with shrink=0 these equal
        // iwidth/iheight == img dims. Wrapped so a failure degrades gracefully.
        try {
            params.med_passes = 2;
            const size_t n = static_cast<size_t>(W) * H;
            for (size_t i = 0; i < n; ++i) {
                img4[i][0] = static_cast<ushort>(
                    std::clamp(img.data[i * 3 + 0], 0.0f, 1.0f) * 65535.0f + 0.5f);
                img4[i][1] = static_cast<ushort>(
                    std::clamp(img.data[i * 3 + 1], 0.0f, 1.0f) * 65535.0f + 0.5f);
                img4[i][2] = static_cast<ushort>(
                    std::clamp(img.data[i * 3 + 2], 0.0f, 1.0f) * 65535.0f + 0.5f);
                img4[i][3] = 0;  // scratch channel used by median_filter
            }
            rawProcessor.median_filter();
            for (size_t i = 0; i < n; ++i) {
                img.data[i * 3 + 0] = static_cast<float>(img4[i][0]) / 65535.0f;
                img.data[i * 3 + 1] = static_cast<float>(img4[i][1]) / 65535.0f;
                img.data[i * 3 + 2] = static_cast<float>(img4[i][2]) / 65535.0f;
            }
        } catch (const std::exception& e) {
            fprintf(stderr,
                "[CustomPipeline] median_filter failed (%s); skipping\n", e.what());
        }

        // Step 10: orientation + camera->ProPhoto matrix + clip.
        applyFlip(img, mosaic.flip);
        auto M = cameraToProphotoMatrix(mosaic);
        applyColorMatrix(img, M);
        img.clamp();
        return img;
    } catch (const std::exception& e) {
        fprintf(stderr,
            "[CustomPipeline] denoise path failed (%s); fallback\n", e.what());
    } catch (...) {
        fprintf(stderr,
            "[CustomPipeline] denoise path failed; fallback\n");
    }

    // Fallback: proven non-denoised custom pipeline. nullptr exifCollector
    // avoids duplicating EXIF tags (the primary open already collected them).
    return decodeCustomPipelineNoDenoise(inputPath);
}

} // namespace rawalchemy
