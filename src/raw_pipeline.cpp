/**
 * @file raw_pipeline.cpp
 * @brief Shared custom CPU demosaic pipeline (Phases 1-5) — used by BOTH the
 *        CLI (src/main.cpp) and the C API (src/raw_alchemy_capi.cpp).
 *
 * Single source of truth for the custom RAW decode. The pipeline reuses
 * LibRaw's OWN compiled denoise stages (green_matching + wavelet + FBDD +
 * post-demosaic chroma median) via the DenoiseLibRaw subclass, restoring the
 * color-noise control the old dcraw_process path had but the hand-rolled
 * custom path had dropped.
 *
 * DECOUPLED pipeline (ora-1): WB and denoise are kept separate so the green-
 * anchored applyWhiteBalance (not LibRaw's max-anchored scale_colors) controls
 * white balance. scale_colors is NOT called. Stage order:
 *   open -> unpack -> raw2image_ex(1) [inline black subtract] ->
 *   green_matching -> [set green-anchored pre_mul] ->
 *   [ISO-adaptive threshold -> wavelet_denoise] -> [fbdd (Bayer only)] ->
 *   extract denoised CFA mosaic (normalize by post-wavelet maximum) ->
 *   fixHotPixels -> highlightInpaintOpposed ->
 *   (rcdDemosaic | xtransMarkesteijnDemosaic) -> applyWhiteBalance ->
 *   median_filter (post-demosaic chroma median) ->
 *   applyFlip -> cameraToProphotoMatrix -> applyColorMatrix -> clamp [0,1].
 *
 * If any LibRaw denoise stage throws, the function falls back to the proven
 * non-denoised custom pipeline (decodeRawMosaic -> ... -> applyWhiteBalance)
 * so callers always get valid output.
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
// runs the original 7-stage custom path.
ImageBuffer decodeCustomPipelineNoDenoise(const std::string& inputPath) {
    auto mosaic = decodeRawMosaic(inputPath, nullptr);
    subtractBlackLevel(mosaic);
    fixHotPixels(mosaic);
    highlightInpaintOpposed(mosaic);
    ImageBuffer img = (mosaic.filters == 9)
        ? xtransMarkesteijnDemosaic(mosaic)
        : rcdDemosaic(mosaic);
    applyWhiteBalance(img, mosaic.cam_mul);
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
// Decoupled LibRaw denoise + custom WB. Owns a single DenoiseLibRaw across
// the whole flow so the LibRaw denoise stages (which mutate imgdata.image in
// place) run on the same buffer the mosaic is later extracted from, and the
// post-demosaic median_filter can round-trip through the same imgdata.image.
//
// WB is NOT delegated to LibRaw's scale_colors (which is max-anchored and
// crushes green -> magenta cast). Instead the green-anchored applyWhiteBalance
// runs after demosaic, exactly like the non-denoised custom path. The LibRaw
// denoise stages that have NO WB dependency (green_matching, wavelet_denoise,
// fbdd, median_filter) are called directly.
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

        // Step 6: FBDD chroma denoise (Bayer only: filters > 1000). No WB
        // dependency; operates on imgdata.image in-place. fbdd partially
        // demosaics (fills non-CFA channels) — extraction reads ONLY the
        // CFA-active channel, so the partial demosaic is harmless.
        int fbddLevel = 0;
        if (iso <= 100.0f) fbddLevel = 0;
        else if (iso < 3200.0f) fbddLevel = 1;
        else fbddLevel = 2;
        const unsigned filters = rawProcessor.imgdata.idata.filters;
        if (fbddLevel > 0 && filters > 1000) {
            rawProcessor.fbdd(fbddLevel);
        }

        // Step 7: extract denoised CFA mosaic from imgdata.image.
        // After raw2image_ex(1) + green_matching + wavelet + fbdd, imgdata.image
        // holds denoised, black-subtracted CFA data. Normalize by the CURRENT
        // maximum (post-wavelet bit-shift, post-black-subtraction).
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
        // REAL cam_mul for the green-anchored applyWhiteBalance downstream
        // (scale_colors was NOT called, so WB is not yet applied).
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

        // Debug: mosaic data range (REMOVE before final commit)
        // Step 8: custom pipeline (skip subtractBlackLevel — raw2image_ex(1) did it).
        fixHotPixels(mosaic);
        highlightInpaintOpposed(mosaic);

        ImageBuffer img = (mosaic.filters == 9)
            ? xtransMarkesteijnDemosaic(mosaic)
            : rcdDemosaic(mosaic);

        // Green-anchored WB (correct — scale_colors was NOT called).
        applyWhiteBalance(img, mosaic.cam_mul);

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
