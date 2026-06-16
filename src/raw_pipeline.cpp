/**
 * @file raw_pipeline.cpp
 * @brief Shared custom CPU demosaic pipeline (Phases 1-5) — used by BOTH the
 *        CLI (src/main.cpp) and the C API (src/raw_alchemy_capi.cpp).
 *
 * Single source of truth for the custom RAW decode. The pipeline now REUSES
 * LibRaw's OWN compiled denoise stages (green_matching + wavelet + FBDD +
 * post-demosaic chroma median) via the DenoiseLibRaw subclass, restoring the
 * color-noise control the old dcraw_process path had but the hand-rolled
 * custom path had dropped.
 *
 * Stage order:
 *   open -> unpack -> raw2image_ex -> subtract_black_internal ->
 *   adjust_maximum -> [green_matching -> scale_colors(WB+wavelet) -> fbdd] ->
 *   extract denoised CFA mosaic -> fixHotPixels -> highlightInpaintOpposed ->
 *   (rcdDemosaic | xtransMarkesteijnDemosaic) ->
 *   median_filter (post-demosaic chroma median) ->
 *   applyFlip -> cameraToProphotoMatrix -> applyColorMatrix -> clamp [0,1].
 *
 * LibRaw's subtract_black_internal + scale_colors replace the custom
 * subtractBlackLevel + applyWhiteBalance (skipped here). If any LibRaw denoise
 * stage throws, the function falls back to the proven non-denoised custom
 * pipeline (decodeRawMosaic -> ... -> applyWhiteBalance) so callers always get
 * valid output.
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
// Owns a single DenoiseLibRaw instance across the WHOLE flow so the LibRaw
// denoise stages (which mutate imgdata.image in place) run on the same buffer
// the mosaic is later extracted from, and the post-demosaic median_filter can
// round-trip through the same imgdata.image.
ImageBuffer decodeImageWithCustomPipeline(
    const std::string& inputPath,
    ExifCollector* exifCollector) {
    try {
        DenoiseLibRaw rawProcessor;

        // EXIF callback must be wired before open_file (same pattern as
        // decodeRaw / decodeRawMosaic). Single collection — JPEG output keeps
        // EXIF. The fallback path re-opens with nullptr so tags are not duped.
        if (exifCollector) {
            rawProcessor.set_exifparser_handler(getExifCallback(), exifCollector);
        }

        // --- Open + unpack ---
        int ret = openRawFile(rawProcessor, inputPath);
        if (ret != LIBRAW_SUCCESS) throwLibRawError(ret, "open_file");
        ret = rawProcessor.unpack();
        if (ret != LIBRAW_SUCCESS) throwLibRawError(ret, "unpack");

        // Non-CFA sensors have no raw_image — bail to the caller's routing
        // (decodeRaw/dcraw). Mirrors decodeRawMosaic's guard.
        if (rawProcessor.imgdata.rawdata.raw_image == nullptr) {
            throw std::runtime_error(
                "[CustomPipeline] non-CFA sensor; use dcraw_process path");
        }

        // IMPORTANT: keep params.threshold == 0 during raw2image_ex. LibRaw
        // sets IO.shrink=1 (halving iheight/iwidth) when threshold != 0
        // (raw2image.cpp:49-56). Since this pipeline does NOT call
        // pre_interpolate() (which would upsample back), shrink would leave a
        // half-size mosaic. We set the ISO-adaptive threshold AFTER
        // raw2image_ex so the buffer stays full-size; wavelet_denoise (called
        // inside scale_colors) then runs on the full-size image.
        rawProcessor.imgdata.params.threshold = 0;
        rawProcessor.imgdata.params.half_size = 0;

        ret = rawProcessor.raw2image_ex(0);  // allocate imgdata.image, copy raw
        if (ret != LIBRAW_SUCCESS) throwLibRawError(ret, "raw2image_ex");

        // Black subtraction + white-point refinement (matches dcraw_process
        // ordering: subtract_black_internal -> adjust_maximum).
        ret = rawProcessor.subtract_black_internal();
        if (ret != LIBRAW_SUCCESS) throwLibRawError(ret, "subtract_black_internal");
        rawProcessor.adjust_maximum();

        // --- ISO-adaptive denoise thresholds ---
        // Mirrors decodeRaw()'s mapping (raw_decoder.cpp:167-202) so the custom
        // path gets the same denoise strength as the old dcraw path.
        const float iso = rawProcessor.imgdata.other.iso_speed;
        auto& params = rawProcessor.imgdata.params;

        if (iso <= 100.0f) {
            params.threshold = 0;          // base ISO — sensor noise floor negligible
        } else if (iso <= 400.0f) {
            params.threshold = 100;        // low ISO — minimal fixed wavelet denoise
        } else {
            // ISO 400-12800+: log2-scale mapping to 100-1000.
            const float logLow  = 8.644f;   // log2(400)
            const float logHigh = 13.644f;  // log2(12800)
            const float t = (std::log2(iso) - logLow) / (logHigh - logLow);
            params.threshold = 100.0f + std::min(std::max(t, 0.0f), 1.0f) * 900.0f;
        }

        int fbddMode;
        if (iso <= 100.0f) {
            fbddMode = 0;
        } else if (iso < 3200.0f) {
            fbddMode = 1;
        } else {
            fbddMode = 2;
        }
        params.fbdd_noiserd = fbddMode;

        // scale_colors() uses O.use_camera_wb + O.highlight to derive pre_mul.
        // use_camera_wb=1 makes it consume the camera WB (cam_mul), and a
        // non-zero highlight keeps dmax = max(pre_mul) so the normalization
        // mirrors applyWhiteBalance's green/max-anchored scaling.
        params.use_camera_wb = 1;
        params.highlight = 2;              // match decodeRaw convention
        params.med_passes = 2;             // post-demosaic chroma median
        params.green_matching = 1;

        // --- Pre-demosaic LibRaw denoise (LibRaw's OWN compiled code) ---
        // green_matching: G1/G3 equalization (CFA domain).
        // scale_colors:   WB multiply + wavelet_denoise (threshold > 0).
        // fbdd:           CFA-domain chroma denoise.
        rawProcessor.green_matching();
        rawProcessor.scale_colors();
        rawProcessor.fbdd(fbddMode);

        // --- Extract denoised CFA mosaic from imgdata.image ---
        // After raw2image_ex (shrink=0) + denoise, imgdata.image is
        // ushort[iheight*iwidth][4] holding the visible region. Each pixel has
        // ONLY its CFA-active channel populated (pre-demosaic). scale_colors
        // already applied WB + normalized the white point to ~65535, so
        // dividing by 65535 yields float in [0,1].
        const auto& sizes = rawProcessor.imgdata.sizes;
        const auto& idata = rawProcessor.imgdata.idata;
        const auto& color = rawProcessor.imgdata.color;
        const int W = static_cast<int>(sizes.iwidth);
        const int H = static_cast<int>(sizes.iheight);
        if (W <= 0 || H <= 0) {
            throw std::runtime_error("[CustomPipeline] invalid iwidth/iheight");
        }

        RawMosaic mosaic;
        mosaic.width   = W;
        mosaic.height  = H;
        mosaic.filters = idata.filters;
        mosaic.colors  = idata.colors;
        std::memcpy(mosaic.xtrans, idata.xtrans, sizeof(mosaic.xtrans));
        // LibRaw's color.cam_xyz is float[4][3]; RawMosaic.cam_xyz is
        // double[4][3] — cast element-wise (NOT memcpy), matching
        // decodeRawMosaic.
        for (int c = 0; c < 4; ++c) {
            for (int k = 0; k < 3; ++k) {
                mosaic.cam_xyz[c][k] = static_cast<double>(color.cam_xyz[c][k]);
            }
        }
        mosaic.flip = sizes.flip;
        // Black is already subtracted and the data is already WB-scaled +
        // normalized to [0,1] — downstream stages must NOT re-subtract black or
        // re-apply WB. cblack=0 / maximum=1 keep subtractBlackLevel (skipped)
        // benign if ever re-introduced.
        for (int c = 0; c < 4; ++c) mosaic.cblack[c] = 0.0f;
        mosaic.maximum = 1.0f;
        // cam_mul is set to neutral so highlightInpaintOpposed (which expects
        // pre-WB data) uses a uniform clip threshold on the post-WB mosaic:
        // the max-gain channel saturates at ~1.0, so a flat 0.987 threshold
        // correctly flags sensor-clipped photosites for inpaint-opposed
        // reconstruction. applyWhiteBalance is intentionally skipped because
        // scale_colors already applied the camera WB.
        for (int c = 0; c < 4; ++c) mosaic.cam_mul[c] = 1.0f;

        mosaic.data.resize(static_cast<size_t>(W) * H);
        ushort(*img4)[4] = rawProcessor.imgdata.image;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const int ch = std::min(cfaColor(mosaic, y, x), 3);
                const ushort val = img4[static_cast<size_t>(y) * W + x][ch];
                mosaic.data[static_cast<size_t>(y) * W + x] =
                    static_cast<float>(val) / 65535.0f;
            }
        }

        // --- Custom pipeline (Phases 2-4) on the denoised mosaic ---
        // subtractBlackLevel: SKIPPED (subtract_black_internal did it).
        // applyWhiteBalance:  SKIPPED (scale_colors did it).
        fixHotPixels(mosaic);
        highlightInpaintOpposed(mosaic);

        ImageBuffer img = (mosaic.filters == 9)
            ? xtransMarkesteijnDemosaic(mosaic)
            : rcdDemosaic(mosaic);

        // --- Post-demosaic chroma median via LibRaw's median_filter ---
        // Round-trip img (camera-RGB float [0,1]) through imgdata.image
        // (ushort[4]) so LibRaw's 3x3 chroma median runs in sensor-native
        // space (before the camera->ProPhoto matrix), matching dcraw_process
        // ordering. median_filter uses sizes.width/height as stride; with
        // shrink=0 these equal iwidth/iheight == img dims. Wrapped so a
        // failure degrades to "no median" rather than losing the image.
        try {
            const size_t n = static_cast<size_t>(W) * H;
            for (size_t i = 0; i < n; ++i) {
                const float r = std::clamp(img.data[i * 3 + 0], 0.0f, 1.0f) * 65535.0f;
                const float g = std::clamp(img.data[i * 3 + 1], 0.0f, 1.0f) * 65535.0f;
                const float b = std::clamp(img.data[i * 3 + 2], 0.0f, 1.0f) * 65535.0f;
                img4[i][0] = static_cast<ushort>(r + 0.5f);
                img4[i][1] = static_cast<ushort>(g + 0.5f);
                img4[i][2] = static_cast<ushort>(b + 0.5f);
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

        // --- Phase 5: orientation + camera->ProPhoto matrix + clip ---
        applyFlip(img, mosaic.flip);
        auto M = cameraToProphotoMatrix(mosaic);
        applyColorMatrix(img, M);
        img.clamp();
        return img;
    } catch (const std::exception& e) {
        fprintf(stderr,
            "[CustomPipeline] LibRaw denoise path failed (%s); "
            "falling back to non-denoised custom pipeline\n", e.what());
    } catch (...) {
        fprintf(stderr,
            "[CustomPipeline] LibRaw denoise path failed (unknown); "
            "falling back to non-denoised custom pipeline\n");
    }

    // Fallback: proven non-denoised custom pipeline. nullptr exifCollector
    // avoids duplicating EXIF tags (the primary open already collected them).
    return decodeCustomPipelineNoDenoise(inputPath);
}

} // namespace rawalchemy
