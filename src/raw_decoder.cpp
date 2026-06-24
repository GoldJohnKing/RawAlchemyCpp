/**
 * @file raw_decoder.cpp
 * @brief LibRaw-based RAW decoder — Standardized Decoding implementation.
 *
 * Core Philosophy Step 1:
 *   Decode RAW files from any camera into a standardized, wide-gamut
 *   intermediate space — ProPhoto RGB (Linear) at 16-bit depth,
 *   then provide as float32 [0.0, 1.0].
 */

#include "raw_decoder.h"
#include "exif_injector.h"
#include "win_unicode.h"

#include <libraw/libraw.h>
#include <cstdio>
#include <cstring>
#include <algorithm>

#include "aligned_allocator.h"
#include "cfa_lookup.h"
#include "demosaic.h"

// Forward declarations
#include "demosaic_rcd.h"
#include "demosaic_markesteijn.h"
#include "denoise_xtrans.h"

namespace rawalchemy {

// ---- X-Trans denoise config (set before dcraw_process, read in the callback) ----
// denoiseXtransPreWB (the pre_scalecolors_cb hook) is a plain C callback
// void(void*); it cannot receive DecodeParams. decodeRaw fills this
// synchronously right before dcraw_process() fires the callback, so it is
// scoped to a single decode.
struct XtransDenoiseConfig {
    float threshold = 0.0f;  // darktable domain; 0.0 = skip denoise
};
namespace {
thread_local XtransDenoiseConfig tl_xtransDenoise;
} // namespace

// ---- Error helper ----
static void throwLibRawError(int ret, const char* context) {
    throw std::runtime_error(
        std::string("[RawDecoder] ") + context + " failed: " +
        std::string(libraw_strerror(ret))
    );
}

// ---- LibRaw upstream bug workaround ----
// When wavelet denoising is enabled (threshold > 0), LibRaw sets shrink=1 and
// halves iheight/iwidth. After wavelet_denoise(), pre_interpolate() upscales
// the buffer to full size and sets shrink=0, but NEVER updates iheight/iwidth.
static void fixDimensions(LibRaw* raw) {
    raw->imgdata.sizes.iheight = raw->imgdata.sizes.height;
    raw->imgdata.sizes.iwidth = raw->imgdata.sizes.width;
}

// ---- Custom demosaic callbacks (LibRaw native API) ----
// LibRaw calls these via interpolate_bayer_cb / interpolate_xtrans_cb,
// replacing its built-in demosaic entirely. LibRaw handles ALL preprocessing
// (FBDD, wavelet, WB, highlight) before the callback, and ALL postprocessing
// (mix_green, median filter, color conversion, gamma) after.

/// Extract single-channel CFA mosaic from LibRaw's 4-channel image[] buffer.
/// Uses fcol() semantics to determine which channel holds the CFA value.
static AlignedVector<float> extractCfa(const unsigned short (*image)[4],
                                        int w, int h, const LibRaw& raw) {
    const size_t n = static_cast<size_t>(w) * h;
    AlignedVector<float> cfa(n);
    const bool xt = (raw.imgdata.idata.filters == 9);

    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            unsigned c;
            if (xt) {
                c = xtransColor(row, col, raw.imgdata.idata.xtrans);
            } else {
                c = bayerColor(row, col, raw.imgdata.idata.filters);
            }
            cfa[static_cast<size_t>(row) * w + col] =
                static_cast<float>(image[static_cast<size_t>(row) * w + col][c]) / 65535.0f;
        }
    }
    return cfa;
}

/// Write demosaiced planar RGB back into LibRaw's 4-channel image[] buffer.
/// All 4 channels are filled so LibRaw's mix_green and convert_to_rgb work.
static void writeBackRgb(unsigned short (*image)[4],
                          const float* planarRgb, int w, int h) {
    const size_t n = static_cast<size_t>(w) * h;
    for (size_t i = 0; i < n; ++i) {
        auto clamp = [](float v) -> unsigned short {
            return static_cast<unsigned short>(
                std::max(0.0f, std::min(65535.0f, v * 65535.0f)));
        };
        image[i][0] = clamp(planarRgb[i]);
        image[i][1] = clamp(planarRgb[i + n]);
        image[i][2] = clamp(planarRgb[i + 2 * n]);
        image[i][3] = image[i][1];  // Gb = G for mix_green
    }
}

/// Write a denoised single-channel CFA mosaic back into LibRaw's 4-channel
/// image[] buffer, at each pixel's CFA channel only. Inverse of extractCfa();
/// used by the pre-WB denoise callback.
static void writeBackCfa(unsigned short (*image)[4], const float* cfa,
                          int w, int h, const LibRaw& raw) {
    const bool xt = (raw.imgdata.idata.filters == 9);
    auto clamp = [](float v) -> unsigned short {
        return static_cast<unsigned short>(
            std::max(0.0f, std::min(65535.0f, v * 65535.0f)));
    };
    for (int row = 0; row < h; ++row)
        for (int col = 0; col < w; ++col) {
            const unsigned c = xt ? xtransColor(row, col, raw.imgdata.idata.xtrans)
                                  : bayerColor(row, col, raw.imgdata.idata.filters);
            const size_t i = static_cast<size_t>(row) * w + col;
            image[i][c] = clamp(cfa[i]);
        }
}

/// Bayer custom demosaic callback (LibRaw interpolate_bayer_cb).
/// Reads CFA from image[], runs RCD, writes full RGB back.
static void customBayerDemosaic(void* ctx) {
    LibRaw* raw = static_cast<LibRaw*>(ctx);
    fixDimensions(raw);

    auto& img = raw->imgdata;
    int w = static_cast<int>(img.sizes.width);
    int h = static_cast<int>(img.sizes.height);

    auto (*image)[4] = reinterpret_cast<unsigned short (*)[4]>(img.image);
    if (!image) return;

    auto cfa = extractCfa(image, w, h, *raw);
    AlignedVector<float> rgb(static_cast<size_t>(3) * w * h);

    rcd_demosaic(cfa.data(), rgb.data(), w, h, img.idata.filters);

    writeBackRgb(image, rgb.data(), w, h);
}

/// X-Trans custom demosaic callback (LibRaw interpolate_xtrans_cb).
/// Reads CFA from image[], runs 3-pass Markesteijn, writes full RGB back.
static void customXtransDemosaic(void* ctx) {
    LibRaw* raw = static_cast<LibRaw*>(ctx);
    fixDimensions(raw);

    auto& img = raw->imgdata;
    int w = static_cast<int>(img.sizes.width);
    int h = static_cast<int>(img.sizes.height);

    auto (*image)[4] = reinterpret_cast<unsigned short (*)[4]>(img.image);
    if (!image) return;

    auto cfa = extractCfa(image, w, h, *raw);
    AlignedVector<float> rgb(static_cast<size_t>(3) * w * h);

    markesteijn_demosaic(cfa.data(), rgb.data(), w, h, img.idata.xtrans);

    writeBackRgb(image, rgb.data(), w, h);
}

/// Pre-WB X-Trans denoise callback (LibRaw pre_scalecolors_cb). Fires after
/// black-subtract / adjust_maximum / green_matching but BEFORE scale_colors
/// (white balance). imgdata.image[] here is black-subtracted raw CFA — the same
/// domain darktable's rawdenoise operates in, so the sqrt variance stabilizer
/// and noise-floor constants are correctly calibrated. No-op for Bayer and for
/// X-Trans when the threshold is <= 0.
static void denoiseXtransPreWB(void* ctx) {
    LibRaw* raw = static_cast<LibRaw*>(ctx);
    auto& img = raw->imgdata;
    if (!isXtrans(img.idata.filters)) return;
    if (tl_xtransDenoise.threshold <= 0.0f) return;

    int w = static_cast<int>(img.sizes.width);
    int h = static_cast<int>(img.sizes.height);
    auto (*image)[4] = reinterpret_cast<unsigned short (*)[4]>(img.image);
    if (!image) return;

    // in/out must not alias across channels (see denoise_xtrans contract), so
    // denoise into a fresh buffer then write back to the CFA channel positions.
    auto cfa = extractCfa(image, w, h, *raw);
    AlignedVector<float> denoised(static_cast<size_t>(w) * h);
    denoise_xtrans(cfa.data(), denoised.data(), w, h,
                   img.idata.xtrans, tl_xtransDenoise.threshold);
    writeBackCfa(image, denoised.data(), w, h, *raw);
}

/// Pre-interpolate callback: just fixes iheight/iwidth (wavelet denoise bug).
static void preInterpolateFix(void* ctx) {
    fixDimensions(static_cast<LibRaw*>(ctx));
}

// LibRaw::callbacks is protected — expose via subclass.
class LibRawAccessor : public LibRaw {
public:
    using LibRaw::callbacks;
};

// ---- decodeRaw ----
ImageBuffer decodeRaw(const std::string& rawPath, const DecodeParams& params,
                       ExifCollector* exifCollector) {
    LibRawAccessor rawProcessor;

    // --- Configure decoding parameters ---
    auto& p = rawProcessor.imgdata.params;

    p.output_color = params.outputColor;
    p.gamm[0] = 1.0 / static_cast<double>(params.gammaPower);
    p.gamm[1] = static_cast<double>(params.gammaSlope);
    p.output_bps = params.outputBps;
    p.use_camera_wb = params.useCameraWb ? 1 : 0;
    p.no_auto_bright = params.noAutoBright ? 1 : 0;
    p.bright = static_cast<double>(params.bright);
    p.highlight = params.highlightMode;
    p.med_passes = params.medPasses;

    if (params.halfSize) {
        p.half_size = 1;
    }

    // --- Set EXIF callback before open_file ---
    if (exifCollector) {
        rawProcessor.set_exifparser_handler(
            rawalchemy::getExifCallback(), exifCollector);
    }

    // --- Open the RAW file ---
#ifdef _WIN32
    auto widePath = rawalchemy::utf8_to_wide(rawPath);
    int ret = rawProcessor.open_file(widePath.c_str());
#else
    int ret = rawProcessor.open_file(rawPath.c_str());
#endif
    if (ret != LIBRAW_SUCCESS) {
        throwLibRawError(ret, "open_file");
    }

    // --- Unpack RAW data ---
    ret = rawProcessor.unpack();
    if (ret != LIBRAW_SUCCESS) {
        throwLibRawError(ret, "unpack");
    }

    // --- Detect CFA type and register demosaic callbacks ---
    // We need to check filters AFTER open_file (which populates idata).
    const bool isXtransFile = (rawProcessor.imgdata.idata.filters == 9);
    const bool useCustomDemosaic = !params.halfSize && (
        params.demosaicAlgorithm == DemosaicAlgorithm::RCD ||
        params.demosaicAlgorithm == DemosaicAlgorithm::MARKESTEIJN_3PASS ||
        params.demosaicAlgorithm == DemosaicAlgorithm::AUTO
    );

    // Always fix iheight/iwidth (wavelet denoise bug workaround)
    rawProcessor.callbacks.pre_interpolate_cb = preInterpolateFix;
    // X-Trans pre-WB wavelet denoise (darktable port). Self-gates inside the
    // callback: no-op for Bayer and when threshold <= 0. Registered for ALL
    // files so X-Trans denoise works even in LIBRAW_FALLBACK demosaic mode.
    rawProcessor.callbacks.pre_scalecolors_cb = denoiseXtransPreWB;

    if (useCustomDemosaic) {
        // Register our custom demosaic callbacks. LibRaw will call the
        // appropriate one based on filters, replacing its built-in demosaic.
        // ALL preprocessing (FBDD/wavelet/WB) and postprocessing (mix_green,
        // median filter, color conversion, gamma) are handled by LibRaw.
        rawProcessor.callbacks.interpolate_bayer_cb = customBayerDemosaic;
        rawProcessor.callbacks.interpolate_xtrans_cb = customXtransDemosaic;
    } else {
        p.user_qual = params.demosaicQuality;
    }

    // Green matching: disable for X-Trans (LibRaw's implementation is
    // Bayer-only; running it on X-Trans corrupts the green channel).
    p.green_matching = params.greenMatching ? 1 : 0;

    // --- ISO-adaptive noise reduction ---
    float iso = rawProcessor.imgdata.other.iso_speed;

    // X-Trans: LibRaw's wavelet_denoise() inside scale_colors() is designed for
    // Bayer's 2x2 grid; applied to X-Trans's 6x6 CFA it produces row-dependent
    // wavelet artifacts (horizontal banding), so it stays disabled. Our own
    // darktable-ported channel-separated wavelet runs earlier, in the
    // pre_scalecolors_cb hook (before white balance — see Step 6).
    if (isXtransFile) {
        p.threshold = 0;
        tl_xtransDenoise.threshold =
            computeXtransDenoiseThreshold(iso, params.xtransDenoiseThreshold);
    } else {
        tl_xtransDenoise.threshold = 0.0f;
    }
    if (!isXtransFile && params.denoiseThreshold < 0.0f) {
        if (iso <= 100.0f) {
            p.threshold = 0;
        } else if (iso <= 400.0f) {
            p.threshold = 100;
        } else {
            float logLow = 8.644f;
            float logHigh = 13.644f;
            float t = (log2f(iso) - logLow) / (logHigh - logLow);
            p.threshold = 100.0f + std::min(std::max(t, 0.0f), 1.0f) * 900.0f;
        }
    } else if (!isXtransFile) {
        p.threshold = params.denoiseThreshold;
    }

    if (params.fbddNoiserd < 0) {
        // Note: LibRaw internally guards FBDD with "filters > 1000" (dcraw_process.cpp:169),
        // so FBDD never runs for X-Trans (filters == 9) regardless of this setting.
        if (iso <= 100.0f) {
            p.fbdd_noiserd = 0;
        } else if (iso < 3200.0f) {
            p.fbdd_noiserd = 1;
        } else {
            p.fbdd_noiserd = 2;
        }
    } else {
        p.fbdd_noiserd = params.fbddNoiserd;
    }

    // --- Process (demosaic + color conversion + gamma) ---
    // With our callbacks registered, LibRaw calls our RCD/Markesteijn
    // during the demosaic step, then continues with its own postprocessing.
    ret = rawProcessor.dcraw_process();
    if (ret != LIBRAW_SUCCESS) {
        throwLibRawError(ret, "dcraw_process");
    }

    // --- Extract processed image (standard LibRaw output path) ---
    int errCode = 0;
    libraw_processed_image_t* processed = rawProcessor.dcraw_make_mem_image(&errCode);
    if (!processed || errCode != LIBRAW_SUCCESS) {
        throw std::runtime_error(
            "[RawDecoder] dcraw_make_mem_image() failed (error code: " +
            std::to_string(errCode) + ")"
        );
    }

    if (processed->type != LIBRAW_IMAGE_BITMAP) {
        LibRaw::dcraw_clear_mem(processed);
        throw std::runtime_error("[RawDecoder] Expected bitmap output, got JPEG/other");
    }

    int imgWidth  = static_cast<int>(processed->width);
    int imgHeight = static_cast<int>(processed->height);
    int imgColors = static_cast<int>(processed->colors);
    if (imgColors < 3) imgColors = 3;

    // --- Convert to float32 ImageBuffer ---
    ImageBuffer result;

    if (processed->bits == 16) {
        const uint16_t* srcData = reinterpret_cast<const uint16_t*>(processed->data);
        if (imgColors == 3) {
            result = ImageBuffer::fromUint16(srcData, imgWidth, imgHeight, 3);
        } else {
            result = ImageBuffer(imgWidth, imgHeight, 3);
            const size_t pixelCount = result.pixelCount();
            for (size_t i = 0; i < pixelCount; ++i) {
                result.data[i * 3 + 0] = static_cast<float>(srcData[i * imgColors + 0]) / 65535.0f;
                result.data[i * 3 + 1] = static_cast<float>(srcData[i * imgColors + 1]) / 65535.0f;
                result.data[i * 3 + 2] = static_cast<float>(srcData[i * imgColors + 2]) / 65535.0f;
            }
        }
    } else if (processed->bits == 8) {
        const uint8_t* srcData = processed->data;
        result = ImageBuffer(imgWidth, imgHeight, 3);
        const size_t pixelCount = result.pixelCount();
        for (size_t i = 0; i < pixelCount; ++i) {
            result.data[i * 3 + 0] = static_cast<float>(srcData[i * imgColors + 0]) / 255.0f;
            result.data[i * 3 + 1] = static_cast<float>(srcData[i * imgColors + 1]) / 255.0f;
            result.data[i * 3 + 2] = static_cast<float>(srcData[i * imgColors + 2]) / 255.0f;
        }
    } else {
        LibRaw::dcraw_clear_mem(processed);
        throw std::runtime_error(
            "[RawDecoder] Unexpected bit depth: " + std::to_string(processed->bits)
        );
    }

    LibRaw::dcraw_clear_mem(processed);
    return result;
}

// ---- extractMetadata ----
CameraMetadata extractMetadata(const std::string& rawPath) {
    CameraMetadata meta;
    LibRaw rawProcessor;

#ifdef _WIN32
    auto widePath = rawalchemy::utf8_to_wide(rawPath);
    int ret = rawProcessor.open_file(widePath.c_str());
#else
    int ret = rawProcessor.open_file(rawPath.c_str());
#endif
    if (ret != LIBRAW_SUCCESS) {
        throwLibRawError(ret, "open_file (metadata)");
    }

    auto& idata = rawProcessor.imgdata.idata;
    auto& lens  = rawProcessor.imgdata.lens;
    auto& other = rawProcessor.imgdata.other;

    if (idata.make[0])  meta.cameraMaker  = idata.make;
    if (idata.model[0]) meta.cameraModel  = idata.model;
    if (lens.Lens[0])   meta.lensModel    = lens.Lens;
    if (lens.LensMake[0]) meta.lensMaker  = lens.LensMake;

    meta.focalLength = static_cast<float>(other.focal_len);
    meta.aperture    = static_cast<float>(other.aperture);
    meta.isoSpeed    = static_cast<int>(other.iso_speed);

    return meta;
}

} // namespace rawalchemy
