/**
 * @file raw_decoder.cpp
 * @brief LibRaw-based RAW decoder — Standardized Decoding implementation.
 *
 * Core Philosophy Step 1:
 *   Decode RAW files from any camera into a standardized, wide-gamut
 *   intermediate space — ProPhoto RGB (Linear) at 16-bit depth,
 *   then provide as float32 [0.0, 1.0].
 *
 * This eliminates color science differences between camera brands,
 * providing a unified starting point for all subsequent operations.
 *
 * Equivalent to the Python project's core.py Step 1:
 *   rawpy.postprocess(
 *       gamma=(1, 1),
 *       no_auto_bright=True,
 *       use_camera_wb=True,
 *       output_bps=16,
 *       output_color=rawpy.ColorSpace.ProPhoto,  # = 4
 *       bright=1.0,
 *       highlight_mode=2,        # Blend
 *       demosaic_algorithm=rawpy.DemosaicAlgorithm.AAHD
 *   )
 */

#include "raw_decoder.h"
#include "exif_injector.h"
#include "win_unicode.h"

#include <libraw/libraw.h>
#include <cstdio>
#include <cstring>

#include "aligned_allocator.h"
#include "cfa_lookup.h"
#include "color_convert.h"
#include "demosaic.h"

namespace rawalchemy {

// Forward declarations — definitions live in demosaic_dispatch.cpp today
// (bilinear stubs); real RCD and Markesteijn ports arrive in Tasks 9 and 13.
void rcd_demosaic(const float* in, float* out, int w, int h, unsigned filters);
void markesteijn_demosaic(const float* in, float* out, int w, int h,
                           const unsigned char xtrans[6][6]);

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
// DHT (quality=11) and AAHD (quality=12) use iheight/iwidth for buffer indexing,
// causing them to process only the top-left quarter of the image.
// Inspired by RawTherapee's integration-layer workaround (rtengine/rawimage.cc).
static void fixPreInterpolateDimensions(void* ctx) {
    LibRaw* raw = static_cast<LibRaw*>(ctx);
    raw->imgdata.sizes.iheight = raw->imgdata.sizes.height;
    raw->imgdata.sizes.iwidth = raw->imgdata.sizes.width;
}

// ---- Demosaic snapshot infrastructure ----
// Thread-local snapshot of LibRaw's pre-demosaic state. Populated by
// capturePreInterpolateState() at the pre_interpolate_cb hook point (after
// FBDD/wavelet/WB/highlight recovery, before demosaic). Consumed by the custom
// demosaic path that replaces LibRaw's built-in interpolation (Task 6+).
struct DemosaicSnapshot {
    bool captured = false;
    bool isXtrans = false;
    unsigned filters = 0;
    unsigned char xtrans[6][6] = {};
    int width = 0;
    int height = 0;
    AlignedVector<float> rawCfa;  // single-channel mosaic, preprocessed
};

// Thread-local because LibRaw processes one image per thread, and the callback
// signature has no user-data parameter — TLS is the only way to reach the buffer.
thread_local DemosaicSnapshot g_demosaicSnapshot;

// LibRaw pre_interpolate_cb: snapshots preprocessed CFA data before demosaic.
// Also applies the iheight/iwidth bug fix (same as fixPreInterpolateDimensions
// above), so the demosaicAlgorithm branch in decodeRaw() can swap the callback
// assignment without losing the workaround.
static void capturePreInterpolateState(void* ctx) {
    LibRaw* raw = static_cast<LibRaw*>(ctx);

    // Preserve existing bug fix (see fixPreInterpolateDimensions above).
    // Runs unconditionally — the dimension correction is needed whether or not
    // snapshot capture succeeds, and must not be skipped by an exception.
    raw->imgdata.sizes.iheight = raw->imgdata.sizes.height;
    raw->imgdata.sizes.iwidth = raw->imgdata.sizes.width;

    // Capture only once per decode (LibRaw may invoke the hook multiple times)
    if (g_demosaicSnapshot.captured) return;

    // Wrap the capture body so a failure (bad alloc, LibRaw state surprise) falls
    // back to LibRaw's own demosaic instead of propagating a C++ exception through
    // LibRaw's C-style internals.
    try {
        auto& img = raw->imgdata;
        int w = static_cast<int>(img.sizes.width);
        int h = static_cast<int>(img.sizes.height);

        g_demosaicSnapshot.isXtrans = isXtrans(img.idata.filters);
        g_demosaicSnapshot.filters = img.idata.filters;
        if (g_demosaicSnapshot.isXtrans) {
            std::memcpy(g_demosaicSnapshot.xtrans, img.idata.xtrans, 36);
        }
        g_demosaicSnapshot.width = w;
        g_demosaicSnapshot.height = h;

        const size_t pixelCount = static_cast<size_t>(w) * h;
        g_demosaicSnapshot.rawCfa.resize(pixelCount);

        // LibRaw stores imgdata.image as ushort (*)[4] (libraw_types.h:1115),
        // NOT float. Task 5's cast to float(*)[4] was UB; we read the native
        // ushort and normalize to float [0,1] inline. extractCfaFromImage() is
        // intentionally untouched (Task 4 tests still consume float input).
        const unsigned short (*image)[4] =
            reinterpret_cast<const unsigned short (*)[4]>(img.image);

        if (!image) {
            std::fprintf(stderr,
                         "[rawalchemy] Demosaic snapshot: imgdata.image is null, skipping\n");
            return;  // captured stays false -> decodeRaw falls back to LibRaw output
        }

        // Extract single-channel CFA with ushort->float normalization.
        for (int row = 0; row < h; ++row) {
            for (int col = 0; col < w; ++col) {
                unsigned c;
                if (g_demosaicSnapshot.isXtrans) {
                    c = xtransColor(row, col, g_demosaicSnapshot.xtrans);
                } else {
                    c = bayerColor(row, col, g_demosaicSnapshot.filters);
                }
                g_demosaicSnapshot.rawCfa[static_cast<size_t>(row) * w + col] =
                    static_cast<float>(image[static_cast<size_t>(row) * w + col][c]) / 65535.0f;
            }
        }

        g_demosaicSnapshot.captured = true;
        std::fprintf(stderr,
                     "[rawalchemy] Demosaic snapshot captured: %dx%d, xtrans=%d\n",
                     w, h, g_demosaicSnapshot.isXtrans ? 1 : 0);
    } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "[rawalchemy] Demosaic snapshot failed: %s (falling back to LibRaw demosaic)\n",
                     e.what());
        g_demosaicSnapshot.captured = false;
    } catch (...) {
        std::fprintf(stderr,
                     "[rawalchemy] Demosaic snapshot failed: unknown error (falling back to LibRaw demosaic)\n");
        g_demosaicSnapshot.captured = false;
    }
}

// LibRaw::callbacks is protected — expose via subclass.
class LibRawAccessor : public LibRaw {
public:
    using LibRaw::callbacks;
};

// ---- decodeRaw ----
ImageBuffer decodeRaw(const std::string& rawPath, const DecodeParams& params,
                       ExifCollector* exifCollector) {
    // Create LibRaw processor
    LibRawAccessor rawProcessor;

    // --- Configure decoding parameters ---
    auto& p = rawProcessor.imgdata.params;

    // Color space: 4 = ProPhoto RGB
    // Matches rawpy.ColorSpace.ProPhoto
    p.output_color = params.outputColor;

    // Gamma: linear response
    // In dcraw/LibRaw convention: gamm[0] = 1/power
    // rawpy passes gamma=(1,1) which sets gamm[0]=1/1=1.0, gamm[1]=1.0
    // Result: pow(x, 1/gamm[0]) = pow(x, 1.0) = x  -> linear
    p.gamm[0] = 1.0 / static_cast<double>(params.gammaPower);
    p.gamm[1] = static_cast<double>(params.gammaSlope);

    // Bit depth: 16-bit to preserve dynamic range for Log conversion
    p.output_bps = params.outputBps;

    // White balance: use camera's built-in WB
    p.use_camera_wb = params.useCameraWb ? 1 : 0;

    // No auto brightness — we control exposure ourselves
    p.no_auto_bright = params.noAutoBright ? 1 : 0;

    // Brightness multiplier: 1.0 = no change
    p.bright = static_cast<double>(params.bright);

    // Highlight recovery: 2 = Blend mode (prevents highlight clipping)
    p.highlight = params.highlightMode;

    // Half-size mode (optional, for fast preview) — bypasses our custom demosaic
    // because LibRaw downsamples before the pre_interpolate hook fires.
    if (params.halfSize) {
        p.half_size = 1;
    }

    // Determine whether to use our custom demosaic or LibRaw's built-in path.
    // LIBRAW_FALLBACK and half_size mode both delegate to LibRaw unchanged.
    // Determine whether to use our custom demosaic or LibRaw's built-in path.
    const bool isXtransFile = (rawProcessor.imgdata.idata.filters == 9);
    const bool useCustomDemosaic = !params.halfSize && (
        params.demosaicAlgorithm == DemosaicAlgorithm::RCD ||
        params.demosaicAlgorithm == DemosaicAlgorithm::MARKESTEIJN_3PASS ||
        params.demosaicAlgorithm == DemosaicAlgorithm::AUTO
    );

    if (useCustomDemosaic) {
        // Run LibRaw preprocessing (FBDD/wavelet/WB/highlight) but use the cheapest
        // demosaic as a placeholder — its output is discarded and replaced by our
        // custom demosaic built from the pre-interpolate snapshot.
        p.user_qual = 0;  // bilinear (fastest)
        // Reset snapshot state for this decode (clears stale thread-local data)
        g_demosaicSnapshot = DemosaicSnapshot{};
        // Register our snapshot callback (also applies the iheight/iwidth bug fix)
        rawProcessor.callbacks.pre_interpolate_cb = capturePreInterpolateState;
    } else {
        // Original LibRaw path
        p.user_qual = params.demosaicQuality;
        rawProcessor.callbacks.pre_interpolate_cb = fixPreInterpolateDimensions;
    }

    // Green channel matching (G1/G3 equalization)
    // Green matching equilibrates Gr/Gb channels. It is designed for Bayer CFA
    // only — LibRaw's implementation uses FC() (which returns 0 for X-Trans
    // filters==9) and accesses channel 3 (always zero for X-Trans). Running it
    // on X-Trans data corrupts the green channel, causing horizontal banding.
    p.green_matching = (params.greenMatching && !isXtransFile) ? 1 : 0;

    // Mix green: average G1/G3 into single channel, forces P1.colors=3.
    // Without this, some sensors output 4-channel RGBG which breaks
    // non-AHD demosaic algorithms (DHT, AAHD).
    // rawProcessor.imgdata.rawdata.ioparams.mix_green = 1;

    // Median filter passes (post-demosaic, chroma only)
    p.med_passes = params.medPasses;

    // Pre-demosaic noise reduction (FBDD)
    p.fbdd_noiserd = params.fbddNoiserd;

    // Set EXIF callback before open_file if collector provided
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

    // --- ISO-adaptive noise reduction ---
    float iso = rawProcessor.imgdata.other.iso_speed;

    if (params.denoiseThreshold < 0.0f) {
        if (iso <= 100.0f) {
            // Base ISO — sensor noise floor is negligible
            p.threshold = 0;
        } else if (iso <= 400.0f) {
            // Low ISO — minimal fixed denoise
            p.threshold = 100;
        } else {
            // ISO 400–12800+: log2-scale mapping to 200–1000
            // ISO is logarithmic (each doubling = same noise increase),
            // so threshold should scale with log2(ISO).
            float logLow = 8.644f;   // log2(400)
            float logHigh = 13.644f; // log2(12800)
            float t = (log2f(iso) - logLow) / (logHigh - logLow);
            p.threshold = 100.0f + std::min(std::max(t, 0.0f), 1.0f) * 900.0f;
        }
    } else {
        p.threshold = params.denoiseThreshold;
    }

    if (params.fbddNoiserd < 0) {
        if (iso <= 100.0f) {
            // Disable FBDD for base ISO (sensor data is clean)
            p.fbdd_noiserd = 0;
        } else if (iso < 3200.0f) {
            // Low ISO — minimal fixed denoise
            p.fbdd_noiserd = 1;
        } else {
            // ISO-adaptive FBDD: upgrade to full mode for high-ISO images
            p.fbdd_noiserd = 2;
        }
    } else {
        p.fbdd_noiserd = params.fbddNoiserd;
    }

    // --- Process (demosaic + color conversion + gamma) ---
    ret = rawProcessor.dcraw_process();
    if (ret != LIBRAW_SUCCESS) {
        throwLibRawError(ret, "dcraw_process");
    }

    // --- Custom demosaic path ---
    if (useCustomDemosaic && g_demosaicSnapshot.captured) {
        auto& snap = g_demosaicSnapshot;
        const size_t pixelCount = static_cast<size_t>(snap.width) * snap.height;

        // Planar RGB output: [R plane | G plane | B plane]
        AlignedVector<float> planarRgb(3 * pixelCount);

        // Dispatch to the appropriate algorithm based on CFA type
        if (snap.isXtrans) {
            if (params.demosaicAlgorithm == DemosaicAlgorithm::RCD) {
                throw std::runtime_error(
                    "RCD demosaic requested but input is X-Trans; "
                    "use MARKESTEIJN_3PASS or AUTO");
            }
            markesteijn_demosaic(snap.rawCfa.data(), planarRgb.data(),
                                  snap.width, snap.height, snap.xtrans);
        } else {
            if (params.demosaicAlgorithm == DemosaicAlgorithm::MARKESTEIJN_3PASS) {
                throw std::runtime_error(
                    "MARKESTEIJN_3PASS requested but input is Bayer; "
                    "use RCD or AUTO");
            }
            rcd_demosaic(snap.rawCfa.data(), planarRgb.data(),
                          snap.width, snap.height, snap.filters);
        }

        // Apply camera-RGB → ProPhoto matrix.
        // Must replicate LibRaw's convert_to_rgb(): out_cam = prophoto_rgb × rgb_cam.
        // Using rgb_cam alone produces sRGB, not ProPhoto, causing wrong colors
        // in the downstream LUT/grading pipeline.
        float out_cam[3][4];
        computeProPhotoMatrix(rawProcessor.imgdata.color.rgb_cam, out_cam);
        applyCameraToProPhoto(planarRgb.data(), snap.width, snap.height, out_cam);

        // Convert planar -> interleaved ImageBuffer and return
        return interleavePlanarRgb(planarRgb.data(), snap.width, snap.height);
    }

    // --- Extract processed image ---
    int errCode = 0;
    libraw_processed_image_t* processed = rawProcessor.dcraw_make_mem_image(&errCode);
    if (!processed || errCode != LIBRAW_SUCCESS) {
        throw std::runtime_error(
            "[RawDecoder] dcraw_make_mem_image() failed (error code: " +
            std::to_string(errCode) + ")"
        );
    }

    // Verify it's a bitmap image (not JPEG)
    if (processed->type != LIBRAW_IMAGE_BITMAP) {
        LibRaw::dcraw_clear_mem(processed);
        throw std::runtime_error("[RawDecoder] Expected bitmap output, got JPEG/other");
    }

    int imgWidth  = static_cast<int>(processed->width);
    int imgHeight = static_cast<int>(processed->height);
    int imgColors = static_cast<int>(processed->colors);  // 3 for RGB

    // Ensure we treat as 3-channel (even if colors=4, the 4th is typically alpha/unused)
    if (imgColors < 3) {
        imgColors = 3;
    }

    // --- Convert to float32 ImageBuffer ---
    ImageBuffer result;

    if (processed->bits == 16) {
        // 16-bit data: uint16 -> float32 [0.0, 1.0]
        const uint16_t* srcData = reinterpret_cast<const uint16_t*>(processed->data);

        if (imgColors == 3) {
            // Direct conversion: 3 channels, interleaved
            result = ImageBuffer::fromUint16(srcData, imgWidth, imgHeight, 3);
        } else {
            // 4-channel (RGBG or RGBA): extract only RGB
            result = ImageBuffer(imgWidth, imgHeight, 3);
            const size_t pixelCount = result.pixelCount();
            for (size_t i = 0; i < pixelCount; ++i) {
                result.data[i * 3 + 0] = static_cast<float>(srcData[i * imgColors + 0]) / 65535.0f;
                result.data[i * 3 + 1] = static_cast<float>(srcData[i * imgColors + 1]) / 65535.0f;
                result.data[i * 3 + 2] = static_cast<float>(srcData[i * imgColors + 2]) / 65535.0f;
            }
        }
    } else if (processed->bits == 8) {
        // 8-bit data: uint8 -> float32 [0.0, 1.0]
        const uint8_t* srcData = processed->data;
        result = ImageBuffer(imgWidth, imgHeight, 3);
        const size_t pixelCount = result.pixelCount();

        for (size_t i = 0; i < pixelCount; ++i) {
            if (imgColors == 3) {
                result.data[i * 3 + 0] = static_cast<float>(srcData[i * 3 + 0]) / 255.0f;
                result.data[i * 3 + 1] = static_cast<float>(srcData[i * 3 + 1]) / 255.0f;
                result.data[i * 3 + 2] = static_cast<float>(srcData[i * 3 + 2]) / 255.0f;
            } else {
                result.data[i * 3 + 0] = static_cast<float>(srcData[i * imgColors + 0]) / 255.0f;
                result.data[i * 3 + 1] = static_cast<float>(srcData[i * imgColors + 1]) / 255.0f;
                result.data[i * 3 + 2] = static_cast<float>(srcData[i * imgColors + 2]) / 255.0f;
            }
        }
    } else {
        LibRaw::dcraw_clear_mem(processed);
        throw std::runtime_error(
            "[RawDecoder] Unexpected bit depth: " + std::to_string(processed->bits)
        );
    }

    // Free the processed image
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

    // Access image data fields populated by open_file
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
