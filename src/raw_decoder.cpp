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

#include <libraw/libraw.h>
#include <cstdio>
#include <cstring>

namespace rawalchemy {

// ---- Error helper ----
static void throwLibRawError(int ret, const char* context) {
    throw std::runtime_error(
        std::string("[RawDecoder] ") + context + " failed: " +
        std::string(libraw_strerror(ret))
    );
}

// ---- decodeRaw ----
ImageBuffer decodeRaw(const std::string& rawPath, const DecodeParams& params,
                       ExifCollector* exifCollector) {
    // Create LibRaw processor
    LibRaw rawProcessor;

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

    // Demosaic algorithm quality
    p.user_qual = params.demosaicQuality;

    // Half-size mode (optional, for fast preview)
    if (params.halfSize) {
        p.half_size = 1;
    }

    // Set EXIF callback before open_file if collector provided
    if (exifCollector) {
        rawProcessor.set_exifparser_handler(
            rawalchemy::getExifCallback(), exifCollector);
    }

    // --- Open the RAW file ---
    int ret = rawProcessor.open_file(rawPath.c_str());
    if (ret != LIBRAW_SUCCESS) {
        throwLibRawError(ret, "open_file");
    }

    // --- Unpack RAW data ---
    ret = rawProcessor.unpack();
    if (ret != LIBRAW_SUCCESS) {
        throwLibRawError(ret, "unpack");
    }

    // --- Process (demosaic + color conversion + gamma) ---
    ret = rawProcessor.dcraw_process();
    if (ret != LIBRAW_SUCCESS) {
        throwLibRawError(ret, "dcraw_process");
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

// ---- decodeRawMosaic (Phase 1) ----
//
// Distinct from decodeRaw(): NO dcraw_process — we read the un-demosaiced
// 1-channel CFA buffer (rawdata.raw_image) and copy the visible region into
// a RawMosaic. All color/demosaic postprocessing is deferred to later phases.
RawMosaic decodeRawMosaic(const std::string& rawPath, ExifCollector* exifCollector) {
    LibRaw rawProcessor;

    // Set EXIF callback before open_file if collector provided (same pattern
    // as decodeRaw()).
    if (exifCollector) {
        rawProcessor.set_exifparser_handler(
            rawalchemy::getExifCallback(), exifCollector);
    }

    // --- Open + unpack only (do NOT call dcraw_process) ---
    int ret = rawProcessor.open_file(rawPath.c_str());
    if (ret != LIBRAW_SUCCESS) {
        throwLibRawError(ret, "open_file (mosaic)");
    }

    ret = rawProcessor.unpack();
    if (ret != LIBRAW_SUCCESS) {
        throwLibRawError(ret, "unpack (mosaic)");
    }

    // --- Reject non-CFA sensors (Foveon / 3-channel color3_image) ---
    // These have no packed raw_image and must use the dcraw_process fallback.
    if (rawProcessor.imgdata.rawdata.raw_image == nullptr) {
        throw std::runtime_error(
            "[RawMosaic] non-CFA sensor; use dcraw_process path");
    }

    const auto& sizes  = rawProcessor.imgdata.sizes;
    const auto& idata  = rawProcessor.imgdata.idata;
    const auto& color  = rawProcessor.imgdata.color;
    const auto& rawdata = rawProcessor.imgdata.rawdata;

    const int visW = static_cast<int>(sizes.width);
    const int visH = static_cast<int>(sizes.height);
    const int top  = static_cast<int>(sizes.top_margin);
    const int left = static_cast<int>(sizes.left_margin);
    const int rawW = static_cast<int>(sizes.raw_width);  // source stride

    if (visW <= 0 || visH <= 0 || rawW <= 0) {
        throw std::runtime_error(
            "[RawMosaic] invalid sizes (width/height/raw_width <= 0)");
    }

    RawMosaic m;
    m.width   = visW;
    m.height  = visH;
    m.filters = idata.filters;
    m.colors  = idata.colors;
    m.is_foveon = (idata.is_foveon != 0);
    std::memcpy(m.xtrans, idata.xtrans, sizeof(m.xtrans));

    // --- Copy visible region as float ---
    // Source stride is raw_width; rows/cols are offset by top/left margins.
    m.data.resize(static_cast<size_t>(visW) * visH);
    const ushort* src = rawdata.raw_image;
    const size_t dstStride = static_cast<size_t>(visW);
    const size_t srcStride = static_cast<size_t>(rawW);
    const size_t srcOffset = static_cast<size_t>(top) * srcStride + left;
    for (int y = 0; y < visH; ++y) {
        const ushort* srcRow = src + srcOffset + static_cast<size_t>(y) * srcStride;
        float* dstRow = m.data.data() + static_cast<size_t>(y) * dstStride;
        for (int x = 0; x < visW; ++x) {
            dstRow[x] = static_cast<float>(srcRow[x]);
        }
    }

    // --- Metadata ---
    m.maximum = static_cast<float>(color.maximum);
    for (int c = 0; c < 4; ++c) {
        m.cam_mul[c] = color.cam_mul[c];
    }
    for (int c = 0; c < 4; ++c) {
        for (int k = 0; k < 3; ++k) {
            m.cam_xyz[c][k] = static_cast<double>(color.cam_xyz[c][k]);
        }
    }
    m.flip = sizes.flip;

    // --- cblack collapse ---
    // Effective per-channel black = scalar `black` + per-channel offset
    // `cblack[c]`. This matches rawpy's black_level_per_channel.
    // NOTE: the 2D darkframe component (cblack[6..], dimensions cblack[4..5])
    // is NOT applied here — Phase 1 limitation. Rare medium-format / Sony
    // sensors with a true per-pixel darkframe must fall back to dcraw_process.
    m.has_2d_darkframe = (color.cblack[4] != 0 || color.cblack[5] != 0);
    for (int c = 0; c < 4; ++c) {
        m.cblack[c] = static_cast<float>(color.black) +
                      static_cast<float>(color.cblack[c]);
    }

    return m;
}

// ---- extractMetadata ----
CameraMetadata extractMetadata(const std::string& rawPath) {
    CameraMetadata meta;
    LibRaw rawProcessor;

    int ret = rawProcessor.open_file(rawPath.c_str());
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

    // Sensor type (populated by open_file, no unpack needed). isNonCfa covers
    // every sensor the custom RCD/Markesteijn pipeline cannot handle: Foveon /
    // non-CFA (filters==0), 4-color RGBE/RGBG2, and sensors with a 2D
    // darkframe (cblack[4..5]) that the custom path does not apply. Also
    // expose the CFA filter code for rcd/xtrans dispatch.
    meta.filters  = idata.filters;
    meta.isNonCfa = (idata.is_foveon != 0)
                 || (idata.filters == 0)
                 || (idata.colors == 4)
                 || (rawProcessor.imgdata.color.cblack[4] != 0)
                 || (rawProcessor.imgdata.color.cblack[5] != 0);

    return meta;
}

} // namespace rawalchemy
