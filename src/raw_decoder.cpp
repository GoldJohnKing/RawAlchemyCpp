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

#include "raw_alchemy/raw_decoder.h"

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
ImageBuffer decodeRaw(const std::string& rawPath, const DecodeParams& params) {
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

    return meta;
}

} // namespace rawalchemy
