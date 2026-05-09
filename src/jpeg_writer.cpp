/**
 * @file jpeg_writer.cpp
 * @brief 8-bit JPEG output with high quality settings using libjpeg-turbo.
 *
 * Matches the Python project's file_io._save_jpeg_or_other():
 *   - quality=95
 *   - subsampling=0  → 4:4:4 (no chroma subsampling)
 *   - optimize=True  → Huffman table optimization (opt-in, off by default)
 *
 * Uses libjpeg-turbo (TurboJPEG 3 API) for fast, high-quality JPEG encoding.
 */

#include "jpeg_writer.h"
#include "icc_srgb.h"

#include <cstdio>
#include <vector>
#include <algorithm>

#include <turbojpeg.h>

namespace rawalchemy {

bool writeJpeg(const ImageBuffer& img, const std::string& outPath,
               int quality, bool optimize) {
    if (img.width <= 0 || img.height <= 0 || img.data.empty()) {
        fprintf(stderr, "[JpegWriter] Invalid image dimensions (%dx%d)\n", img.width, img.height);
        return false;
    }

    // Clamp quality to valid range
    quality = std::max(1, std::min(quality, 100));

    // ---- Step 1: Convert float32 [0,1] → uint8 [0,255] ----
    // Matches Python: output_image_uint8 = (img * 255).astype(np.uint8)
    // with implicit clipping from save_image() np.clip(img, 0.0, 1.0)
    const int w = img.width;
    const int h = img.height;
    std::vector<uint8_t> pixels(static_cast<size_t>(w) * h * 3);

    const float* src = img.data.data();
    uint8_t* dst = pixels.data();

    for (size_t i = 0; i < pixels.size(); ++i) {
        float v = src[i];
        // Clamp + scale
        if (v < 0.0f) v = 0.0f;
        else if (v > 1.0f) v = 1.0f;
        dst[i] = static_cast<uint8_t>(v * 255.0f + 0.5f);
    }

    // ---- Step 2: Write JPEG via libjpeg-turbo (TurboJPEG 3 API) ----
    tjhandle compressor = tj3Init(TJINIT_COMPRESS);
    if (!compressor) {
        fprintf(stderr, "[JpegWriter] Failed to initialize TurboJPEG compressor\n");
        return false;
    }

    // Set compression parameters
    tj3Set(compressor, TJPARAM_QUALITY, quality);
    tj3Set(compressor, TJPARAM_SUBSAMP, TJSAMP_444);  // 4:4:4, matches Pillow subsampling=0
    tj3Set(compressor, TJPARAM_FASTDCT, 0);            // use accurate (slow) DCT
    if (optimize)
        tj3Set(compressor, TJPARAM_OPTIMIZE, 1);       // optimal Huffman tables

    // Embed sRGB ICC profile for output color management
    tj3SetICCProfile(compressor,
                     const_cast<unsigned char*>(SRGB_ICC_PROFILE),
                     SRGB_ICC_PROFILE_SIZE);

    unsigned char* jpegBuf = nullptr;
    size_t jpegSize = 0;

    int result = tj3Compress8(
        compressor,
        pixels.data(),
        w,              // width
        w * 3,          // pitch (row stride in bytes)
        h,              // height
        TJPF_RGB,       // pixel format
        &jpegBuf,
        &jpegSize
    );

    if (result != 0) {
        fprintf(stderr, "[JpegWriter] TurboJPEG compression failed: %s\n",
                tj3GetErrorStr(compressor));
        tj3Destroy(compressor);
        return false;
    }

    // Write compressed JPEG data to file
    FILE* fp = fopen(outPath.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "[JpegWriter] Failed to open output file: %s\n", outPath.c_str());
        tj3Free(jpegBuf);
        tj3Destroy(compressor);
        return false;
    }

    size_t written = fwrite(jpegBuf, 1, jpegSize, fp);
    fclose(fp);

    tj3Free(jpegBuf);
    tj3Destroy(compressor);

    if (written != jpegSize) {
        fprintf(stderr, "[JpegWriter] Failed to write JPEG data: %s\n", outPath.c_str());
        return false;
    }

    return true;
}

} // namespace rawalchemy
