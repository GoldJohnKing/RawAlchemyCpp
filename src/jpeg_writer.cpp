/**
 * @file jpeg_writer.cpp
 * @brief 8-bit JPEG output with high quality settings using libjpeg-turbo.
 *
 * Matches the Python project's file_io._save_jpeg_or_other():
 *   - quality=95
 *   - subsampling=0  → 4:4:4 (no chroma subsampling)
 *   - optimize=True  → Huffman table optimization
 *
 * Uses libjpeg-turbo (TurboJPEG API) for fast, high-quality JPEG encoding.
 */

#include "raw_alchemy/jpeg_writer.h"

#include <cstdio>
#include <vector>
#include <algorithm>

#include <turbojpeg.h>

namespace rawalchemy {

bool writeJpeg(const ImageBuffer& img, const std::string& outPath, int quality) {
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

    // ---- Step 2: Write JPEG via libjpeg-turbo (TurboJPEG API) ----
    tjhandle compressor = tjInitCompress();
    if (!compressor) {
        fprintf(stderr, "[JpegWriter] Failed to initialize TurboJPEG compressor\n");
        return false;
    }

    unsigned char* jpegBuf = nullptr;
    unsigned long jpegSize = 0;

    // Use 4:4:4 subsampling (TJSAMP_444) to match Python Pillow's subsampling=0
    int result = tjCompress2(
        compressor,
        pixels.data(),
        w,              // width
        w * 3,          // pitch (row stride in bytes)
        h,              // height
        TJPF_RGB,       // pixel format
        &jpegBuf,
        &jpegSize,
        TJSAMP_444,     // 4:4:4 chroma subsampling (no subsampling)
        quality,        // quality 1-100
        TJFLAG_ACCURATEDCT  // use accurate DCT for best quality
    );

    if (result != 0) {
        fprintf(stderr, "[JpegWriter] TurboJPEG compression failed: %s\n", tjGetErrorStr());
        tjDestroy(compressor);
        return false;
    }

    // Write compressed JPEG data to file
    FILE* fp = fopen(outPath.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "[JpegWriter] Failed to open output file: %s\n", outPath.c_str());
        tjFree(jpegBuf);
        tjDestroy(compressor);
        return false;
    }

    size_t written = fwrite(jpegBuf, 1, jpegSize, fp);
    fclose(fp);

    tjFree(jpegBuf);
    tjDestroy(compressor);

    if (written != jpegSize) {
        fprintf(stderr, "[JpegWriter] Failed to write JPEG data: %s\n", outPath.c_str());
        return false;
    }

    return true;
}

} // namespace rawalchemy
