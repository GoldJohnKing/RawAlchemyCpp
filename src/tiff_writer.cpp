/**
 * @file tiff_writer.cpp
 * @brief 16-bit TIFF output with optional ZLIB compression.
 *
 * Matches the Python project's file_io._save_tiff():
 *   - photometric='rgb'
 *   - compression='zlib' (COMPRESSION_ADOBE_DEFLATE) when available
 *   - predictor=2 (horizontal differencing for better compression)
 *   - compression level 8 (balanced speed/size)
 *
 * Falls back to uncompressed if libtiff was built without ZLIB support.
 */

#include "tiff_writer.h"

#include <tiffio.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

namespace rawalchemy {

// ---- Suppress libtiff warnings during compression probing ----
static int sSavedTiffWarningHandler = 0;
static void silentTiffWarningHandler(const char*, const char*, va_list) {
    // intentionally empty — suppress warnings during compression probe
}

static void pushSilentWarnings() {
    sSavedTiffWarningHandler = 1;
    TIFFSetWarningHandler(silentTiffWarningHandler);
}

static void popSilentWarnings() {
    if (sSavedTiffWarningHandler) {
        TIFFSetWarningHandler(nullptr);  // Restore default
        sSavedTiffWarningHandler = 0;
    }
}

// ---- RAII wrapper for TIFF handle ----
struct TiffCloser {
    TIFF* tif;
    explicit TiffCloser(TIFF* t) : tif(t) {}
    ~TiffCloser() { if (tif) TIFFClose(tif); }
    TiffCloser(const TiffCloser&) = delete;
    TiffCloser& operator=(const TiffCloser&) = delete;
};

// ---- Internal: convert float32 image to uint16 buffer ----
static std::vector<uint16_t> floatToUint16(const ImageBuffer& img) {
    const size_t n = img.size();
    std::vector<uint16_t> buf(n);
    for (size_t i = 0; i < n; ++i) {
        buf[i] = static_cast<uint16_t>(
            std::max(0.0f, std::min(1.0f, img.data[i])) * 65535.0f + 0.5f
        );
    }
    return buf;
}

// ---- Internal: write scanlines from a uint16 buffer ----
static bool writeScanlines(TIFF* tif, const std::vector<uint16_t>& buf, int width, int height) {
    const uint32_t rowBytes = static_cast<uint32_t>(width) * 3 * sizeof(uint16_t);
    const uint8_t* rowPtr = reinterpret_cast<const uint8_t*>(buf.data());

    for (int row = 0; row < height; ++row) {
        if (TIFFWriteScanline(tif, const_cast<uint8_t*>(rowPtr + row * rowBytes), row, 0) < 0) {
            fprintf(stderr, "[TiffWriter] Error writing row %d\n", row);
            return false;
        }
    }
    return true;
}

// ---- Internal: set common TIFF tags ----
static void setCommonTags(TIFF* tif, int width, int height) {
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,     static_cast<uint32_t>(width));
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH,    static_cast<uint32_t>(height));
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,  16);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 3);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,    PHOTOMETRIC_RGB);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG,   PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ORIENTATION,    ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP,   1);
    TIFFSetField(tif, TIFFTAG_SOFTWARE,       "Raw Alchemy C++ (Standardized Decoding)");
}

bool writeTiff16(const ImageBuffer& img, const std::string& outPath) {
    if (img.width <= 0 || img.height <= 0 || img.data.empty()) {
        fprintf(stderr, "[TiffWriter] Invalid image dimensions (%dx%d)\n", img.width, img.height);
        return false;
    }

    auto buf16 = floatToUint16(img);

    // Suppress libtiff warnings during compression probing
    pushSilentWarnings();

    // Open TIFF for writing
    TIFF* tif = TIFFOpen(outPath.c_str(), "w");
    if (!tif) {
        popSilentWarnings();
        fprintf(stderr, "[TiffWriter] Cannot open output file: %s\n", outPath.c_str());
        return false;
    }
    TiffCloser closer(tif);

    setCommonTags(tif, img.width, img.height);

    // Attempt ZLIB compression (Adobe Deflate) with horizontal differencing
    // This matches Python's tifffile.imwrite(compression='zlib', predictor=2)
    bool compressed = false;

    // Check if the codec is actually available before trying
    if (TIFFIsCODECConfigured(COMPRESSION_ADOBE_DEFLATE)) {
        TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_ADOBE_DEFLATE);
        TIFFSetField(tif, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
        TIFFSetField(tif, TIFFTAG_ZIPQUALITY, 8);

        if (writeScanlines(tif, buf16, img.width, img.height)) {
            compressed = true;
        }
    }

    if (!compressed) {
        // Close and reopen with no compression
        // (TIFFClose is handled by TiffCloser RAII)
        TIFFClose(tif);
        closer.tif = nullptr;

        // Reopen
        tif = TIFFOpen(outPath.c_str(), "w");
        if (!tif) {
            fprintf(stderr, "[TiffWriter] Cannot reopen output file for uncompressed write\n");
            return false;
        }
        closer.tif = tif;

        setCommonTags(tif, img.width, img.height);
        TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);

        fprintf(stderr, "[TiffWriter] Warning: ZLIB not available, writing uncompressed TIFF\n");

        if (!writeScanlines(tif, buf16, img.width, img.height)) {
            return false;
        }
    }

    TIFFWriteDirectory(tif);
    return true;
}

bool writeTiff16Uncompressed(const ImageBuffer& img, const std::string& outPath) {
    if (img.width <= 0 || img.height <= 0 || img.data.empty()) {
        return false;
    }

    auto buf16 = floatToUint16(img);

    TIFF* tif = TIFFOpen(outPath.c_str(), "w");
    if (!tif) return false;
    TiffCloser closer(tif);

    setCommonTags(tif, img.width, img.height);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);

    if (!writeScanlines(tif, buf16, img.width, img.height)) {
        return false;
    }

    TIFFWriteDirectory(tif);
    return true;
}

} // namespace rawalchemy
