/**
 * @file capi_test.cpp
 * @brief End-to-end test for the C API (raProcessFileWithLUT).
 *
 * Exercises the C API one-shot path on a real RAW file with a real log space
 * (e.g. F-Log2), writes the output via raProcessFileWithLUT (format auto-
 * detected from the output path extension: .tif/.tiff -> 16-bit TIFF,
 * .jpg/.jpeg -> JPEG), then reads the dimensions back to confirm a well-formed
 * file. Proves the C API now routes through the custom CPU demosaic pipeline
 * (Phases 1-5), producing the SAME output as `raw_alchemy_cli --demosaic auto`.
 *
 * The JPEG path also exercises the EXIF pipeline (exifCollector is passed
 * through decodeRawMosaic -> callback fires during open_file -> EXIF APP1 is
 * embedded in the JPEG). Test/cross_validate_capi.py verifies the EXIF marker.
 *
 * Used by Test/cross_validate_capi.py.
 *
 * Usage: raw_alchemy_capi_test <input.raw> <output.{tiff|jpg}> <log-space>
 *   Prints "CAPI_OK <W> <H>" on success (return 0); error message + return 1
 *   on failure.
 */

#include "raw_alchemy_capi.h"

#include <tiffio.h>

#include <cstdio>
#include <cstring>
#include <string>

namespace {

bool hasSuffix(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

bool isJpegPath(const std::string& p) {
    return hasSuffix(p, ".jpg") || hasSuffix(p, ".jpeg") ||
           hasSuffix(p, ".JPG") || hasSuffix(p, ".JPEG");
}

// Read image dimensions from a TIFF via libtiff. Returns true on success.
bool readTiffDims(const char* path, uint32_t& w, uint32_t& h) {
    TIFFSetWarningHandler(nullptr);  // silence libtiff read warnings
    TIFF* tif = TIFFOpen(path, "r");
    if (!tif) return false;
    bool ok = (TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w) == 1) &&
              (TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h) == 1);
    TIFFClose(tif);
    return ok && w > 0 && h > 0;
}

// Read image dimensions from a JPEG by scanning for a SOF (Start Of Frame)
// marker. SOF carries: [precision(1)][height(2)][width(2)] big-endian. The SOF
// always precedes SOS (entropy data), so the scan is safe. Returns true on
// success.
bool readJpegDims(const char* path, uint32_t& w, uint32_t& h) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    unsigned char b[2];
    // Expect SOI (FF D8).
    if (std::fread(b, 1, 2, f) != 2 || b[0] != 0xFF || b[1] != 0xD8) {
        std::fclose(f);
        return false;
    }
    for (;;) {
        // Align to a marker: read 0xFF (skipping any FF fill bytes).
        do {
            if (std::fread(b, 1, 1, f) != 1) { std::fclose(f); return false; }
        } while (b[0] != 0xFF);
        // Read the marker code (skip FF padding).
        do {
            if (std::fread(b, 1, 1, f) != 1) { std::fclose(f); return false; }
        } while (b[0] == 0xFF);
        const unsigned char marker = b[0];

        // Standalone markers (no segment body): RSTn (D0-D7), SOI (D8),
        // EOI (D9), TEM (01).
        if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD9)) continue;

        // All other markers carry a 2-byte big-endian segment length.
        unsigned char lenb[2];
        if (std::fread(lenb, 1, 2, f) != 2) { std::fclose(f); return false; }
        const unsigned segLen = static_cast<unsigned>((lenb[0] << 8) | lenb[1]);
        if (segLen < 2) { std::fclose(f); return false; }

        // SOF markers: C0-CF except C4 (DHT), C8 (JPG), CC (DAC).
        const bool isSof = (marker >= 0xC0 && marker <= 0xCF) &&
                           marker != 0xC4 && marker != 0xC8 && marker != 0xCC;
        if (isSof) {
            // SOF payload (after the 2-byte length): precision(1) height(2) width(2).
            unsigned char sof[5];
            if (std::fread(sof, 1, 5, f) != 5) { std::fclose(f); return false; }
            h = static_cast<uint32_t>((sof[1] << 8) | sof[2]);
            w = static_cast<uint32_t>((sof[3] << 8) | sof[4]);
            std::fclose(f);
            return w > 0 && h > 0;
        }
        // Skip the remainder of this segment (segLen includes its 2 length bytes).
        if (std::fseek(f, static_cast<long>(segLen - 2), SEEK_CUR) != 0) {
            std::fclose(f);
            return false;
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr,
            "Usage: %s <input.raw> <output.{tiff|jpg}> <log-space>\n",
            argv[0]);
        return 2;
    }

    const char* inputPath = argv[1];
    const char* outputPath = argv[2];
    const char* logSpace = argv[3];
    const std::string outStr(outputPath);

    printf("[capi_test] raProcessFileWithLUT('%s' -> '%s', log='%s')\n",
           inputPath, outputPath, logSpace);

    // Run the C API one-shot path: decode (custom demosaic pipeline) ->
    // exposure -> sat/contrast -> log -> (no LUT) -> output. No LUT (NULL
    // table), auto exposure via matrix metering, no lens correction. Output
    // format is auto-detected from the extension by raProcessFileWithLUT.
    RaResult res = raProcessFileWithLUT(
        inputPath,
        outputPath,
        logSpace,
        /*lutTable*/ nullptr,
        /*lutSize*/ 0,
        /*lutDomainMin*/ nullptr,
        /*lutDomainMax*/ nullptr,
        /*metering*/ "matrix",
        /*manualEv*/ 0.0f,
        /*useAutoExposure*/ 1,
        /*jpegQuality*/ 90,
        /*enableLensCorrection*/ 0,
        /*customLensfunDb*/ nullptr);

    if (res != RA_OK) {
        const char* err = raGetLastError();
        fprintf(stderr, "[capi_test] raProcessFileWithLUT failed: %d (%s)\n",
                static_cast<int>(res), err ? err : "(no error message)");
        return 1;
    }

    // Read back dimensions (format-aware) to confirm a well-formed output and
    // surface the image size to the cross-validation script.
    uint32_t W = 0, H = 0;
    bool ok;
    if (isJpegPath(outStr)) {
        ok = readJpegDims(outputPath, W, H);
    } else {
        ok = readTiffDims(outputPath, W, H);
    }
    if (!ok) {
        fprintf(stderr,
                "[capi_test] output written, but dimension read-back failed: %s\n",
                outputPath);
        return 1;
    }

    printf("CAPI_OK %u %u\n", W, H);
    return 0;
}
