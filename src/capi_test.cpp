/**
 * @file capi_test.cpp
 * @brief End-to-end test for the C API — exercises all 3 one-shot entry points.
 *
 * Calls raProcessFileWithLUT (path/table-based LUT), raProcessFile (path-based
 * LUT, same pipeline minus the LUT-table params), and raProcessToBuffer
 * (returns a buffer handle instead of writing a file). All three share
 * runPipelineImpl internally, so this is API-surface coverage: every one-shot
 * entry point must return RA_OK and produce sane output.
 *
 * The PRIMARY output (raProcessFileWithLUT) is written to argv[2] unchanged so
 * Test/cross_validate_capi.py's per-channel-means + EXIF regression checks
 * keep working. raProcessFile writes a sibling file (suffix before the
 * extension) whose read-back dimensions must match the primary; raProcessToBuffer
 * returns a handle whose W/H must match and which must be destroyed.
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

// Build a sibling output path by inserting `suffix` before the extension, so
// the format-determining extension is preserved (e.g. "/o/out.tiff" + ".pf"
// -> "/o/out.pf.tiff"). Falls back to appending if no extension is present.
std::string makeSiblingPath(const std::string& base, const std::string& suffix) {
    auto slash = base.find_last_of("/\\");
    auto dot = base.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return base + suffix;
    }
    return base.substr(0, dot) + suffix + base.substr(dot);
}

// Read back the dimensions of a written file (TIFF via libtiff, JPEG via SOF
// scan). Format is picked from the path extension. Returns true on success.
bool readBackDims(const std::string& path, uint32_t& w, uint32_t& h) {
    if (isJpegPath(path)) {
        return readJpegDims(path.c_str(), w, h);
    }
    return readTiffDims(path.c_str(), w, h);
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

    // ============================================================
    // 1) raProcessFileWithLUT — PRIMARY output (written to argv[2]).
    //    cross_validate_capi.py reads this back for means + EXIF checks.
    // ============================================================
    printf("[capi_test] raProcessFileWithLUT('%s' -> '%s', log='%s')\n",
           inputPath, outputPath, logSpace);

    // No LUT (NULL table), auto exposure (evOffset=0 = matrix metering, no
    // manual bias), no lens correction. Output format is auto-detected from
    // the extension by raProcessFileWithLUT.
    RaResult res = raProcessFileWithLUT(
        inputPath,
        outputPath,
        logSpace,
        /*lutTable*/ nullptr,
        /*lutSize*/ 0,
        /*lutDomainMin*/ nullptr,
        /*lutDomainMax*/ nullptr,
        /*metering*/ "matrix",
        /*evOffset*/ 0.0f,
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
    if (!readBackDims(outStr, W, H)) {
        fprintf(stderr,
                "[capi_test] primary output written, but dimension read-back failed: %s\n",
                outputPath);
        return 1;
    }

    // ============================================================
    // 2) raProcessFile — path-based LUT entry point (same pipeline,
    //    NULL lutPath). Writes a sibling file; dims must match primary.
    // ============================================================
    const std::string pfPath = makeSiblingPath(outStr, ".pf");
    printf("[capi_test] raProcessFile('%s' -> '%s', log='%s')\n",
           inputPath, pfPath.c_str(), logSpace);

    res = raProcessFile(
        inputPath,
        pfPath.c_str(),
        logSpace,
        /*lutPath*/ nullptr,
        /*metering*/ "matrix",
        /*evOffset*/ 0.0f,
        /*jpegQuality*/ 90,
        /*enableLensCorrection*/ 0,
        /*customLensfunDb*/ nullptr);

    if (res != RA_OK) {
        const char* err = raGetLastError();
        fprintf(stderr, "[capi_test] raProcessFile failed: %d (%s)\n",
                static_cast<int>(res), err ? err : "(no error message)");
        return 1;
    }

    uint32_t pfW = 0, pfH = 0;
    if (!readBackDims(pfPath, pfW, pfH)) {
        fprintf(stderr,
                "[capi_test] raProcessFile output written, but dimension read-back failed: %s\n",
                pfPath.c_str());
        return 1;
    }
    if (pfW != W || pfH != H) {
        fprintf(stderr,
                "[capi_test] raProcessFile dims %ux%u != primary %ux%u\n",
                pfW, pfH, W, H);
        return 1;
    }

    // ============================================================
    // 3) raProcessToBuffer — returns a buffer handle (no file output).
    //    Handle dims must match the primary; caller must destroy.
    // ============================================================
    printf("[capi_test] raProcessToBuffer('%s', log='%s')\n",
           inputPath, logSpace);

    RaImageBuffer buf = nullptr;
    res = raProcessToBuffer(
        inputPath,
        logSpace,
        /*lutPath*/ nullptr,
        /*metering*/ "matrix",
        /*evOffset*/ 0.0f,
        /*enableLensCorrection*/ 0,
        /*customLensfunDb*/ nullptr,
        &buf);

    if (res != RA_OK) {
        const char* err = raGetLastError();
        fprintf(stderr, "[capi_test] raProcessToBuffer failed: %d (%s)\n",
                static_cast<int>(res), err ? err : "(no error message)");
        return 1;
    }
    const int bufW = raImageGetWidth(buf);
    const int bufH = raImageGetHeight(buf);
    const int bufBytes = raImageGetDataSizeBytes(buf);
    raImageBufferDestroy(buf);

    if (static_cast<uint32_t>(bufW) != W || static_cast<uint32_t>(bufH) != H) {
        fprintf(stderr,
                "[capi_test] raProcessToBuffer dims %dx%d != primary %ux%u\n",
                bufW, bufH, W, H);
        return 1;
    }
    if (bufBytes != static_cast<int>(W) * static_cast<int>(H) * 3 * 4) {
        fprintf(stderr,
                "[capi_test] raProcessToBuffer data size %d != expected %d\n",
                bufBytes, static_cast<int>(W) * static_cast<int>(H) * 3 * 4);
        return 1;
    }

    printf("CAPI_OK %u %u\n", W, H);
    return 0;
}
