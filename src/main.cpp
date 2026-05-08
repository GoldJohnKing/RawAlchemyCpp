/**
 * @file main.cpp
 * @brief Raw Alchemy C++ — CLI entry point for Standardized Decoding.
 *
 * Step 1 of Core Philosophy:
 *   RAW (Camera Native) -> ProPhoto RGB (Linear) -> 16-bit TIFF
 *
 * Usage:
 *   raw_alchemy <input.raw> <output.tif> [options]
 *
 * Options:
 *   --half-size      Decode at half resolution (fast preview)
 *   --no-camera-wb   Don't use camera white balance
 *   --demosaic N     Demosaic quality (3=AHD, 11=AAHD, default: 3)
 *   --no-compress    Save TIFF without compression
 */

#include "raw_alchemy/raw_decoder.h"
#include "raw_alchemy/tiff_writer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>

// ---- Supported RAW file extensions ----
static bool isRawExtension(const std::string& path) {
    static const char* exts[] = {
        ".dng", ".cr2", ".cr3", ".nef", ".nrw",
        ".arw", ".srf", ".sr2", ".rw2", ".raf",
        ".orf", ".pef", ".srw", ".kdc", ".dcr",
        ".raw", ".3fr", ".iiq", ".mef", ".mos",
        nullptr
    };
    std::string lower = path;
    for (auto& c : lower) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));

    for (int i = 0; exts[i]; ++i) {
        size_t len = strlen(exts[i]);
        if (lower.size() >= len &&
            lower.compare(lower.size() - len, len, exts[i]) == 0) {
            return true;
        }
    }
    return false;
}

static void printUsage(const char* prog) {
    printf(
        "Raw Alchemy C++ — Standardized Decoding\n"
        "========================================\n"
        "\n"
        "Step 1: RAW (Camera Native) -> ProPhoto RGB (Linear) -> 16-bit TIFF\n"
        "\n"
        "Usage:\n"
        "  %s <input.raw> <output.tif> [options]\n"
        "\n"
        "Supported formats: NEF, CR2, CR3, ARW, RW2, RAF, ORF, PEF, DNG, SRW, etc.\n"
        "\n"
        "Options:\n"
        "  --half-size      Decode at half resolution (fast preview)\n"
        "  --no-camera-wb   Don't use camera white balance\n"
        "  --demosaic N     Demosaic quality: 3=AHD (default), 11=AAHD\n"
        "  --no-compress    Save TIFF without ZLIB compression\n"
        "  --info           Only print file metadata, don't decode\n"
        "  -h, --help       Show this help\n"
        "\n",
        prog
    );
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return argc == 1 ? 0 : 1;
    }

    std::string inputPath  = argv[1];
    std::string outputPath = argv[2];

    // Parse options
    bool halfSize    = false;
    bool useCameraWb = true;
    bool compress    = true;
    bool infoOnly    = false;
    int  demosaicQ   = 3;  // AHD

    for (int i = 3; i < argc; ++i) {
        std::string opt = argv[i];
        if (opt == "--half-size")     { halfSize = true; }
        else if (opt == "--no-camera-wb") { useCameraWb = false; }
        else if (opt == "--no-compress")  { compress = false; }
        else if (opt == "--info")     { infoOnly = true; }
        else if (opt == "--demosaic" && i + 1 < argc) {
            demosaicQ = std::atoi(argv[++i]);
        }
        else if (opt == "-h" || opt == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        else {
            fprintf(stderr, "Unknown option: %s\n", opt.c_str());
            return 1;
        }
    }

    // Validate input extension
    if (!isRawExtension(inputPath)) {
        fprintf(stderr, "Warning: Input file '%s' may not be a RAW file.\n", inputPath.c_str());
    }

    printf("[Raw Alchemy] Standardized Decoding\n");
    printf("  Input:  %s\n", inputPath.c_str());
    printf("  Output: %s\n", outputPath.c_str());
    printf("\n");

    try {
        // --- Extract and print metadata ---
        printf("[Step 0] Extracting metadata...\n");
        auto meta = rawalchemy::extractMetadata(inputPath);
        printf("  Camera:  %s %s\n", meta.cameraMaker.c_str(), meta.cameraModel.c_str());
        printf("  Lens:    %s %s\n", meta.lensMaker.c_str(), meta.lensModel.c_str());
        printf("  Focal:   %.1f mm\n", meta.focalLength);
        printf("  Aperture: f/%.1f\n", meta.aperture);
        printf("  ISO:     %d\n", meta.isoSpeed);
        printf("\n");

        if (infoOnly) {
            return 0;
        }

        // --- Step 1: Decode RAW to ProPhoto RGB (Linear) ---
        printf("[Step 1] Decoding RAW -> ProPhoto RGB (Linear)...\n");
        printf("  Color space:  ProPhoto RGB\n");
        printf("  Gamma:        Linear (1.0)\n");
        printf("  Bit depth:    16-bit\n");
        printf("  White balance: %s\n", useCameraWb ? "Camera" : "Auto");
        printf("  Demosaic:     %s (quality=%d)\n",
               demosaicQ == 3 ? "AHD" : demosaicQ == 11 ? "AAHD" : "Other",
               demosaicQ);
        if (halfSize) {
            printf("  Mode:         Half-size (preview)\n");
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        rawalchemy::DecodeParams params;
        params.halfSize = halfSize;
        params.useCameraWb = useCameraWb;
        params.demosaicQuality = demosaicQ;
        // All other params use defaults: ProPhoto, Linear gamma, 16-bit, etc.

        rawalchemy::ImageBuffer img = rawalchemy::decodeRaw(inputPath, params);

        auto t1 = std::chrono::high_resolution_clock::now();
        double decodeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        printf("  -> Done. %dx%d (%.1f MP) in %.0f ms\n",
               img.width, img.height,
               static_cast<double>(img.width * img.height) / 1e6,
               decodeMs);
        printf("\n");

        // --- Step 6: Save as 16-bit TIFF ---
        printf("[Save] Writing 16-bit TIFF...\n");
        printf("  Format:     TIFF (16-bit, %s)\n", compress ? "ZLIB" : "Uncompressed");

        auto t2 = std::chrono::high_resolution_clock::now();

        bool ok;
        if (compress) {
            ok = rawalchemy::writeTiff16(img, outputPath);
        } else {
            ok = rawalchemy::writeTiff16Uncompressed(img, outputPath);
        }

        auto t3 = std::chrono::high_resolution_clock::now();
        double saveMs = std::chrono::duration<double, std::milli>(t3 - t2).count();

        if (!ok) {
            fprintf(stderr, "Error: Failed to write output file.\n");
            return 1;
        }

        printf("  -> Done in %.0f ms\n", saveMs);
        printf("\n");
        printf("[Complete] Standardized Decoding finished.\n");

    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
