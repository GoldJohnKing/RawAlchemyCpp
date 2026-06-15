/**
 * @file main.cpp
 * @brief Raw Alchemy C++ — Full pipeline CLI.
 *
 * Pipeline: RAW -> ProPhoto Linear -> [Exposure] -> [Sat/Cont Boost]
 *           -> Log Gamut -> Log Curve -> [LUT] -> 16-bit TIFF
 */

#include "raw_decoder.h"
#include "raw_mosaic.h"
#include "raw_preprocess.h"
#include "highlight.h"
#include "demosaic.h"
#include "colorspace_matrices.h"
#include "raw_postprocess.h"
#include "metering.h"
#include "stylize.h"
#include "log_transform.h"
#include "lut_applier.h"
#include "tiff_writer.h"
#include "jpeg_writer.h"
#include "lens_correction.h"
#include "exif_injector.h"

#if defined(__aarch64__)
#include "half_buffer.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>

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
        if (lower.size() >= len && lower.compare(lower.size() - len, len, exts[i]) == 0)
            return true;
    }
    return false;
}

static void printUsage(const char* prog) {
    printf(
        "Raw Alchemy C++\n"
        "===============\n"
        "\n"
        "Pipeline: RAW -> ProPhoto Linear -> [Lens Correction] -> Exposure -> Sat/Cont -> Log -> LUT -> Image\n"
        "\n"
        "Usage:\n"
        "  %s <input.raw> <output> --log-space <space> [options]\n"
        "\n"
        "  Output format is auto-detected from file extension (.tif/.tiff/.jpg/.jpeg)\n"
        "  or can be set explicitly with --format.\n"
        "\n"
        "Log Spaces: F-Log, F-Log2, F-Log2C, V-Log, N-Log, L-Log,\n"
        "            Canon Log 2, Canon Log 3, S-Log3, S-Log3.Cine,\n"
        "            Arri LogC3, Arri LogC4, Log3G10, D-Log\n"
        "\n"
        "Exposure (auto by default, mutually exclusive):\n"
        "  --exposure EV      Manual exposure in stops (disables auto)\n"
        "  --metering MODE    Auto metering mode (default: matrix)\n"
        "                     Modes: average, center-weighted, highlight-safe,\n"
        "                            hybrid, matrix\n"
        "\n"
        "Lens Correction:\n"
        "  --lens-correction  Enable lens correction (default: enabled)\n"
        "  --no-lens-correction\n"
        "                     Disable lens correction\n"
        "  --custom-lensfun-db PATH\n"
        "                     Path to custom Lensfun XML database (file or directory)\n"
        "\n"
        "Options:\n"
        "  --lut FILE         Apply a .cube 3D LUT after log encoding\n"
        "  --format FMT       Output format: tif, jpg (default: auto from extension)\n"
        "  --no-boost         Disable saturation/contrast boost\n"
        "  --half-size        Decode at half resolution (fast preview)\n"
        "  --no-camera-wb     Don't use camera white balance\n"
        "  --demosaic MODE    Demosaic path: auto (default), rcd, xtrans, ahd.\n"
        "                     auto = RCD on Bayer / Markesteijn on X-Trans;\n"
        "                     ahd (or non-CFA / 4-color sensors) uses LibRaw's\n"
        "                     dcraw path. rcd/xtrans error if sensor mismatches.\n"
        "  --no-compress      Save TIFF without compression\n"
        "  --jpeg-quality N   JPEG quality 1-100 (default: 95)\n"
        "  --jpeg-optimize   Optimize Huffman tables (smaller file, slower)\n"
        "  --info             Only print metadata\n"
        "  --dump-mosaic PATH Debug: dump normalized CFA mosaic (uint16 LE)\n"
        "                     after black-subtraction + hot-pixel fix, then exit.\n"
        "                     Prints 'MOSAIC W H' to stdout for cross-validation.\n"
        "  --dump-mosaic-stage {black|highlight}\n"
        "                     Stage at which to dump (default: black).\n"
        "                     'highlight' also runs highlight reconstruction\n"
        "                     before dumping.\n"
        "  --dump-demosaic PATH\n"
        "                     Debug: run the CPU RAW pipeline\n"
        "                     (decode -> black -> hot-pixel -> highlight ->\n"
        "                     demosaic) and write the resulting camera-RGB\n"
        "                     ImageBuffer (H*W*3 float32, RGB interleaved,\n"
        "                     pre-WB/pre-matrix) to PATH, then exit.\n"
        "                     Demosaic auto-selects by sensor: Markesteijn\n"
        "                     1-pass for X-Trans (filters==9), RCD otherwise.\n"
        "                     Prints 'DEMOSAIC W H' to stdout for\n"
        "                     cross-validation. Diagnostic only — full\n"
        "                     --demosaic output wiring (WB+matrix) is Phase 5.\n"
        "  --dump-matrix PATH Debug: derive the Camera->ProPhoto matrix for the\n"
        "                     input RAW and write it (3 lines x 3 floats, row-\n"
        "                     major, whitespace-separated, full precision) to\n"
        "                     PATH, then exit. Prints 'MATRIX' to stdout.\n"
        "  -h, --help         Show this help\n"
        "\n",
        prog
    );
}

int main(int argc, char* argv[]) {
    if (argc < 3) { printUsage(argv[0]); return argc == 1 ? 0 : 1; }

    std::string inputPath  = argv[1];
    std::string outputPath = argv[2];

    // Parse options
    std::string logSpace;
    std::string lutPath;
    std::string meteringMode = "matrix";  // default
    std::string outputFormat;             // empty = auto-detect from extension
    std::string customLensfunDb;          // empty = system default
    std::string dumpMosaicPath;           // empty = no mosaic dump
    std::string dumpMosaicStage = "black"; // default stage
    std::string dumpDemosaicPath;         // empty = no demosaic dump
    std::string dumpMatrixPath;           // empty = no matrix dump
    float  exposure   = 0.0f;
    bool   hasExposure = false;  // true only if --exposure explicitly given
    bool   halfSize    = false;
    bool   useCameraWb = true;
    bool   doBoost     = true;
    bool   compress    = true;
    bool   infoOnly    = false;
    bool   lensCorrection = true;
    std::string demosaicMode = "auto";   // {auto, rcd, xtrans, ahd}
    int    jpegQuality  = 95;
    bool   jpegOptimize = false;

    for (int i = 3; i < argc; ++i) {
        std::string opt = argv[i];
        if (opt == "--log-space" && i + 1 < argc)      { logSpace = argv[++i]; }
        else if (opt == "--lut" && i + 1 < argc)        { lutPath = argv[++i]; }
        else if (opt == "--metering" && i + 1 < argc)    { meteringMode = argv[++i]; }
        else if (opt == "--exposure" && i + 1 < argc)    { exposure = std::strtof(argv[++i], nullptr); hasExposure = true; }
        else if (opt == "--format" && i + 1 < argc)      { outputFormat = argv[++i]; }
        else if (opt == "--half-size")                   { halfSize = true; }
        else if (opt == "--no-camera-wb")                { useCameraWb = false; }
        else if (opt == "--no-boost")                    { doBoost = false; }
        else if (opt == "--no-compress")                 { compress = false; }
        else if (opt == "--jpeg-quality" && i + 1 < argc){ jpegQuality = std::atoi(argv[++i]); }
        else if (opt == "--jpeg-optimize")               { jpegOptimize = true; }
        else if (opt == "--info")                        { infoOnly = true; }
        else if (opt == "--demosaic" && i + 1 < argc)    { demosaicMode = argv[++i]; }
        else if (opt == "--lens-correction")             { lensCorrection = true; }
        else if (opt == "--no-lens-correction")          { lensCorrection = false; }
        else if (opt == "--custom-lensfun-db" && i + 1 < argc) { customLensfunDb = argv[++i]; }
        else if (opt == "--dump-mosaic" && i + 1 < argc)   { dumpMosaicPath = argv[++i]; }
        else if (opt == "--dump-mosaic-stage" && i + 1 < argc) {
            dumpMosaicStage = argv[++i];
            if (dumpMosaicStage != "black" && dumpMosaicStage != "highlight") {
                fprintf(stderr, "Error: --dump-mosaic-stage must be 'black' or 'highlight' (got '%s')\n",
                        dumpMosaicStage.c_str());
                return 1;
            }
        }
        else if (opt == "--dump-demosaic" && i + 1 < argc) { dumpDemosaicPath = argv[++i]; }
        else if (opt == "--dump-matrix" && i + 1 < argc) { dumpMatrixPath = argv[++i]; }
        else if (opt == "-h" || opt == "--help")         { printUsage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown option: %s\n", opt.c_str()); return 1; }
    }

    // Validate --demosaic mode.
    if (demosaicMode != "auto" && demosaicMode != "rcd" &&
        demosaicMode != "xtrans" && demosaicMode != "ahd") {
        fprintf(stderr, "Error: --demosaic must be one of {auto, rcd, xtrans, ahd} (got '%s')\n",
                demosaicMode.c_str());
        return 1;
    }

    if (!isRawExtension(inputPath))
        fprintf(stderr, "Warning: '%s' may not be a RAW file.\n", inputPath.c_str());

    if (!infoOnly && !logSpace.empty() && !rawalchemy::isLogSpaceSupported(logSpace)) {
        fprintf(stderr, "Error: Unsupported log space '%s'\n", logSpace.c_str());
        return 1;
    }
    if (hasExposure && !rawalchemy::isMeteringModeSupported(meteringMode)) {
        fprintf(stderr, "Error: Unsupported metering mode '%s'\n", meteringMode.c_str());
        return 1;
    }

    printf("[Raw Alchemy] C++ Edition\n");
    printf("  Input:  %s\n", inputPath.c_str());
    printf("  Output: %s\n", outputPath.c_str());
    if (!logSpace.empty()) printf("  Log:    %s\n", logSpace.c_str());
    if (!lutPath.empty())  printf("  LUT:    %s\n", lutPath.c_str());
    if (hasExposure)       printf("  Exposure: %+.2f stops (manual)\n", exposure);
    else                   printf("  Exposure: auto (%s)\n", meteringMode.c_str());
    printf("\n");

    try {
        // --- Step 0: Metadata ---
        printf("[Step 0] Metadata\n");
        auto meta = rawalchemy::extractMetadata(inputPath);
        printf("  Camera:  %s %s\n", meta.cameraMaker.c_str(), meta.cameraModel.c_str());
        printf("  Lens:    %s %s\n", meta.lensMaker.c_str(), meta.lensModel.c_str());
        printf("  Focal:   %.1f mm  Aperture: f/%.1f  ISO: %d\n",
               meta.focalLength, meta.aperture, meta.isoSpeed);
        printf("\n");

        if (infoOnly) return 0;

        // --- Debug: dump normalized CFA mosaic and exit (Phase 1 hook) ---
        if (!dumpMosaicPath.empty()) {
            // Decode mosaic (no dcraw_process), run Phase 1 preprocess,
            // optionally Phase 2 highlight reconstruction, then write uint16
            // LE binary and print "MOSAIC W H" for cross-validation.
            auto mosaic = rawalchemy::decodeRawMosaic(inputPath);
            rawalchemy::subtractBlackLevel(mosaic);
            rawalchemy::fixHotPixels(mosaic);
            if (dumpMosaicStage == "highlight") {
                rawalchemy::highlightInpaintOpposed(mosaic);
            }

            std::vector<uint16_t> buf(mosaic.data.size());
            for (size_t i = 0; i < mosaic.data.size(); ++i) {
                float v = mosaic.data[i];
                if (!(v > 0.0f)) v = 0.0f;        // also catches NaN
                if (v > 1.0f) v = 1.0f;
                buf[i] = static_cast<uint16_t>(v * 65535.0f + 0.5f);
            }

            FILE* fp = std::fopen(dumpMosaicPath.c_str(), "wb");
            if (!fp) {
                fprintf(stderr, "Error: cannot write mosaic to '%s'\n",
                        dumpMosaicPath.c_str());
                return 1;
            }
            const size_t n = buf.size() * sizeof(uint16_t);
            if (std::fwrite(buf.data(), 1, n, fp) != n) {
                std::fclose(fp);
                fprintf(stderr, "Error: short write to '%s'\n",
                        dumpMosaicPath.c_str());
                return 1;
            }
            std::fclose(fp);

            // Single-line machine-readable marker (cross-validation script
            // parses this). Keep on its own line.
            printf("MOSAIC %d %d\n", mosaic.width, mosaic.height);
            return 0;
        }

        // --- Debug: dump demosaiced camera-RGB and exit (Phase 3-4 hook) ---
        if (!dumpDemosaicPath.empty()) {
            // Run the full CPU RAW pipeline through demosaic and write the
            // resulting camera-RGB ImageBuffer (H*W*3 float32, RGB interleaved,
            // PRE white-balance / PRE color matrix) to PATH.
            //
            // Phase 4: auto-select the demosaic by sensor — X-Trans sensors
            // (filters == 9) use Markesteijn 1-pass; all others use RCD.
            auto mosaic = rawalchemy::decodeRawMosaic(inputPath);
            rawalchemy::subtractBlackLevel(mosaic);
            rawalchemy::fixHotPixels(mosaic);
            rawalchemy::highlightInpaintOpposed(mosaic);
            auto img = (mosaic.filters == 9)
                ? rawalchemy::xtransMarkesteijnDemosaic(mosaic)
                : rawalchemy::rcdDemosaic(mosaic);

            FILE* fp = std::fopen(dumpDemosaicPath.c_str(), "wb");
            if (!fp) {
                fprintf(stderr, "Error: cannot write demosaic to '%s'\n",
                        dumpDemosaicPath.c_str());
                return 1;
            }
            const size_t n = img.data.size() * sizeof(float);
            if (std::fwrite(img.data.data(), 1, n, fp) != n) {
                std::fclose(fp);
                fprintf(stderr, "Error: short write to '%s'\n",
                        dumpDemosaicPath.c_str());
                return 1;
            }
            std::fclose(fp);

            // Machine-readable marker for the cross-validation script.
            printf("DEMOSAIC %d %d\n", img.width, img.height);
            return 0;
        }

        // --- Debug: dump derived Camera->ProPhoto matrix and exit (Phase 5) ---
        if (!dumpMatrixPath.empty()) {
            // Decode the mosaic (open + unpack only) and derive the analytical
            // Camera -> ProPhoto RGB matrix (port of colorspace_matrices.py).
            auto mosaic = rawalchemy::decodeRawMosaic(inputPath);
            auto M = rawalchemy::cameraToProphotoMatrix(mosaic);

            FILE* fp = std::fopen(dumpMatrixPath.c_str(), "w");
            if (!fp) {
                fprintf(stderr, "Error: cannot write matrix to '%s'\n",
                        dumpMatrixPath.c_str());
                return 1;
            }
            // 3 lines of 3 whitespace-separated floats (row-major, full precision).
            for (int i = 0; i < 3; ++i) {
                std::fprintf(fp, "%.16g %.16g %.16g\n",
                             static_cast<double>(M[i][0]),
                             static_cast<double>(M[i][1]),
                             static_cast<double>(M[i][2]));
            }
            std::fclose(fp);

            // Machine-readable marker for the cross-validation script.
            printf("MATRIX\n");
            return 0;
        }

        // --- Step 1: Decode ---
        printf("[Step 1] Decode -> ProPhoto RGB (Linear)...\n");
        auto t0 = std::chrono::high_resolution_clock::now();

        rawalchemy::DecodeParams params;
        params.halfSize = halfSize;
        params.useCameraWb = useCameraWb;
        params.demosaicQuality = 3;  // AHD for the dcraw fallback path

        // Collect EXIF tags during decode if output is JPEG
        // (must be set before open_file, so we create collector before decode)
        bool needExif = false;
        {
            std::string ext = outputPath;
            for (auto& c : ext) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
            if (!outputFormat.empty())
                needExif = (outputFormat == "jpg" || outputFormat == "jpeg");
            else
                needExif = (ext.size() >= 4 && ext.compare(ext.size()-4, 4, ".jpg") == 0) ||
                           (ext.size() >= 5 && ext.compare(ext.size()-5, 5, ".jpeg") == 0);
        }

        rawalchemy::ExifCollector* exifCollector = nullptr;
        if (needExif) {
            exifCollector = rawalchemy::createExifCollector();
        }

        rawalchemy::ImageBuffer img;  // ProPhoto RGB linear — filled below.

        // ============================================================
        // Phase 5: route between the custom CPU demosaic pipeline
        // (decodeRawMosaic -> preprocess -> highlight -> demosaic ->
        // WB -> flip -> camera->ProPhoto matrix) and the legacy dcraw
        // path (decodeRaw -> dcraw_process). Both produce a ProPhoto RGB
        // linear ImageBuffer that the downstream consumes.
        // ============================================================
        enum class DecodePath { Custom, Dcraw };
        DecodePath path = DecodePath::Dcraw;

        if (demosaicMode == "ahd") {
            // Explicit dcraw/AHD — skip the mosaic probe entirely.
            path = DecodePath::Dcraw;
        } else {
            // Probe the mosaic to learn sensor type (filters/colors/darkframe).
            // decodeRawMosaic throws for non-CFA (Foveon) sensors.
            try {
                auto mosaic = rawalchemy::decodeRawMosaic(inputPath);

                if (mosaic.has_2d_darkframe) {
                    // 2D darkframe isn't applied in the custom path — rare
                    // medium-format / some Sony sensors. Fall back + warn.
                    fprintf(stderr,
                        "Warning: sensor has a 2D darkframe (cblack[4..5]); "
                        "falling back to dcraw path (custom pipeline does not "
                        "apply the 2D darkframe).\n");
                    path = DecodePath::Dcraw;
                } else if (mosaic.is_foveon || mosaic.filters == 0) {
                    // Non-CFA sensor — dcraw only.
                    path = DecodePath::Dcraw;
                } else if (mosaic.colors == 4) {
                    // 4-color RGBE/RGBG2 — enforce dcraw fallback.
                    path = DecodePath::Dcraw;
                } else if (demosaicMode == "auto") {
                    path = DecodePath::Custom;  // RCD on Bayer, Markesteijn on X-Trans.
                } else if (demosaicMode == "rcd") {
                    if (mosaic.filters == 9) {
                        if (exifCollector) rawalchemy::destroyExifCollector(exifCollector);
                        fprintf(stderr,
                            "Error: --demosaic rcd requires a Bayer sensor "
                            "(this sensor reports filters==9 / X-Trans).\n");
                        return 1;
                    }
                    path = DecodePath::Custom;
                } else if (demosaicMode == "xtrans") {
                    if (mosaic.filters != 9) {
                        if (exifCollector) rawalchemy::destroyExifCollector(exifCollector);
                        fprintf(stderr,
                            "Error: --demosaic xtrans requires an X-Trans "
                            "sensor (filters==9; this sensor is Bayer or other).\n");
                        return 1;
                    }
                    path = DecodePath::Custom;
                }

                if (path == DecodePath::Custom) {
                    // Run the full custom CPU RAW pipeline.
                    printf("  [custom path] mosaic %dx%d filters=0x%x colors=%d\n",
                           mosaic.width, mosaic.height, mosaic.filters, mosaic.colors);
                    rawalchemy::subtractBlackLevel(mosaic);
                    rawalchemy::fixHotPixels(mosaic);
                    rawalchemy::highlightInpaintOpposed(mosaic);
                    auto camRgb = (mosaic.filters == 9)
                        ? rawalchemy::xtransMarkesteijnDemosaic(mosaic)
                        : rawalchemy::rcdDemosaic(mosaic);

                    // WB multiply (green-anchored) on demosaiced RGB.
                    rawalchemy::applyWhiteBalance(camRgb, mosaic.cam_mul);
                    // Orientation flip (post-demosaic, on RGB).
                    rawalchemy::applyFlip(camRgb, mosaic.flip);

                    // Camera -> ProPhoto RGB matrix (float64-derived, cast here).
                    auto M = rawalchemy::cameraToProphotoMatrix(mosaic);
                    float Mf[3][3] = {
                        {M[0][0], M[0][1], M[0][2]},
                        {M[1][0], M[1][1], M[1][2]},
                        {M[2][0], M[2][1], M[2][2]},
                    };
                    rawalchemy::applyColorMatrix(camRgb, Mf);
                    // Clip to [0,1] — matches the Python reference (core.py:221)
                    // and the dcraw path's uint16->float normalization.
                    for (auto& v : camRgb.data) {
                        v = std::max(0.0f, std::min(1.0f, v));
                    }
                    img = std::move(camRgb);
                }
                // else: fall through to the dcraw path below (double-decode;
                // the mosaic probe is discarded).
            } catch (const std::exception& e) {
                // Non-CFA sensor (decodeRawMosaic threw) — fall back to dcraw.
                // Only an error if the user explicitly asked for rcd/xtrans.
                if (demosaicMode == "rcd" || demosaicMode == "xtrans") {
                    if (exifCollector) rawalchemy::destroyExifCollector(exifCollector);
                    fprintf(stderr,
                        "Error: --demosaic %s not supported for this sensor (%s).\n",
                        demosaicMode.c_str(), e.what());
                    return 1;
                }
                path = DecodePath::Dcraw;
            }
        }

        if (path == DecodePath::Dcraw) {
            printf("  [dcraw path] LibRaw dcraw_process (AHD demosaic + ProPhoto matrix)\n");
            try {
                img = rawalchemy::decodeRaw(inputPath, params, exifCollector);
            } catch (...) {
                if (exifCollector) rawalchemy::destroyExifCollector(exifCollector);
                throw;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        printf("  -> %dx%d (%.1f MP) in %.0f ms\n",
               img.width, img.height,
               static_cast<double>(img.width * img.height) / 1e6,
               std::chrono::duration<double, std::milli>(t1 - t0).count());
        printf("\n");

        // --- Step 1.5: Lens Correction ---
        if (lensCorrection) {
            printf("[Step 1.5] Lens Correction...\n");
            auto tLC0 = std::chrono::high_resolution_clock::now();

            rawalchemy::LensCorrectionParams lcParams;
            lcParams.enabled = true;
            lcParams.correctDistortion = true;
            lcParams.correctTca = true;
            lcParams.correctVignetting = true;
            lcParams.distance = 1000.0f;
            lcParams.customDbPath = customLensfunDb;

            bool applied = rawalchemy::applyLensCorrection(img, meta, lcParams);

            auto tLC1 = std::chrono::high_resolution_clock::now();
            if (!applied) {
                printf("  -> Skipped (lens not found or no correction data)\n");
            }
            printf("  -> Done in %.0f ms\n",
                   std::chrono::duration<double, std::milli>(tLC1 - tLC0).count());
            printf("\n");
        }

        // --- Step 2: Exposure ---
        if (hasExposure) {
            float gain = std::pow(2.0f, exposure);
            printf("[Step 2] Manual Exposure: %+.2f stops (gain=%.4f)\n", exposure, gain);
            img.applyGain(gain);
        } else {
            printf("[Step 2] Auto Exposure (mode=%s)...\n", meteringMode.c_str());
            float gain = rawalchemy::computeAutoGain(img, meteringMode);
            printf("  -> Gain: %.4f\n", gain);
            img.applyGain(gain);
        }
        printf("\n");

        // --- Step 3: Camera-Match Boost ---
        if (doBoost) {
            printf("[Step 3] Camera-Match Boost (sat=1.25, cont=1.10)\n");
            auto tB0 = std::chrono::high_resolution_clock::now();
            rawalchemy::applySaturationContrast(img, 1.25f, 1.10f);
            auto tB1 = std::chrono::high_resolution_clock::now();
            printf("  -> Done in %.0f ms\n",
                   std::chrono::duration<double, std::milli>(tB1 - tB0).count());
            printf("\n");
        }

        // --- Step 4: Log Signal Preparation ---
        if (!logSpace.empty()) {
            printf("[Step 4] Log Transform: %s\n", logSpace.c_str());
            auto t2 = std::chrono::high_resolution_clock::now();

            // Gamut transform always in float32 (linear light, needs precision)
            rawalchemy::applyGamutTransform(img, logSpace);

            #if defined(__aarch64__)
            // ARM64: convert to float16 for log encoding + LUT (halves memory bandwidth)
            auto imgF16 = rawalchemy::convertToF16(img);
            rawalchemy::applyLogEncodingF16(imgF16, logSpace);

            // LUT also in float16 on ARM64
            if (!lutPath.empty()) {
                printf("[Step 5] LUT: %s\n", lutPath.c_str());
                auto tL0 = std::chrono::high_resolution_clock::now();
                auto lut = rawalchemy::loadCubeLUT(lutPath);
                rawalchemy::applyLUT3DF16(imgF16, lut);
                auto tL1 = std::chrono::high_resolution_clock::now();
                printf("  -> Done in %.0f ms\n",
                       std::chrono::duration<double, std::milli>(tL1 - tL0).count());
                printf("\n");
            }

            // Convert back to float32 for output
            img = rawalchemy::convertToF32(imgF16);
            #else
            // Non-ARM: use float32 pipeline as before
            rawalchemy::applyLogEncoding(img, logSpace);
            #endif

            auto t3 = std::chrono::high_resolution_clock::now();
            printf("  -> Done in %.0f ms\n",
                   std::chrono::duration<double, std::milli>(t3 - t2).count());
            printf("\n");
        }

        #if !defined(__aarch64__)
        // --- Step 5: LUT (non-ARM path only; ARM handles LUT inside Step 4) ---
        if (!lutPath.empty()) {
            printf("[Step 5] LUT: %s\n", lutPath.c_str());
            auto tL0 = std::chrono::high_resolution_clock::now();
            auto lut = rawalchemy::loadCubeLUT(lutPath);
            rawalchemy::applyLUT3D(img, lut);
            auto tL1 = std::chrono::high_resolution_clock::now();
            printf("  -> Done in %.0f ms\n",
                   std::chrono::duration<double, std::milli>(tL1 - tL0).count());
            printf("\n");
        }
        #endif

        // --- Determine output format ---
        // Auto-detect from file extension if --format not specified
        // Matches Python's file_io.save_image() behavior: dispatch by extension
        enum class OutFormat { Tiff, Jpeg };
        OutFormat fmt = OutFormat::Tiff;  // default

        if (!outputFormat.empty()) {
            // Explicit --format override
            if (outputFormat == "jpg" || outputFormat == "jpeg") {
                fmt = OutFormat::Jpeg;
            } else if (outputFormat == "tif" || outputFormat == "tiff") {
                fmt = OutFormat::Tiff;
            } else {
                fprintf(stderr, "Error: Unsupported format '%s' (use: tif, jpg)\n", outputFormat.c_str());
                return 1;
            }
        } else {
            // Auto-detect from output path extension
            std::string ext = outputPath;
            for (auto& c : ext) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
            if (ext.size() >= 4 && ext.compare(ext.size() - 4, 4, ".jpg") == 0)
                fmt = OutFormat::Jpeg;
            else if (ext.size() >= 5 && ext.compare(ext.size() - 5, 5, ".jpeg") == 0)
                fmt = OutFormat::Jpeg;
        }

        // --- Save ---
        bool ok = false;
        auto tS0 = std::chrono::high_resolution_clock::now();

        switch (fmt) {
        case OutFormat::Tiff:
            printf("[Save] 16-bit TIFF (%s)...\n", compress ? "ZLIB" : "Uncompressed");
            ok = compress
                ? rawalchemy::writeTiff16(img, outputPath)
                : rawalchemy::writeTiff16Uncompressed(img, outputPath);
            break;

        case OutFormat::Jpeg:
            printf("[Save] 8-bit JPEG (quality=%d, 4:4:4%s)...\n", jpegQuality,
                   jpegOptimize ? ", optimize" : "");
            {
                std::vector<uint8_t> exifBlob;
                if (exifCollector) {
                    exifBlob = rawalchemy::buildExifBlob(*exifCollector, img.width, img.height);
                    if (!exifBlob.empty())
                        printf("  -> EXIF: %zu bytes embedded\n", exifBlob.size());
                    else
                        printf("  -> EXIF: no tags collected\n");
                }
                ok = rawalchemy::writeJpeg(img, outputPath, jpegQuality, jpegOptimize,
                                            exifBlob.empty() ? nullptr : &exifBlob);
            }
            break;
        }

        auto tS1 = std::chrono::high_resolution_clock::now();
        if (!ok) { fprintf(stderr, "Error: Failed to write output.\n"); return 1; }
        printf("  -> Done in %.0f ms\n",
               std::chrono::duration<double, std::milli>(tS1 - tS0).count());
        printf("\n[Complete]\n");

        // Cleanup EXIF collector
        if (exifCollector) rawalchemy::destroyExifCollector(exifCollector);

    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
