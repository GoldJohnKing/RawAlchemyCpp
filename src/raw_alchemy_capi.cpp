/**
 * @file raw_alchemy_capi.cpp
 * @brief C API implementation — one-shot processing via C++ rawalchemy API.
 */

#include "raw_alchemy_capi.h"

#include "raw_decoder.h"
#include "raw_pipeline.h"
#include "metering.h"
#include "tiff_writer.h"
#include "jpeg_writer.h"
#include "lens_correction.h"
#include "exif_injector.h"
#include "image_resize.h"
#include "lut_applier.h"
#include "grading_fused.h"

#include <libraw/libraw.h>
#include <cstring>
#include <memory>
#include <string>
#include <stdexcept>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
namespace {
    std::wstring utf8_to_wide(const std::string& utf8) {
        if (utf8.empty()) return std::wstring();
        int size = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
        if (size <= 0) return std::wstring();
        std::wstring wide(static_cast<size_t>(size - 1), 0);
        MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, &wide[0], size);
        return wide;
    }
}
#endif

// ----------------------------------------------------------------
//  Thread-local error message storage
// ----------------------------------------------------------------
namespace {

thread_local std::string g_lastError;

void setError(const std::string& msg) {
    g_lastError = msg;
}

void clearError() {
    g_lastError.clear();
}

// Probe sensor type via the cheap metadata-only open, then route: non-CFA
// (Foveon / 4-color / 2D-darkframe) sensors to decodeRaw()/dcraw; everything
// else to the custom RCD/Markesteijn pipeline. EXIF is collected exactly once
// by the chosen decode path. Shared across all three one-shot entry points
// (raProcessFile, raProcessFileWithLUT, raProcessToBuffer).
rawalchemy::ImageBuffer decodeWithFallback(
    const rawalchemy::CameraMetadata& meta,
    const std::string& path,
    rawalchemy::ExifCollector* ec) {
    return meta.isNonCfa
        ? rawalchemy::decodeRaw(path, rawalchemy::DecodeParams{}, ec)
        : rawalchemy::decodeImageWithCustomPipeline(path, ec);
}

RaResult catchExceptions(const char* context) {
    try {
        throw;
    } catch (const std::runtime_error& e) {
        setError(std::string(context) + ": " + e.what());
        return RA_ERR_UNKNOWN;
    } catch (const std::bad_alloc&) {
        setError(std::string(context) + ": out of memory");
        return RA_ERR_OUT_OF_MEMORY;
    } catch (...) {
        setError(std::string(context) + ": unknown error");
        return RA_ERR_UNKNOWN;
    }
}

} // anonymous namespace

// ----------------------------------------------------------------
//  Opaque handle implementation
// ----------------------------------------------------------------
struct RaImageBuffer_ {
    rawalchemy::ImageBuffer img;
};

struct RaPreviewSession_ {
    rawalchemy::ImageBuffer baseImage;     // full-res: lens-corrected (if enabled) or raw decode
    rawalchemy::ImageBuffer previewCache;  // downscaled preview base (no grading), width=0 until built
    std::string inputPath;
    int cacheMaxWidth;                      // previewCache's maxWidth (0 = uninitialized)
    int cacheMaxHeight;                     // previewCache's maxHeight
};

// ----------------------------------------------------------------
//  Handle Lifecycle
// ----------------------------------------------------------------

RA_API void RA_CALL raImageBufferDestroy(RaImageBuffer buf) {
    delete buf;
}

// ----------------------------------------------------------------
//  Image Buffer Access
// ----------------------------------------------------------------

RA_API int RA_CALL raImageGetWidth(RaImageBuffer buf) {
    return buf ? buf->img.width : 0;
}

RA_API int RA_CALL raImageGetHeight(RaImageBuffer buf) {
    return buf ? buf->img.height : 0;
}

RA_API const float* RA_CALL raImageGetData(RaImageBuffer buf) {
    return buf ? buf->img.ptr() : nullptr;
}

RA_API int RA_CALL raImageGetDataSizeBytes(RaImageBuffer buf) {
    if (!buf) return 0;
    return static_cast<int>(buf->img.size() * sizeof(float));
}

// ----------------------------------------------------------------
//  Internal: run the full pipeline on an already-decoded image
// ----------------------------------------------------------------
namespace {

// ----------------------------------------------------------------
//  Internal: unified pipeline. The three public-ish helpers below
//  (runPipeline / runPipelineWithLUT / runGradingOnly) are thin
//  forwarders that select the LUT source and whether lens correction
//  runs; ~90% of the body lives here.
// ----------------------------------------------------------------
// Contract: at most one of {lutPath, lutPtr} non-null (path takes precedence).
// meta is read ONLY when enableLensCorrection != 0.
static RaResult runPipelineImpl(
    rawalchemy::ImageBuffer& img,
    const rawalchemy::CameraMetadata& meta,
    const char* logSpace,
    const char* metering,
    float evOffset,
    const char* lutPath,                         // mode A: load from file
    const rawalchemy::LUT3D* lutPtr,             // mode B: pre-loaded
    int enableLensCorrection,
    const char* customLensfunDb) {
    // Lens correction (separate pass — spatial operation)
    if (enableLensCorrection) {
        try {
            rawalchemy::LensCorrectionParams lcParams;
            lcParams.enabled = true;
            lcParams.correctDistortion = true;
            lcParams.correctTca = true;
            lcParams.correctVignetting = true;
            lcParams.distance = 1000.0f;
            if (customLensfunDb) lcParams.customDbPath = customLensfunDb;
            rawalchemy::applyLensCorrection(img, meta, lcParams);
        } catch (...) {
            return catchExceptions("lens correction");
        }
    }

    // Exposure metering (subsampled read)
    float gain = 1.0f;
    try {
        std::string mode(metering ? metering : "matrix");
        if (!rawalchemy::isMeteringModeSupported(mode)) {
            setError(std::string("Unsupported metering mode: ") + mode);
            return RA_ERR_INVALID_PARAM;
        }
        gain = rawalchemy::computeAutoGain(img, mode);
        gain *= std::pow(2.0f, evOffset);
    } catch (...) {
        return catchExceptions("exposure");
    }

    // LUT acquisition: mode A loads from a file path, mode B reuses a
    // pre-loaded table. On a path-mode load failure we abort before any
    // grading state is touched.
    rawalchemy::LUT3D loadedLut;
    const rawalchemy::LUT3D* lutToUse = lutPtr;
    if (lutPath) {
        try {
            loadedLut = rawalchemy::loadCubeLUT(std::string(lutPath));
        } catch (...) {
            return catchExceptions("LUT load");
        }
        lutToUse = &loadedLut;
    }

    // Build grading params
    rawalchemy::GradingParams gp;
    gp.gain = gain;
    gp.enableBoost = true;
    gp.saturation = 1.25f;
    gp.contrast = 1.10f;
    gp.pivot = 0.18f;

    if (logSpace) {
        auto it = rawalchemy::LOG_SPACES.find(std::string(logSpace));
        if (it == rawalchemy::LOG_SPACES.end()) {
            setError(std::string("Unsupported log space: ") + logSpace);
            return RA_ERR_LOG_UNSUPPORTED;
        }
        gp.logSpaceInfo = &(it->second);
    }

    // Dual-branch LUT gate (preserves per-mode semantics):
    //   - path mode (runPipeline): set unconditionally after a successful load
    //   - pointer mode (runPipelineWithLUT / runGradingOnly): set only when non-empty
    if (lutPath) {
        gp.lut = lutToUse;
    } else if (lutToUse && !lutToUse->empty()) {
        gp.lut = lutToUse;
    }

    // Fused single-pass grading
    try {
        rawalchemy::applyGradingFused(img, gp);
    } catch (...) {
        return catchExceptions("grading");
    }

    return RA_OK;
}

RaResult runPipeline(rawalchemy::ImageBuffer& img,
                     const rawalchemy::CameraMetadata& meta,
                     const char* logSpace,
                     const char* lutPath,
                     const char* metering,
                     float evOffset,
                     int enableLensCorrection,
                     const char* customLensfunDb) {
    return runPipelineImpl(img, meta, logSpace, metering, evOffset, lutPath, nullptr,
                           enableLensCorrection, customLensfunDb);
}

RaResult runPipelineWithLUT(rawalchemy::ImageBuffer& img,
                            const rawalchemy::CameraMetadata& meta,
                            const char* logSpace,
                            const rawalchemy::LUT3D* lut,
                            const char* metering,
                            float evOffset,
                            int enableLensCorrection,
                            const char* customLensfunDb) {
    return runPipelineImpl(img, meta, logSpace, metering, evOffset, nullptr, lut,
                           enableLensCorrection, customLensfunDb);
}

/// Apply grading pipeline (exposure -> sat/contrast -> log -> LUT) to an image.
/// Does NOT decode or apply lens correction — used for preview re-grading.
RaResult runGradingOnly(
    rawalchemy::ImageBuffer& img,
    const char* logSpace,
    const rawalchemy::LUT3D* lut,
    const char* metering,
    float evOffset)
{
    return runPipelineImpl(img, rawalchemy::CameraMetadata{}, logSpace, metering, evOffset,
                           nullptr, lut, 0, nullptr);
}

} // anonymous namespace

// ----------------------------------------------------------------
//  One-Shot Processing
// ----------------------------------------------------------------

RA_API RaResult RA_CALL raProcessFile(
    const char* inputPath,
    const char* outputPath,
    const char* logSpace,
    const char* lutPath,
    const char* metering,
    float       evOffset,
    int         jpegQuality,
    int         enableLensCorrection,
    const char* customLensfunDb
) {
    if (!inputPath || !outputPath) {
        setError("raProcessFile: null path parameter");
        return RA_ERR_INVALID_PARAM;
    }
    clearError();

    try {
        // Determine output format from extension
        std::string ext = outputPath;
        for (auto& c : ext) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
        bool isJpeg = (ext.size() >= 4 && ext.compare(ext.size()-4, 4, ".jpg") == 0) ||
                      (ext.size() >= 5 && ext.compare(ext.size()-5, 5, ".jpeg") == 0);

        // Create EXIF collector for JPEG output. Scope-guarded so ANY exit
        // (success, runPipeline failure, or an exception thrown by
        // buildExifBlob/writeJpeg/writeTiff16) frees the collector.
        auto exifGuard = std::unique_ptr<rawalchemy::ExifCollector, void(*)(rawalchemy::ExifCollector*)>(
            isJpeg ? rawalchemy::createExifCollector() : nullptr,
            [](rawalchemy::ExifCollector* p) { if (p) rawalchemy::destroyExifCollector(p); });
        rawalchemy::ExifCollector* exifCollector = exifGuard.get();

        // Probe sensor type first (extractMetadata is a metadata-only open_file,
        // cheap — no unpack/dcraw_process). Route non-CFA/Foveon sensors to
        // decodeRaw/dcraw; everything else to the custom pipeline. EXIF is
        // collected exactly once, by the chosen decode path.
        auto meta = rawalchemy::extractMetadata(std::string(inputPath));
        auto img = decodeWithFallback(meta, std::string(inputPath), exifCollector);

        // Run pipeline
        RaResult res = runPipeline(img, meta, logSpace, lutPath, metering,
                                   evOffset,
                                   enableLensCorrection, customLensfunDb);
        if (res != RA_OK) {
            return res;
        }

        bool ok;
        if (isJpeg) {
            std::vector<uint8_t> exifBlob;
            if (exifCollector) {
                exifBlob = rawalchemy::buildExifBlob(*exifCollector, img.width, img.height);
            }
            ok = rawalchemy::writeJpeg(img, std::string(outputPath), jpegQuality, false,
                                        exifBlob.empty() ? nullptr : &exifBlob);
        } else {
            ok = rawalchemy::writeTiff16(img, std::string(outputPath));
        }

        if (!ok) {
            setError("Failed to write output file");
            return RA_ERR_WRITE_FAILED;
        }
        return RA_OK;
    } catch (...) {
        return catchExceptions("raProcessFile");
    }
}

RA_API RaResult RA_CALL raProcessFileWithLUT(
    const char* inputPath,
    const char* outputPath,
    const char* logSpace,
    const float* lutTable,
    int         lutSize,
    const float* lutDomainMin,
    const float* lutDomainMax,
    const char* metering,
    float       evOffset,
    int         jpegQuality,
    int         enableLensCorrection,
    const char* customLensfunDb
) {
    if (!inputPath || !outputPath) {
        setError("raProcessFileWithLUT: null path parameter");
        return RA_ERR_INVALID_PARAM;
    }
    clearError();

    try {
        // Determine output format from extension
        std::string ext = outputPath;
        for (auto& c : ext) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
        bool isJpeg = (ext.size() >= 4 && ext.compare(ext.size()-4, 4, ".jpg") == 0) ||
                      (ext.size() >= 5 && ext.compare(ext.size()-5, 5, ".jpeg") == 0);

        // Create EXIF collector for JPEG output. Scope-guarded so ANY exit
        // (success, runPipeline failure, or an exception thrown by
        // buildExifBlob/writeJpeg/writeTiff16) frees the collector.
        auto exifGuard = std::unique_ptr<rawalchemy::ExifCollector, void(*)(rawalchemy::ExifCollector*)>(
            isJpeg ? rawalchemy::createExifCollector() : nullptr,
            [](rawalchemy::ExifCollector* p) { if (p) rawalchemy::destroyExifCollector(p); });
        rawalchemy::ExifCollector* exifCollector = exifGuard.get();

        // Probe sensor type first; route non-CFA/Foveon to dcraw, else custom.
        auto meta = rawalchemy::extractMetadata(std::string(inputPath));
        auto img = decodeWithFallback(meta, std::string(inputPath), exifCollector);

        rawalchemy::LUT3D lut;
        const rawalchemy::LUT3D* lutPtr = nullptr;
        if (lutTable && lutSize > 0) {
            lut.size = lutSize;
            int totalFloats = lutSize * lutSize * lutSize * 3;
            lut.table.assign(lutTable, lutTable + totalFloats);
            if (lutDomainMin) {
                lut.domainMin[0] = lutDomainMin[0];
                lut.domainMin[1] = lutDomainMin[1];
                lut.domainMin[2] = lutDomainMin[2];
            }
            if (lutDomainMax) {
                lut.domainMax[0] = lutDomainMax[0];
                lut.domainMax[1] = lutDomainMax[1];
                lut.domainMax[2] = lutDomainMax[2];
            }
            lutPtr = &lut;
        }

        RaResult res = runPipelineWithLUT(img, meta, logSpace, lutPtr, metering,
                                   evOffset,
                                   enableLensCorrection, customLensfunDb);
        if (res != RA_OK) {
            return res;
        }

        bool ok;
        if (isJpeg) {
            std::vector<uint8_t> exifBlob;
            if (exifCollector) {
                exifBlob = rawalchemy::buildExifBlob(*exifCollector, img.width, img.height);
            }
            ok = rawalchemy::writeJpeg(img, std::string(outputPath), jpegQuality, false,
                                        exifBlob.empty() ? nullptr : &exifBlob);
        } else {
            ok = rawalchemy::writeTiff16(img, std::string(outputPath));
        }

        if (!ok) {
            setError("Failed to write output file");
            return RA_ERR_WRITE_FAILED;
        }
        return RA_OK;
    } catch (...) {
        return catchExceptions("raProcessFileWithLUT");
    }
}

RA_API RaResult RA_CALL raProcessToBuffer(
    const char* inputPath,
    const char* logSpace,
    const char* lutPath,
    const char* metering,
    float       evOffset,
    int         enableLensCorrection,
    const char* customLensfunDb,
    RaImageBuffer* outBuf
) {
    if (!inputPath || !outBuf) {
        setError("raProcessToBuffer: null parameter");
        return RA_ERR_INVALID_PARAM;
    }
    clearError();

    try {
        // Probe sensor type first; route non-CFA/Foveon to dcraw, else custom.
        // No EXIF collector for the buffer entry point.
        auto meta = rawalchemy::extractMetadata(std::string(inputPath));
        auto img = decodeWithFallback(meta, std::string(inputPath), nullptr);

        // Run pipeline
        RaResult res = runPipeline(img, meta, logSpace, lutPath, metering,
                                   evOffset,
                                   enableLensCorrection, customLensfunDb);
        if (res != RA_OK) return res;

        *outBuf = new RaImageBuffer_{std::move(img)};
        return RA_OK;
    } catch (...) {
        return catchExceptions("raProcessToBuffer");
    }
}

// ----------------------------------------------------------------
//  Utility
// ----------------------------------------------------------------

RA_API const char* RA_CALL raGetLastError(void) {
    return g_lastError.empty() ? nullptr : g_lastError.c_str();
}

RA_API const char* RA_CALL raGetVersion(void) {
    return "0.1.0";
}

// ----------------------------------------------------------------
//  Preview Session
// ----------------------------------------------------------------

RA_API RaResult RA_CALL raBeginPreviewSession(
    const char* inputPath,
    int         enableLensCorrection,
    const char* customLensfunDb,
    int         halfSize,
    int         maxPreviewWidth,
    int         maxPreviewHeight,
    RaPreviewSession* outSession)
{
    if (!inputPath || !outSession) {
        setError("raBeginPreviewSession: null parameter");
        return RA_ERR_INVALID_PARAM;
    }
    clearError();

    try {
        rawalchemy::DecodeParams params;
        if (halfSize) {
            params.halfSize = true;
            // Preview mode: disable expensive noise reduction
            params.medPasses = 0;
            params.fbddNoiserd = 0;
            params.denoiseThreshold = 0.0f;
        }

        auto img = rawalchemy::decodeRaw(std::string(inputPath), params);

        if (enableLensCorrection) {
            try {
                auto meta = rawalchemy::extractMetadata(std::string(inputPath));
                rawalchemy::LensCorrectionParams lcParams;
                lcParams.enabled = true;
                lcParams.correctDistortion = true;
                lcParams.correctTca = true;
                lcParams.correctVignetting = true;
                lcParams.distance = 1000.0f;
                if (customLensfunDb) lcParams.customDbPath = customLensfunDb;
                rawalchemy::applyLensCorrection(img, meta, lcParams);
            } catch (...) {
                return catchExceptions("lens correction");
            }
        }

        // Downscale to target preview dimensions after lens correction
        if (maxPreviewWidth > 0 || maxPreviewHeight > 0) {
            img = rawalchemy::resizeImage(img, maxPreviewWidth, maxPreviewHeight);
        }

        *outSession = new RaPreviewSession_{
            std::move(img),
            {},                                    // previewCache (empty, built lazily)
            std::string(inputPath),
            0,                                     // cacheMaxWidth
            0                                      // cacheMaxHeight
        };
        return RA_OK;
    } catch (...) {
        return catchExceptions("raBeginPreviewSession");
    }
}


RA_API RaResult RA_CALL raApplyPreviewGrading(
    RaPreviewSession session,
    const char*      logSpace,
    const float*     lutTable,
    int              lutSize,
    const float*     lutDomainMin,
    const float*     lutDomainMax,
    const char*      metering,
    float            evOffset,
    int              jpegQuality,
    int              maxWidth,
    int              maxHeight,
    unsigned char**  outBuffer,
    int*             outLen)
{
    if (!session || !outBuffer || !outLen) {
        setError("raApplyPreviewGrading: null parameter");
        return RA_ERR_INVALID_PARAM;
    }
    clearError();
    *outBuffer = nullptr;
    *outLen = 0;

    try {
        auto& source = session->baseImage;

        // Build or reuse preview cache (scaled base image without grading)
        if (session->previewCache.width == 0 ||
            session->cacheMaxWidth != maxWidth ||
            session->cacheMaxHeight != maxHeight) {
            session->previewCache = rawalchemy::resizeImage(source, maxWidth, maxHeight);
            session->cacheMaxWidth = maxWidth;
            session->cacheMaxHeight = maxHeight;
        }

        // Clone from cached preview base — grading operates on already-downscaled image
        auto img = session->previewCache;

        // Build LUT from pre-parsed data
        rawalchemy::LUT3D lut;
        const rawalchemy::LUT3D* lutPtr = nullptr;
        if (lutTable && lutSize > 0) {
            lut.size = lutSize;
            int totalFloats = lutSize * lutSize * lutSize * 3;
            lut.table.assign(lutTable, lutTable + totalFloats);
            if (lutDomainMin) {
                lut.domainMin[0] = lutDomainMin[0];
                lut.domainMin[1] = lutDomainMin[1];
                lut.domainMin[2] = lutDomainMin[2];
            }
            if (lutDomainMax) {
                lut.domainMax[0] = lutDomainMax[0];
                lut.domainMax[1] = lutDomainMax[1];
                lut.domainMax[2] = lutDomainMax[2];
            }
            lutPtr = &lut;
        }

        RaResult res = runGradingOnly(img, logSpace, lutPtr, metering, evOffset);
        if (res != RA_OK) return res;

        // Encode to memory buffer
        std::vector<uint8_t> jpegBytes = rawalchemy::writeJpegToBuffer(img, jpegQuality, false, nullptr);
        if (jpegBytes.empty()) {
            setError("Failed to encode preview JPEG");
            return RA_ERR_WRITE_FAILED;
        }

        size_t len = jpegBytes.size();
        unsigned char* buf = new unsigned char[len];
        std::memcpy(buf, jpegBytes.data(), len);
        *outBuffer = buf;
        *outLen = static_cast<int>(len);
        return RA_OK;
    } catch (...) {
        return catchExceptions("raApplyPreviewGrading");
    }
}

RA_API void RA_CALL raFreePreviewBuffer(unsigned char* buffer) {
    delete[] buffer;
}

RA_API void RA_CALL raEndPreviewSession(RaPreviewSession session) {
    delete session;
}
