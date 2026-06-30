/**
 * @file raw_alchemy_capi.cpp
 * @brief C API implementation — one-shot processing via C++ rawalchemy API.
 */

#include "raw_alchemy_capi.h"

#include "nn_session.h"
#include "raw_decoder.h"
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
#include <cstdlib>
#include <string>
#include <stdexcept>

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

RaResult runPipeline(rawalchemy::ImageBuffer& img,
                     const rawalchemy::CameraMetadata& meta,
                     const char* logSpace,
                     const char* lutPath,
                     const char* metering,
                     float evOffset,
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

    // Load LUT from file
    rawalchemy::LUT3D lut;
    if (lutPath) {
        try {
            lut = rawalchemy::loadCubeLUT(std::string(lutPath));
        } catch (...) {
            return catchExceptions("LUT load");
        }
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

    if (lutPath) {
        gp.lut = &lut;
    }

    // Fused single-pass grading
    try {
        rawalchemy::applyGradingFused(img, gp);
    } catch (...) {
        return catchExceptions("grading");
    }

    return RA_OK;
}

RaResult runPipelineWithLUT(rawalchemy::ImageBuffer& img,
                            const rawalchemy::CameraMetadata& meta,
                            const char* logSpace,
                            const rawalchemy::LUT3D* lut,
                            const char* metering,
                            float evOffset,
                            int enableLensCorrection,
                            const char* customLensfunDb) {
    // Lens correction
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

    // Exposure metering
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

    if (lut && !lut->empty()) {
        gp.lut = lut;
    }

    // Fused single-pass grading
    try {
        rawalchemy::applyGradingFused(img, gp);
    } catch (...) {
        return catchExceptions("grading");
    }

    return RA_OK;
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
    // Exposure metering (subsampled)
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

    if (lut && !lut->empty()) {
        gp.lut = lut;
    }

    try {
        rawalchemy::applyGradingFused(img, gp);
    } catch (...) {
        return catchExceptions("grading");
    }

    return RA_OK;
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
    const char* customLensfunDb,
    int         enableNnDemosaic
) {
    if (!inputPath || !outputPath) {
        setError("raProcessFile: null path parameter");
        return RA_ERR_INVALID_PARAM;
    }
    clearError();

    // DEBUG: classical fallback removed. When NN demosaic is requested and fails
    // (init/null-DML, NaN guard, inference error), the exception propagates so the
    // caller sees the failure reason in raGetLastError() instead of a silent retry.
    rawalchemy::ExifCollector* exifCollector = nullptr;
    try {
        // Determine output format from extension
        std::string ext = outputPath;
        for (auto& c : ext) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
        bool isJpeg = (ext.size() >= 4 && ext.compare(ext.size()-4, 4, ".jpg") == 0) ||
                      (ext.size() >= 5 && ext.compare(ext.size()-5, 5, ".jpeg") == 0);

        // Create EXIF collector for JPEG output
        exifCollector = isJpeg ? rawalchemy::createExifCollector() : nullptr;

        // Decode (with optional EXIF collection)
        rawalchemy::DecodeParams params;
        params.enableNnDemosaic = (enableNnDemosaic != 0);
        // NN model paths: populated by caller via env or set globally (Task 7).
        // For now, read from env vars RA_NN_BAYER_MODEL / RA_NN_XTRANS_MODEL.
        if (const char* p = std::getenv("RA_NN_BAYER_MODEL")) params.nnBayerModelPath = p;
        if (const char* p = std::getenv("RA_NN_XTRANS_MODEL")) params.nnXtransModelPath = p;
        // DirectML.dll path (Windows). Set by the Rust host to the extraction
        // dir so nn_session.cpp's SetDllDirectoryA steers ORT's DML EP to our
        // app-local DirectML.dll (defense-in-depth on top of the host preload).
        if (const char* p = std::getenv("RA_NN_DIRECTML_DLL")) params.nnDirectmlDllPath = p;
        if (const char* p = std::getenv("RA_NN_SOC_MODEL")) params.nnSocModel = p;
        if (const char* p = std::getenv("RA_NN_HTP_ARCH")) params.nnHtpArch = p;
        if (const char* p = std::getenv("RA_NN_CTX_DIR")) params.nnCtxDir = p;
        if (const char* p = std::getenv("RA_NN_APP_VERSION")) params.nnAppVersion = p;
        auto img = rawalchemy::decodeRaw(std::string(inputPath), params, exifCollector);

        // Metadata (for lens correction)
        auto meta = rawalchemy::extractMetadata(std::string(inputPath));

        // Run pipeline
        RaResult res = runPipeline(img, meta, logSpace, lutPath, metering,
                                   evOffset,
                                   enableLensCorrection, customLensfunDb);
        if (res != RA_OK) {
            if (exifCollector) rawalchemy::destroyExifCollector(exifCollector);
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

        if (exifCollector) rawalchemy::destroyExifCollector(exifCollector);

        if (!ok) {
            setError("Failed to write output file");
            return RA_ERR_WRITE_FAILED;
        }
        return RA_OK;
    } catch (...) {
        if (exifCollector) rawalchemy::destroyExifCollector(exifCollector);
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
    const char* customLensfunDb,
    int         enableNnDemosaic
) {
    if (!inputPath || !outputPath) {
        setError("raProcessFileWithLUT: null path parameter");
        return RA_ERR_INVALID_PARAM;
    }
    clearError();

    // DEBUG: classical fallback removed — see raProcessFile.
    rawalchemy::ExifCollector* exifCollector = nullptr;
    try {
        // Determine output format from extension
        std::string ext = outputPath;
        for (auto& c : ext) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
        bool isJpeg = (ext.size() >= 4 && ext.compare(ext.size()-4, 4, ".jpg") == 0) ||
                      (ext.size() >= 5 && ext.compare(ext.size()-5, 5, ".jpeg") == 0);

        // Create EXIF collector for JPEG output
        exifCollector = isJpeg ? rawalchemy::createExifCollector() : nullptr;

        rawalchemy::DecodeParams params;
        params.enableNnDemosaic = (enableNnDemosaic != 0);
        // NN model paths: populated by caller via env or set globally (Task 7).
        // For now, read from env vars RA_NN_BAYER_MODEL / RA_NN_XTRANS_MODEL.
        if (const char* p = std::getenv("RA_NN_BAYER_MODEL")) params.nnBayerModelPath = p;
        if (const char* p = std::getenv("RA_NN_XTRANS_MODEL")) params.nnXtransModelPath = p;
        // DirectML.dll path (Windows). Set by the Rust host to the extraction
        // dir so nn_session.cpp's SetDllDirectoryA steers ORT's DML EP to our
        // app-local DirectML.dll (defense-in-depth on top of the host preload).
        if (const char* p = std::getenv("RA_NN_DIRECTML_DLL")) params.nnDirectmlDllPath = p;
        if (const char* p = std::getenv("RA_NN_SOC_MODEL")) params.nnSocModel = p;
        if (const char* p = std::getenv("RA_NN_HTP_ARCH")) params.nnHtpArch = p;
        if (const char* p = std::getenv("RA_NN_CTX_DIR")) params.nnCtxDir = p;
        if (const char* p = std::getenv("RA_NN_APP_VERSION")) params.nnAppVersion = p;
        auto img = rawalchemy::decodeRaw(std::string(inputPath), params, exifCollector);
        auto meta = rawalchemy::extractMetadata(std::string(inputPath));

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
            if (exifCollector) rawalchemy::destroyExifCollector(exifCollector);
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

        if (exifCollector) rawalchemy::destroyExifCollector(exifCollector);

        if (!ok) {
            setError("Failed to write output file");
            return RA_ERR_WRITE_FAILED;
        }
        return RA_OK;
    } catch (...) {
        if (exifCollector) rawalchemy::destroyExifCollector(exifCollector);
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
    RaImageBuffer* outBuf,
    int         enableNnDemosaic
) {
    if (!inputPath || !outBuf) {
        setError("raProcessToBuffer: null parameter");
        return RA_ERR_INVALID_PARAM;
    }
    clearError();

    // DEBUG: classical fallback removed — see raProcessFile.
    try {
        // Decode
        rawalchemy::DecodeParams params;
        params.enableNnDemosaic = (enableNnDemosaic != 0);
        // NN model paths: populated by caller via env or set globally (Task 7).
        // For now, read from env vars RA_NN_BAYER_MODEL / RA_NN_XTRANS_MODEL.
        if (const char* p = std::getenv("RA_NN_BAYER_MODEL")) params.nnBayerModelPath = p;
        if (const char* p = std::getenv("RA_NN_XTRANS_MODEL")) params.nnXtransModelPath = p;
        // DirectML.dll path (Windows). Set by the Rust host to the extraction
        // dir so nn_session.cpp's SetDllDirectoryA steers ORT's DML EP to our
        // app-local DirectML.dll (defense-in-depth on top of the host preload).
        if (const char* p = std::getenv("RA_NN_DIRECTML_DLL")) params.nnDirectmlDllPath = p;
        if (const char* p = std::getenv("RA_NN_SOC_MODEL")) params.nnSocModel = p;
        if (const char* p = std::getenv("RA_NN_HTP_ARCH")) params.nnHtpArch = p;
        if (const char* p = std::getenv("RA_NN_CTX_DIR")) params.nnCtxDir = p;
        if (const char* p = std::getenv("RA_NN_APP_VERSION")) params.nnAppVersion = p;
        auto img = rawalchemy::decodeRaw(std::string(inputPath), params);

        // Metadata (for lens correction)
        auto meta = rawalchemy::extractMetadata(std::string(inputPath));

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
//  NN session warmup
// ----------------------------------------------------------------

RA_API void RA_CALL raWarmupNnSession(void) {
    // Defensive guard: init() shouldn't throw after the thread-safety fix, but
    // this runs on a fire-and-forget background thread at app launch — never let
    // an exception escape into the async runtime.
    try {
        rawalchemy::NnSessionConfig cfg;
        if (const char* p = std::getenv("RA_NN_BAYER_MODEL"))  cfg.bayerModelPath  = p;
        if (const char* p = std::getenv("RA_NN_XTRANS_MODEL")) cfg.xtransModelPath = p;
#ifdef _WIN32
        if (const char* p = std::getenv("RA_NN_DIRECTML_DLL")) cfg.directmlDllPath = p;
#elif defined(__ANDROID__)
        if (const char* p = std::getenv("RA_NN_SOC_MODEL"))   cfg.socModel   = (p && *p) ? std::string(p) : std::string("0");
        if (const char* p = std::getenv("RA_NN_HTP_ARCH"))    cfg.htpArch    = (p ? std::string(p) : std::string());
        if (const char* p = std::getenv("RA_NN_CTX_DIR"))     cfg.ctxDir     = (p ? std::string(p) : std::string());
        if (const char* p = std::getenv("RA_NN_APP_VERSION")) cfg.appVersion = (p ? std::string(p) : std::string());
#endif
        // Best-effort: init() returns false on failure (never throws). The edit
        // path will re-attempt via decodeRawNn if this background call didn't
        // succeed. If it did succeed, the edit path's init() is a lock-free no-op.
        rawalchemy::NnDemosaicSession::instance().init(cfg);
    } catch (...) {
        // Swallow — background warmup must never crash the app.
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
