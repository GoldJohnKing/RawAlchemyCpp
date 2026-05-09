/**
 * @file raw_alchemy_capi.cpp
 * @brief C API implementation — one-shot processing via C++ rawalchemy API.
 */

#include "raw_alchemy_capi.h"

#include "raw_decoder.h"
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

#include <cstring>
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
                     float manualEv,
                     int useAutoExposure,
                     int enableLensCorrection,
                     const char* customLensfunDb) {
    #if defined(__aarch64__)
    rawalchemy::HalfImageBuffer imgF16;
    bool usingF16 = false;
    #endif

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
            // Ignore return — lens not found is not an error
            rawalchemy::applyLensCorrection(img, meta, lcParams);
        } catch (...) {
            return catchExceptions("lens correction");
        }
    }

    // Exposure
    try {
        if (useAutoExposure) {
            std::string mode(metering ? metering : "matrix");
            if (!rawalchemy::isMeteringModeSupported(mode)) {
                setError(std::string("Unsupported metering mode: ") + mode);
                return RA_ERR_INVALID_PARAM;
            }
            float gain = rawalchemy::computeAutoGain(img, mode);
            img.applyGain(gain);
        } else {
            img.applyGain(std::pow(2.0f, manualEv));
        }
    } catch (...) {
        return catchExceptions("exposure");
    }

    // Saturation / Contrast boost
    try {
        rawalchemy::applySaturationContrast(img, 1.25f, 1.10f);
    } catch (...) {
        return catchExceptions("saturation/contrast");
    }

    // Log transform
    if (logSpace) {
        try {
            std::string space(logSpace);
            if (!rawalchemy::isLogSpaceSupported(space)) {
                setError(std::string("Unsupported log space: ") + logSpace);
                return RA_ERR_LOG_UNSUPPORTED;
            }
            // Gamut transform always in float32
            rawalchemy::applyGamutTransform(img, space);

            #if defined(__aarch64__)
            imgF16 = rawalchemy::convertToF16(img);
            rawalchemy::applyLogEncodingF16(imgF16, space);
            usingF16 = true;
            #else
            rawalchemy::applyLogEncoding(img, space);
            #endif
        } catch (...) {
            return catchExceptions("log transform");
        }
    }

    // LUT
    if (lutPath) {
        try {
            auto lut = rawalchemy::loadCubeLUT(std::string(lutPath));
            #if defined(__aarch64__)
            if (usingF16) {
                rawalchemy::applyLUT3DF16(imgF16, lut);
            } else {
                rawalchemy::applyLUT3D(img, lut);
            }
            #else
            rawalchemy::applyLUT3D(img, lut);
            #endif
        } catch (...) {
            return catchExceptions("LUT");
        }
    }

    // Convert back from float16 before output
    #if defined(__aarch64__)
    if (usingF16) {
        img = rawalchemy::convertToF32(imgF16);
    }
    #endif

    return RA_OK;
}

RaResult runPipelineWithLUT(rawalchemy::ImageBuffer& img,
                            const rawalchemy::CameraMetadata& meta,
                            const char* logSpace,
                            const rawalchemy::LUT3D* lut,
                            const char* metering,
                            float manualEv,
                            int useAutoExposure,
                            int enableLensCorrection,
                            const char* customLensfunDb) {
    #if defined(__aarch64__)
    rawalchemy::HalfImageBuffer imgF16;
    bool usingF16 = false;
    #endif

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

    // Exposure
    try {
        if (useAutoExposure) {
            std::string mode(metering ? metering : "matrix");
            if (!rawalchemy::isMeteringModeSupported(mode)) {
                setError(std::string("Unsupported metering mode: ") + mode);
                return RA_ERR_INVALID_PARAM;
            }
            float gain = rawalchemy::computeAutoGain(img, mode);
            img.applyGain(gain);
        } else {
            img.applyGain(std::pow(2.0f, manualEv));
        }
    } catch (...) {
        return catchExceptions("exposure");
    }

    // Saturation / Contrast boost
    try {
        rawalchemy::applySaturationContrast(img, 1.25f, 1.10f);
    } catch (...) {
        return catchExceptions("saturation/contrast");
    }

    // Log transform
    if (logSpace) {
        try {
            std::string space(logSpace);
            if (!rawalchemy::isLogSpaceSupported(space)) {
                setError(std::string("Unsupported log space: ") + logSpace);
                return RA_ERR_LOG_UNSUPPORTED;
            }
            // Gamut transform always in float32
            rawalchemy::applyGamutTransform(img, space);

            #if defined(__aarch64__)
            imgF16 = rawalchemy::convertToF16(img);
            rawalchemy::applyLogEncodingF16(imgF16, space);
            usingF16 = true;
            #else
            rawalchemy::applyLogEncoding(img, space);
            #endif
        } catch (...) {
            return catchExceptions("log transform");
        }
    }

    // LUT — use pre-parsed data directly
    if (lut && !lut->empty()) {
        try {
            #if defined(__aarch64__)
            if (usingF16) {
                rawalchemy::applyLUT3DF16(imgF16, *lut);
            } else {
                rawalchemy::applyLUT3D(img, *lut);
            }
            #else
            rawalchemy::applyLUT3D(img, *lut);
            #endif
        } catch (...) {
            return catchExceptions("LUT");
        }
    }

    // Convert back from float16 before output
    #if defined(__aarch64__)
    if (usingF16) {
        img = rawalchemy::convertToF32(imgF16);
    }
    #endif

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
    float       manualEv,
    int         useAutoExposure,
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

        // Create EXIF collector for JPEG output
        rawalchemy::ExifCollector* exifCollector = isJpeg
            ? rawalchemy::createExifCollector() : nullptr;

        // Decode (with optional EXIF collection)
        auto img = rawalchemy::decodeRaw(std::string(inputPath), {}, exifCollector);

        // Metadata (for lens correction)
        auto meta = rawalchemy::extractMetadata(std::string(inputPath));

        // Run pipeline
        RaResult res = runPipeline(img, meta, logSpace, lutPath, metering,
                                   manualEv, useAutoExposure,
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
    float       manualEv,
    int         useAutoExposure,
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

        // Create EXIF collector for JPEG output
        rawalchemy::ExifCollector* exifCollector = isJpeg
            ? rawalchemy::createExifCollector() : nullptr;

        auto img = rawalchemy::decodeRaw(std::string(inputPath), {}, exifCollector);
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
                                   manualEv, useAutoExposure,
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
        return catchExceptions("raProcessFileWithLUT");
    }
}

RA_API RaResult RA_CALL raProcessToBuffer(
    const char* inputPath,
    const char* logSpace,
    const char* lutPath,
    const char* metering,
    float       manualEv,
    int         useAutoExposure,
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
        // Decode
        auto img = rawalchemy::decodeRaw(std::string(inputPath));

        // Metadata (for lens correction)
        auto meta = rawalchemy::extractMetadata(std::string(inputPath));

        // Run pipeline
        RaResult res = runPipeline(img, meta, logSpace, lutPath, metering,
                                   manualEv, useAutoExposure,
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
