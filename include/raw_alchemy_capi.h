#pragma once
/**
 * @file raw_alchemy_capi.h
 * @brief C API for Raw Alchemy shared library.
 *
 * Stable C interface for calling Raw Alchemy from any language via FFI.
 * Uses opaque handles to avoid exposing C++ types across the DLL boundary.
 *
 * Only one-shot processing functions are exposed. Individual pipeline steps
 * are handled internally.
 *
 * Usage on Windows: link against raw_alchemy.dll + raw_alchemy.lib
 * Usage on Android: System.loadLibrary("raw_alchemy")
 * Usage on Linux:   dlopen("libraw_alchemy.so", ...)
 */

#include "raw_alchemy_export.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------------------
 *  Error Codes
 * ---------------------------------------------------------------- */
typedef enum RaResult_ {
    RA_OK                = 0,
    RA_ERR_UNKNOWN       = -1,
    RA_ERR_FILE_NOT_FOUND = -2,
    RA_ERR_DECODE_FAILED  = -3,
    RA_ERR_INVALID_PARAM  = -4,
    RA_ERR_LOG_UNSUPPORTED = -5,
    RA_ERR_LUT_LOAD_FAILED = -6,
    RA_ERR_WRITE_FAILED    = -7,
    RA_ERR_NO_LENS_PROFILE = -8,
    RA_ERR_OUT_OF_MEMORY   = -9,
} RaResult;

/* ----------------------------------------------------------------
 *  Opaque Handles
 * ---------------------------------------------------------------- */
typedef struct RaImageBuffer_*  RaImageBuffer;

/* ----------------------------------------------------------------
 *  Handle Lifecycle
 * ---------------------------------------------------------------- */

/** Destroy an image buffer handle. Safe to pass NULL. */
RA_API void RA_CALL raImageBufferDestroy(RaImageBuffer buf);

/* ----------------------------------------------------------------
 *  Image Buffer Access
 * ---------------------------------------------------------------- */

/** Get image width. Returns 0 if buf is NULL. */
RA_API int RA_CALL raImageGetWidth(RaImageBuffer buf);

/** Get image height. Returns 0 if buf is NULL. */
RA_API int RA_CALL raImageGetHeight(RaImageBuffer buf);

/** Get read-only pointer to float32 pixel data (RGB interleaved, row-major).
 *  Pointer is valid as long as the handle is not destroyed.
 *  Returns NULL if buf is NULL. */
RA_API const float* RA_CALL raImageGetData(RaImageBuffer buf);

/** Get total data size in bytes (width * height * 3 * sizeof(float)). */
RA_API int RA_CALL raImageGetDataSizeBytes(RaImageBuffer buf);

/* ----------------------------------------------------------------
 *  One-Shot Processing
 * ---------------------------------------------------------------- */

/** Process a RAW file through the full pipeline and save to disk.
 *
 *  Pipeline: Decode -> [Lens Correction] -> [Exposure] -> [Sat/Cont Boost]
 *            -> [Log Transform] -> [LUT] -> Save
 *
 *  All intermediate memory is managed internally.
 *
 *  @param inputPath   UTF-8 path to input RAW file.
 *  @param outputPath  UTF-8 path to output file (extension determines format).
 *  @param logSpace    Log space name, or NULL to skip log transform.
 *  @param lutPath     Path to .cube LUT file, or NULL to skip LUT.
 *  @param metering    Metering mode, or NULL for "matrix".
 *  @param manualEv    Manual exposure in stops. Ignored if useAutoExposure != 0.
 *  @param useAutoExposure  If non-zero, use auto metering; else use manualEv.
 *  @param jpegQuality JPEG quality 1-100 (only used for JPEG output).
 *  @param enableLensCorrection  If non-zero, enable lens correction.
 *  @param customLensfunDb      Custom Lensfun DB path, or NULL.
 *  @return RA_OK on success. */
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
);

/** Process a RAW file with a pre-parsed LUT (avoids repeated file I/O).
 *
 *  Same as raProcessFile but accepts LUT data directly as a flat float array
 *  instead of a file path. The table layout matches .cube format:
 *  [size³ × 3] floats, row-major (R changes fastest).
 *
 *  This allows callers to cache parsed LUT data in memory.
 *
 *  @param inputPath   UTF-8 path to input RAW file.
 *  @param outputPath  UTF-8 path to output file (extension determines format).
 *  @param logSpace    Log space name, or NULL to skip log transform.
 *  @param lutTable    Pointer to pre-parsed LUT float data [size³ × 3], or NULL to skip LUT.
 *  @param lutSize     LUT dimension (e.g., 65 for a 65³ grid). Ignored if lutTable is NULL.
 *  @param lutDomainMin  LUT domain minimum [R, G, B]. Pass NULL for default {0,0,0}.
 *  @param lutDomainMax  LUT domain maximum [R, G, B]. Pass NULL for default {1,1,1}.
 *  @param metering    Metering mode, or NULL for "matrix".
 *  @param manualEv    Manual exposure in stops.
 *  @param useAutoExposure  If non-zero, use auto metering.
 *  @param jpegQuality JPEG quality 1-100.
 *  @param enableLensCorrection  If non-zero, enable lens correction.
 *  @param customLensfunDb      Custom Lensfun DB path, or NULL.
 *  @return RA_OK on success. */
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
);

/** Process a RAW file through the full pipeline and return pixel data.
 *
 *  Pipeline: Decode -> [Lens Correction] -> [Exposure] -> [Sat/Cont Boost]
 *            -> [Log Transform] -> [LUT]
 *
 *  @param inputPath   UTF-8 path to input RAW file.
 *  @param logSpace    Log space name, or NULL to skip.
 *  @param lutPath     Path to .cube LUT, or NULL to skip.
 *  @param metering    Metering mode, or NULL for "matrix".
 *  @param manualEv    Manual exposure. Ignored if useAutoExposure != 0.
 *  @param useAutoExposure  If non-zero, use auto metering.
 *  @param enableLensCorrection  If non-zero, enable lens correction.
 *  @param customLensfunDb      Custom Lensfun DB path, or NULL.
 *  @param outBuf      Receives the processed image. Caller must destroy.
 *  @return RA_OK on success. */
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
);

/* ----------------------------------------------------------------
 *  Utility
 * ---------------------------------------------------------------- */

/** Get the last error message (thread-local). Returns NULL if no error.
 *  Pointer is valid until the next Raw Alchemy call on this thread. */
RA_API const char* RA_CALL raGetLastError(void);

/** Get library version string (e.g., "0.1.0"). */
RA_API const char* RA_CALL raGetVersion(void);

#ifdef __cplusplus
}
#endif
