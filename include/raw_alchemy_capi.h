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

#include <stddef.h>  // size_t

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
    RA_ERR_NN_NOT_INITIALIZED  = -10,
    RA_ERR_NN_NAN_OUTPUT       = -11,
    RA_ERR_NN_INFERENCE_FAILED = -12,
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
 *  @param evOffset Exposure offset in stops, applied on top of auto-metered exposure.
 *  @param jpegQuality JPEG quality 1-100 (only used for JPEG output).
 *  @param enableLensCorrection  If non-zero, enable lens correction.
 *  @param customLensfunDb      Custom Lensfun DB path, or NULL.
 *  @param enableNnDemosaic  0 = classical demosaic (RCD/Markesteijn). Non-zero = NN demosaic
 *                           (x-veon). Ignored when halfSize != 0 (preview path). If NN is not
 *                           initialized or fails, returns RA_ERR_NN_* (caller decides retry).
 *  @return RA_OK on success. */
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
 *  @param evOffset Exposure offset in stops, applied on top of auto-metered exposure.
 *  @param jpegQuality JPEG quality 1-100.
 *  @param enableLensCorrection  If non-zero, enable lens correction.
 *  @param customLensfunDb      Custom Lensfun DB path, or NULL.
 *  @param enableNnDemosaic  0 = classical demosaic (RCD/Markesteijn). Non-zero = NN demosaic
 *                           (x-veon). Ignored when halfSize != 0 (preview path). If NN is not
 *                           initialized or fails, returns RA_ERR_NN_* (caller decides retry).
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
    float       evOffset,
    int         jpegQuality,
    int         enableLensCorrection,
    const char* customLensfunDb,
    int         enableNnDemosaic
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
 *  @param evOffset Exposure offset in stops, applied on top of auto-metered exposure.
 *  @param enableLensCorrection  If non-zero, enable lens correction.
 *  @param customLensfunDb      Custom Lensfun DB path, or NULL.
 *  @param enableNnDemosaic  0 = classical demosaic (RCD/Markesteijn). Non-zero = NN demosaic
 *                           (x-veon). Ignored when halfSize != 0 (preview path). If NN is not
 *                           initialized or fails, returns RA_ERR_NN_* (caller decides retry).
 *  @param outBuf      Receives the processed image. Caller must destroy.
 *  @return RA_OK on success. */
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
);

/** Eagerly initialize the NN demosaic session in the background. Reads the
 *  config set via ra_set_nn_config and calls NnDemosaicSession::init().
 *  Best-effort: any failure is logged and swallowed (init() itself never throws
 *  after the thread-safety fix); the edit path will re-attempt if this didn't
 *  succeed. Intended to be called from a background thread at app launch so the
 *  ~2s QNN graph compile overlaps with the user browsing photos. */
RA_API void RA_CALL raWarmupNnSession(void);

/** Returns true iff the NN demosaic session successfully initialized (NPU
 *  engaged). Intended for the Rust fallback router to distinguish an NN-
 *  structural-unavailability failure (init failed → latch classical for the
 *  session) from a per-file NN error on a ready session (fall back for this
 *  file only, no latch). */
RA_API bool RA_CALL raIsNnReady(void);

/* ----------------------------------------------------------------
 *  NN Runtime Configuration (explicit C-ABI transport)
 *
 *  Replaces the legacy RA_NN_* environment-variable transport. Rust's
 *  std::env::set_var calls SetEnvironmentVariableW, which is invisible to
 *  MSVC's std::getenv (CRT/Win32 environment desync), so on Windows all NN
 *  config read as NULL. These explicit setters carry the config through a
 *  stable C struct instead, with the implementation deep-copying every
 *  non-NULL string so the caller may free immediately after the call.
 * ---------------------------------------------------------------- */

/* NN runtime config. All fields are UTF-8 strings and individually nullable
 * (NULL = unset / N/A for this platform). Field semantics:
 *   directml_dll_path — DirectML.dll path; parent dir → SetDllDirectoryA (Windows only)
 *   soc_model         — QNN numeric SoC model, e.g. "43" for SM8550 (Android only)
 *   htp_arch          — QNN Hexagon arch, e.g. "73" (Android only)
 *   ctx_dir           — QNN context-cache dir (Android only)
 *   app_version       — embedded in the context-cache filename (Android)
 *
 * Model WEIGHTS are NOT carried here — they are in-memory ONNX byte buffers
 * supplied via ra_set_nn_model() (Option D: the host embeds gzip-compressed
 * models in its binary, decompresses them, and hands the bytes to the C++ core,
 * which feeds them to ORT's in-memory Session ctor — no disk file/path).
 */
typedef struct RaNnConfig {
    const char* directml_dll_path;
    const char* soc_model;
    const char* htp_arch;
    const char* ctx_dir;
    const char* app_version;
} RaNnConfig;

/** Set the NN runtime config. Deep-copies every non-NULL field under a mutex
 *  (last-call-wins). Safe to pass NULL (no-op, returns RA_OK). Caller may free
 *  the strings immediately after this returns.
 *  @return RA_OK, or RA_ERR_OUT_OF_MEMORY if a string copy throws bad_alloc. */
RA_API RaResult RA_CALL ra_set_nn_config(const RaNnConfig* cfg);

/* Supply an NN model's ONNX weights as an in-memory byte buffer. `kind` selects
 * the model: 0 = bayer, 1 = xtrans. `data`/`len` are deep-copied under the NN
 * config mutex, so the caller may free `data` immediately after this returns.
 * Replaces the former bayer_model_path / xtrans_model_path config fields
 * (Option D: ORT loads weights from memory, no on-disk model file). Pass len=0
 * to clear a model (sets it to "absent"). Best-effort allocation errors return
 * RA_ERR_OUT_OF_MEMORY; otherwise RA_OK. */
RA_API RaResult RA_CALL ra_set_nn_model(int kind, const void* data, size_t len);

/** Redirect C++ NN diagnostics (nnlog::info) to `path` (opened in append mode).
 *  Pass NULL to revert to stderr. Deep-copies the path under a mutex. The
 *  nnlog fallback is: if the path is unset or the file can't be opened, write
 *  to stderr instead. */
RA_API void RA_CALL ra_set_log_file(const char* path);

/* ----------------------------------------------------------------
 *  Preview Session — two-phase preview pipeline
 *
 *  Decodes RAW + applies lens correction once, then allows fast
 *  re-grading with different LUT/exposure parameters.
 * ---------------------------------------------------------------- */
typedef struct RaPreviewSession_* RaPreviewSession;

/** Decode a RAW file and apply lens correction, caching the result for
 *  fast re-grading.  Call raEndPreviewSession when done.
 *
 *  @param inputPath             UTF-8 path to input RAW file.
 *  @param enableLensCorrection  If non-zero, apply lens correction.
 *  @param customLensfunDb       Custom Lensfun DB path, or NULL.
 *  @param halfSize              If non-zero, use LibRaw half-size demosaic
 *                               (faster, lower resolution).
 *  @param maxPreviewWidth       Max width after lens correction (0 = no resize).
 *  @param maxPreviewHeight      Max height after lens correction (0 = no resize).
 *  @param outSession            Receives the session handle.
 *  @return RA_OK on success. */
RA_API RaResult RA_CALL raBeginPreviewSession(
    const char* inputPath,
    int         enableLensCorrection,
    const char* customLensfunDb,
    int         halfSize,
    int         maxPreviewWidth,
    int         maxPreviewHeight,
    RaPreviewSession* outSession
);

/** Apply grading parameters to the session's cached decoded image.
 *
 *  The session's internal data is NOT modified — safe to call repeatedly
 *  with different parameters.  Internally clones the cached buffer, applies
 *  the full grading pipeline, resizes to fit maxWidth×maxHeight, and JPEG-encodes
 *  to memory.
 *
 *  Buffer is allocated internally and must be freed by the caller with raFreePreviewBuffer.
 *
 *  Pipeline on cloned data:
 *    Exposure -> Sat/Contrast Boost -> Log Transform -> LUT -> Resize -> JPEG encode
 *
 *  @param session         Active preview session.
 *  @param logSpace        Log space name, or NULL to skip.
 *  @param lutTable        Pre-parsed LUT float data [size^3 x 3], or NULL.
 *  @param lutSize         LUT dimension. Ignored if lutTable is NULL.
 *  @param lutDomainMin    LUT domain min [R,G,B], or NULL for {0,0,0}.
 *  @param lutDomainMax    LUT domain max [R,G,B], or NULL for {1,1,1}.
 *  @param metering        Metering mode, or NULL for "matrix".
 *  @param evOffset        Exposure offset in stops.
 *  @param jpegQuality     JPEG quality 1-100.
 *  @param maxWidth        Max output width (0 = keep original resolution).
 *  @param maxHeight       Max output height (0 = keep original resolution).
 *  @param outBuffer       Receives JPEG data (caller must free via raFreePreviewBuffer).
 *  @param outLen          Receives JPEG data length.
 *  @return RA_OK on success. */
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
    int*             outLen
);

/** Free a buffer allocated by raApplyPreviewGrading. Safe to pass NULL. */
RA_API void RA_CALL raFreePreviewBuffer(unsigned char* buffer);

/** End a preview session and release all cached resources.
 *  Safe to pass NULL (no-op). */
RA_API void RA_CALL raEndPreviewSession(RaPreviewSession session);

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
