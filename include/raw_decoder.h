#pragma once
/**
 * @file raw_decoder.h
 * @brief RAW file decoding using LibRaw — Step 1: Standardized Decoding.
 *
 * Decodes RAW files from any camera into ProPhoto RGB (Linear) at 16-bit,
 * then provides the data as float32 [0.0, 1.0] for downstream processing.
 *
 * Equivalent to Python rawpy.postprocess() with:
 *   gamma=(1,1), no_auto_bright=True, use_camera_wb=True,
 *   output_bps=16, output_color=ProPhoto, bright=1.0,
 *   highlight_mode=2 (blend), demosaic=AHD/AAHD
 */

#include "common.h"
#include "demosaic.h"
#include <string>

namespace rawalchemy {

/// Forward declaration — see exif_injector.h for full API
struct ExifCollector;

struct ExifCollector;  // Forward declaration — see exif_injector.h

/// Decoder configuration — mirrors rawpy postprocess parameters
struct DecodeParams {
    /// Output color space: 4 = ProPhoto RGB (default, matches Python project)
    int outputColor = 4;

    /// Gamma power: 1.0 = linear (rawpy gamma[0]=1 maps to gamm[0]=1/1=1.0)
    float gammaPower = 1.0f;

    /// Gamma toe slope: 1.0 = linear (rawpy gamma[1]=1)
    float gammaSlope = 1.0f;

    /// Output bit depth: 16 (required for preserving dynamic range)
    int outputBps = 16;

    /// Use camera white balance (True in Python project)
    bool useCameraWb = true;

    /// Disable auto brightness (True in Python project)
    bool noAutoBright = true;

    /// Brightness multiplier: 1.0 = no adjustment
    float bright = 1.0f;

    /// Highlight recovery mode: 2 = Blend (prevents highlight clipping)
    int highlightMode = 2;

    /// Demosaic quality:
    ///   3  = AHD (Adaptive Homogeneity-Directed)
    ///   9  = LMMSE (best moiré suppression for high-ISO)
    ///   11 = DHT (Directional — best detail, sharp edges)
    int demosaicQuality = 11;

    /// Half-size mode for fast preview (optional)
    bool halfSize = false;

    /// Green channel matching: equalize G1/G3 difference (Fixed Pattern Noise)
    bool greenMatching = true;

    /// Median filter passes (post-demosaic, chroma only): 0=off
    int medPasses = 2;

    /// Pre-demosaic noise reduction (FBDD): 0=off, 1=light, 2=full
    /// Operates on raw Bayer data before demosaic — more effective than post-demosaic
    /// denoising for chroma noise and false color artifacts.
    int fbddNoiserd = -1;

    /// Wavelet denoise threshold: <0 = auto (ISO-adaptive), 0 = off, >0 = manual
    float denoiseThreshold = -1.0f;

    /// X-Trans wavelet denoise threshold (darktable domain): <0 = auto
    /// (ISO-adaptive, mirroring the Bayer curve above), 0 = off, >0 = manual
    /// value in ~[0.01, 0.1]. Only affects Fujifilm X-Trans (.RAF) files;
    /// Bayer files use LibRaw's wavelet via `denoiseThreshold` and ignore this.
    float xtransDenoiseThreshold = -1.0f;

    /// Demosaic algorithm selection (default AUTO: Bayer→RCD, X-Trans→Markesteijn).
    /// Set to LIBRAW_FALLBACK to use the original LibRaw path (uses demosaicQuality).
    DemosaicAlgorithm demosaicAlgorithm = DemosaicAlgorithm::AUTO;

    /// Enable NN (x-veon) demosaic. When true, the decoder dispatches to the NN
    /// path instead of the classical RCD/Markesteijn kernels. Ignored for halfSize.
    bool enableNnDemosaic = false;

    // Paths populated by the CAPI layer before calling decodeRaw when enableNnDemosaic:
    std::string nnBayerModelPath;
    std::string nnXtransModelPath;
    std::string nnQnnContextBinaryDir;   // Android only
    std::string nnDirectmlDllPath;       // Windows only
    std::string nnSocModel;  // Android only: QNN "soc_model" numeric string (e.g. "69" for SM8750). "0" = auto-detect.
    std::string nnHtpArch;   // Android only: QNN "htp_arch" ("73"/"75"/...). Empty = infer.
    std::string nnCtxDir;    // Android only: app data dir for QNN context cache. Empty = no cache.
    std::string nnAppVersion;  // Android only: app version (cache key component).
};

/**
 * @brief Decode a RAW file into ProPhoto RGB (Linear) float32 image.
 *
 * @param rawPath  Path to the RAW file (.NEF, .CR3, .ARW, .RW2, etc.)
 * @param params   Decoding parameters (use defaults for Standardized Decoding)
 * @return ImageBuffer  Float32 [0.0, 1.0] image in ProPhoto RGB Linear space
 * @throws std::runtime_error on any error
 */
ImageBuffer decodeRaw(const std::string& rawPath, const DecodeParams& params = DecodeParams{},
                       ExifCollector* exifCollector = nullptr);

/**
 * @brief Extract camera/lens EXIF metadata from a RAW file.
 *
 * Returns a struct with make, model, lens info, focal length, aperture, etc.
 */
struct CameraMetadata {
    std::string cameraMaker;
    std::string cameraModel;
    std::string lensMaker;
    std::string lensModel;
    float       focalLength = 0.0f;
    float       aperture    = 0.0f;
    int         isoSpeed    = 0;
};

CameraMetadata extractMetadata(const std::string& rawPath);

} // namespace rawalchemy
