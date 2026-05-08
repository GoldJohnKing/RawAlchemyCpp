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

#include "raw_alchemy/common.h"
#include <string>

namespace rawalchemy {

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
    ///   3  = AHD (Adaptive Homogeneity-Directed, widely used)
    ///   11 = AAHD (Adaptive AHD, slightly better, needs LibRaw >= 0.17)
    int demosaicQuality = 3;

    /// Half-size mode for fast preview (optional)
    bool halfSize = false;
};

/**
 * @brief Decode a RAW file into ProPhoto RGB (Linear) float32 image.
 *
 * @param rawPath  Path to the RAW file (.NEF, .CR3, .ARW, .RW2, etc.)
 * @param params   Decoding parameters (use defaults for Standardized Decoding)
 * @return ImageBuffer  Float32 [0.0, 1.0] image in ProPhoto RGB Linear space
 * @throws std::runtime_error on any error
 */
ImageBuffer decodeRaw(const std::string& rawPath, const DecodeParams& params = DecodeParams{});

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
