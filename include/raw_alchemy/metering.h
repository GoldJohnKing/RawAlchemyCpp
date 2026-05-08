#pragma once
/**
 * @file metering.h
 * @brief Auto exposure metering strategies.
 *
 * Direct C++ port of the Python project's metering.py.
 * 5 strategies: average, center-weighted, highlight-safe, hybrid, matrix.
 *
 * All strategies operate on a subsampled view (~1024px wide) for speed,
 * compute luminance using ProPhoto RGB coefficients, and return a gain
 * value that is applied to the full image.
 */

#include "raw_alchemy/common.h"
#include <string>
#include <vector>

namespace rawalchemy {

/// ProPhoto RGB → Luminance coefficients (Y-row of RGB-to-XYZ matrix)
constexpr float PROPHOTO_LUMA_R = 0.2880747f;
constexpr float PROPHOTO_LUMA_G = 0.7118632f;
constexpr float PROPHOTO_LUMA_B = 0.0000622f;

/// Subsample the image to ~targetSize on the long edge
struct SubsampledView {
    const float* data;
    int origWidth;
    int channels;
    int step;
    int width, height;

    SubsampledView(const ImageBuffer& img, int targetSize = 1024)
        : data(img.ptr()), origWidth(img.width), channels(img.channels),
          step(std::max(1, std::max(img.width, img.height) / targetSize)),
          width((img.width + step - 1) / step),
          height((img.height + step - 1) / step)
    {}

    float get(int r, int c, int ch) const {
        return data[(static_cast<size_t>(r * step) * origWidth + c * step) * channels + ch];
    }
};

/**
 * @brief Compute auto-exposure gain using the specified metering mode.
 *
 * @param img           Image buffer (NOT modified — gain is returned)
 * @param mode          Metering mode name
 * @param targetGray    Target middle-gray value (default 0.18)
 * @return float        Gain value to apply via img.applyGain(gain)
 */
float computeAutoGain(const ImageBuffer& img, const std::string& mode, float targetGray = 0.18f);

/// Get list of supported metering mode names
std::vector<std::string> getSupportedMeteringModes();

/// Check if a metering mode name is supported
bool isMeteringModeSupported(const std::string& mode);

} // namespace rawalchemy
