/**
 * @file lens_correction.cpp
 * @brief Lens correction implementation using the Lensfun C++ API.
 *
 * Direct C++ port of the Python project's lensfun_wrapper.py.
 * Uses Lensfun's C++ classes (lfDatabase, lfModifier) directly,
 * rather than ctypes bindings.
 *
 * Key differences from Python implementation:
 *   - Uses Lensfun C++ API instead of ctypes
 *   - Implements bicubic interpolation inline (no scipy dependency)
 *   - OpenMP parallelization for coordinate remapping
 */

#include "lens_correction.h"

#ifdef RA_LENSFUN_ENABLED
#include <lensfun/lensfun.h>
#endif

#include <cstdio>
#include <cstring>
#include <cmath>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>

namespace rawalchemy {

// ============================================================================
// When Lensfun is not available, provide a stub
// ============================================================================

#ifndef RA_LENSFUN_ENABLED

bool applyLensCorrection(ImageBuffer& /*img*/,
                         const CameraMetadata& /*meta*/,
                         const LensCorrectionParams& /*params*/) {
    printf("  [Lensfun] Not compiled with Lensfun support. Skipping.\n");
    return false;
}

#else // RA_LENSFUN_ENABLED

// ============================================================================
// Bicubic interpolation helpers
// ============================================================================

/**
 * @brief Catmull-Rom cubic interpolation kernel.
 *
 * This is an interpolating kernel (passes through data points exactly),
 * equivalent to the cubic convolution with a=-0.5.
 * Produces results visually identical to scipy.ndimage.map_coordinates(order=3).
 */
static inline float cubicKernel(float t) {
    t = std::abs(t);
    if (t <= 1.0f) {
        // 1.5*t^3 - 2.5*t^2 + 1
        return t * t * (1.5f * t - 2.5f) + 1.0f;
    }
    if (t <= 2.0f) {
        // -0.5*t^3 + 2.5*t^2 - 4*t + 2
        return t * (t * (-0.5f * t + 2.5f) - 4.0f) + 2.0f;
    }
    return 0.0f;
}

/**
 * @brief Bicubic interpolation of a single channel at fractional coordinates.
 *
 * @param data     Image data (float32, RGB interleaved)
 * @param width    Image width
 * @param height   Image height
 * @param channels Channel count (always 3)
 * @param x        X coordinate (column)
 * @param y        Y coordinate (row)
 * @param ch       Channel index (0=R, 1=G, 2=B)
 * @return Interpolated value (0.0 for out-of-bounds, matching scipy mode='constant')
 */
static inline float bicubicSample(const float* data, int width, int height,
                                  int channels, float x, float y, int ch) {
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));

    float fx = x - static_cast<float>(x0);
    float fy = y - static_cast<float>(y0);

    float result = 0.0f;

    for (int j = -1; j <= 2; ++j) {
        int sy = y0 + j;
        if (sy < 0 || sy >= height) continue;

        float wy = cubicKernel(fy - static_cast<float>(j));

        for (int i = -1; i <= 2; ++i) {
            int sx = x0 + i;
            if (sx < 0 || sx >= width) continue;

            float wx = cubicKernel(fx - static_cast<float>(i));
            float w = wx * wy;

            result += w * data[(static_cast<size_t>(sy) * width + sx) * channels + ch];
        }
    }

    return result;
}

// ============================================================================
// RAII helpers for Lensfun objects
// ============================================================================

struct LensfunDbDeleter {
    void operator()(lfDatabase* db) const {
        if (db) lf_db_destroy(db);
    }
};

struct LensfunModifierDeleter {
    void operator()(lfModifier* mod) const {
        if (mod) lf_modifier_destroy(mod);
    }
};

// ============================================================================
// Main implementation
// ============================================================================

bool applyLensCorrection(ImageBuffer& img,
                         const CameraMetadata& meta,
                         const LensCorrectionParams& params) {
    if (!params.enabled) return false;

    // --- Validate metadata ---
    if (meta.cameraModel.empty() && meta.lensModel.empty()) {
        printf("  [Lensfun] No camera/lens metadata. Skipping correction.\n");
        return false;
    }

    // --- Create database ---
    std::unique_ptr<lfDatabase, LensfunDbDeleter> db(lf_db_create());
    if (!db) {
        printf("  [Lensfun] Failed to create database.\n");
        return false;
    }

    // Load database: try custom path first, then system default
    lfError err;
    if (!params.customDbPath.empty()) {
        // Load system default first, then overlay custom database
        err = db->Load();
        if (err != LF_NO_ERROR) {
            printf("  [Lensfun] Warning: system database load failed (code %d).\n", err);
        }

        // Load custom database from file
        FILE* f = fopen(params.customDbPath.c_str(), "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            if (sz > 0) {
                std::vector<char> xmlData(static_cast<size_t>(sz));
                if (fread(xmlData.data(), 1, static_cast<size_t>(sz), f) == static_cast<size_t>(sz)) {
                    err = db->Load(xmlData.data(), static_cast<size_t>(sz));
                    if (err != LF_NO_ERROR) {
                        printf("  [Lensfun] Warning: custom database load failed (code %d).\n", err);
                    } else {
                        printf("  [Lensfun] Loaded custom database: %s\n", params.customDbPath.c_str());
                    }
                }
            }
            fclose(f);
        } else {
            // Maybe it's a directory path
            err = db->Load(params.customDbPath.c_str());
            if (err != LF_NO_ERROR) {
                printf("  [Lensfun] Warning: custom database path failed (code %d): %s\n",
                       err, params.customDbPath.c_str());
            } else {
                printf("  [Lensfun] Loaded custom database: %s\n", params.customDbPath.c_str());
            }
        }
    } else {
        // Load from system default paths
        err = db->Load();
        if (err != LF_NO_ERROR) {
            printf("  [Lensfun] Failed to load database (code %d). "
                   "Install lensfun data or use --custom-lensfun-db.\n", err);
            return false;
        }
        printf("  [Lensfun] Loaded system database.\n");
    }

    // --- Find camera ---
    const lfCamera* camera = nullptr;
    const lfCamera** cameras = db->FindCamerasExt(
        meta.cameraMaker.empty() ? nullptr : meta.cameraMaker.c_str(),
        meta.cameraModel.c_str(), 0);

    if (cameras && cameras[0]) {
        camera = cameras[0];
        printf("  [Lensfun] Camera found: %s %s\n",
               lf_mlstr_get(camera->Maker), lf_mlstr_get(camera->Model));
    } else {
        printf("  [Lensfun] Camera not found: %s %s\n",
               meta.cameraMaker.c_str(), meta.cameraModel.c_str());
    }
    lf_free(cameras);

    // --- Find lens ---
    const lfLens* lens = nullptr;
    const lfLens** lenses = db->FindLenses(
        camera,
        meta.lensMaker.empty() ? nullptr : meta.lensMaker.c_str(),
        meta.lensModel.c_str(), 0);

    if (lenses && lenses[0]) {
        lens = lenses[0];
        printf("  [Lensfun] Lens found: %s %s (score: %d)\n",
               lf_mlstr_get(lens->Maker), lf_mlstr_get(lens->Model), lens->Score);
    } else {
        printf("  [Lensfun] Lens not found: %s %s. Skipping correction.\n",
               meta.lensMaker.c_str(), meta.lensModel.c_str());
        lf_free(lenses);
        return false;
    }
    lf_free(lenses);

    // --- Determine crop factor ---
    float cropFactor = 1.0f;
    if (camera && camera->CropFactor > 0.0f) {
        cropFactor = camera->CropFactor;
    }

    // --- Create modifier ---
    std::unique_ptr<lfModifier, LensfunModifierDeleter> modifier(
        new lfModifier(lens, meta.focalLength, cropFactor,
                       img.width, img.height, LF_PF_F32, false));
    if (!modifier) {
        printf("  [Lensfun] Failed to create modifier.\n");
        return false;
    }

    // --- Enable corrections ---
    // Order matters: enable distortion first (for auto-scale), then TCA, then vignetting
    if (params.correctDistortion) {
        modifier->EnableDistortionCorrection();
        // Auto-scale to eliminate black borders from distortion correction
        float autoScale = modifier->GetAutoScale(false);
        if (autoScale > 0.0f && autoScale != 1.0f) {
            modifier->EnableScaling(autoScale);
            printf("  [Lensfun] Auto-scaling: %.4f\n", autoScale);
        }
    }

    if (params.correctTca) {
        modifier->EnableTCACorrection();
    }

    if (params.correctVignetting) {
        modifier->EnableVignettingCorrection(meta.aperture, params.distance);
    }

    // --- Step 1: Vignetting correction (in-place) ---
    if (params.correctVignetting) {
        // ApplyColorModification works in-place on the pixel data
        int rowStride = img.width * img.channels * static_cast<int>(sizeof(float));
        bool ok = modifier->ApplyColorModification(
            img.ptr(), 0.0f, 0.0f, img.width, img.height,
            LF_CR_3(RED, GREEN, BLUE), rowStride);

        if (ok) {
            printf("  [Lensfun] Vignetting correction applied.\n");
        }
    }

    // --- Step 2: Distortion + TCA correction (coordinate remapping) ---
    if (params.correctDistortion || params.correctTca) {
        // Allocate coordinate buffer: width * height * 2 * 3 floats
        // Layout: for each pixel, 3 (RGB) pairs of (x, y) source coordinates
        size_t coordSize = static_cast<size_t>(img.width) * img.height * 2 * 3;
        std::vector<float> coords(coordSize);

        bool ok = modifier->ApplySubpixelGeometryDistortion(
            0.0f, 0.0f, img.width, img.height, coords.data());

        if (ok) {
            printf("  [Lensfun] Distortion/TCA correction: remapping %dx%d pixels...\n",
                   img.width, img.height);

            // Create output buffer
            ImageBuffer output(img.width, img.height);

            // Remap with bicubic interpolation, per-channel
            const int w = img.width;
            const int h = img.height;
            const int ch = img.channels;

#ifdef RA_USE_OPENMP
            #pragma omp parallel for schedule(dynamic, 16)
#endif
            for (int row = 0; row < h; ++row) {
                for (int col = 0; col < w; ++col) {
                    size_t baseIdx = (static_cast<size_t>(row) * w + col) * 6; // 2*3 per pixel
                    float* outPixel = output.pixel(row, col);

                    for (int c = 0; c < 3; ++c) {
                        // coords layout: [R_x, R_y, G_x, G_y, B_x, B_y] per pixel
                        float srcX = coords[baseIdx + c * 2];
                        float srcY = coords[baseIdx + c * 2 + 1];

                        outPixel[c] = bicubicSample(img.ptr(), w, h, ch, srcX, srcY, c);
                    }
                }
            }

            // Replace image data
            img = std::move(output);
            printf("  [Lensfun] Distortion/TCA correction applied.\n");
        } else {
            printf("  [Lensfun] No distortion/TCA data to apply.\n");
        }
    }

    printf("  [Lensfun] Correction complete.\n");
    return true;
}

#endif // RA_LENSFUN_ENABLED

} // namespace rawalchemy
