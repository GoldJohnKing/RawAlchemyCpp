// SPDX-License-Identifier: AGPL-3.0-or-later
// Apply a pre-composed camRGB -> linear ProPhoto RGB matrix.
// The matrix MUST be composed as LibRaw's convert_to_rgb does:
//   out_cam = prophoto_rgb · rgb_cam
// where rgb_cam = imgdata.color.rgb_cam (derived from cam_xyz via cam_xyz_coeff).
// Applying cam_xyz directly (with or without inversion) is WRONG — LibRaw never
// applies cam_xyz to pixels.
#include "nn_color_adapt.h"

namespace rawalchemy {

void camRgbToProPhotoLinear(float* dst, const float* camRgb, size_t pixelCount,
                            const float outCam[9]) {
    for (size_t i = 0; i < pixelCount; ++i) {
        const float r = camRgb[i * 3 + 0];
        const float g = camRgb[i * 3 + 1];
        const float b = camRgb[i * 3 + 2];
        dst[i * 3 + 0] = outCam[0] * r + outCam[1] * g + outCam[2] * b;
        dst[i * 3 + 1] = outCam[3] * r + outCam[4] * g + outCam[5] * b;
        dst[i * 3 + 2] = outCam[6] * r + outCam[7] * g + outCam[8] * b;
    }
}

} // namespace rawalchemy
