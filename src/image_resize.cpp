// CameraFTP - A Cross-platform FTP companion for camera photo transfer
// Copyright (C) 2026 GoldJohnKing <GoldJohnKing@Live.cn>
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "image_resize.h"
#include <algorithm>
#include <cmath>

namespace rawalchemy {

ImageBuffer resizeImage(const ImageBuffer& src, int maxWidth, int maxHeight) {
    if (maxWidth <= 0 && maxHeight <= 0) return src;

    int dstW = src.width;
    int dstH = src.height;

    if (maxWidth > 0 && dstW > maxWidth) {
        float scale = static_cast<float>(maxWidth) / dstW;
        dstW = maxWidth;
        dstH = static_cast<int>(src.height * scale);
    }
    if (maxHeight > 0 && dstH > maxHeight) {
        float scale = static_cast<float>(maxHeight) / dstH;
        dstH = maxHeight;
        dstW = static_cast<int>(dstW * scale);
    }
    if (dstW < 1) dstW = 1;
    if (dstH < 1) dstH = 1;
    if (dstW == src.width && dstH == src.height) return src;

    ImageBuffer dst(dstW, dstH);
    float xRatio = static_cast<float>(src.width) / dstW;
    float yRatio = static_cast<float>(src.height) / dstH;

    for (int y = 0; y < dstH; ++y) {
        float srcY = y * yRatio;
        int y0 = static_cast<int>(srcY);
        int y1 = std::min(y0 + 1, src.height - 1);
        float fy = srcY - y0;

        for (int x = 0; x < dstW; ++x) {
            float srcX = x * xRatio;
            int x0 = static_cast<int>(srcX);
            int x1 = std::min(x0 + 1, src.width - 1);
            float fx = srcX - x0;

            for (int c = 0; c < src.channels; ++c) {
                float v00 = *(src.pixel(y0, x0) + c);
                float v10 = *(src.pixel(y0, x1) + c);
                float v01 = *(src.pixel(y1, x0) + c);
                float v11 = *(src.pixel(y1, x1) + c);

                float v0 = v00 + (v10 - v00) * fx;
                float v1 = v01 + (v11 - v01) * fx;
                *(dst.pixel(y, x) + c) = v0 + (v1 - v0) * fy;
            }
        }
    }
    return dst;
}

} // namespace rawalchemy
