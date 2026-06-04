// CameraFTP - A Cross-platform FTP companion for camera photo transfer
// Copyright (C) 2026 GoldJohnKing <GoldJohnKing@Live.cn>
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include "common.h"

namespace rawalchemy {

/** Downscale an image to fit inside maxWidth × maxHeight, preserving aspect ratio.
 *  If both maxWidth and maxHeight are 0 or negative, returns a copy unchanged. */
ImageBuffer resizeImage(const ImageBuffer& src, int maxWidth, int maxHeight);

} // namespace rawalchemy
