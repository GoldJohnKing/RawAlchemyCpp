// CameraFTP - A Cross-platform FTP companion for camera photo transfer
// Copyright (C) 2026 GoldJohnKing <GoldJohnKing@Live.cn>
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <cstdarg>
#include <cstdio>
#include <string>

#if defined(__ANDROID__)
extern "C" int __android_log_write(int prio, const char* tag, const char* text);
#endif

// Minimal logging shim for the NN demosaic core. ORT runs on Android without a
// stdout/stderr that logcat surfaces, so route through __android_log_write.
// Everywhere else, fprintf(stderr, ...) is sufficient.
namespace nnlog {

inline void info(const char* fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    int n = std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    if (n < 0) {
        return;
    }
#if defined(__ANDROID__)
    __android_log_write(4 /*ANDROID_LOG_INFO*/, "CameraFTP-NN", buf);
#else
    std::fprintf(stderr, "[CameraFTP-NN] %s\n", buf);
#endif
}

}  // namespace nnlog
