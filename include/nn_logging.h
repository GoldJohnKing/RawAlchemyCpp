// CameraFTP - A Cross-platform FTP companion for camera photo transfer
// Copyright (C) 2026 GoldJohnKing <GoldJohnKing@Live.cn>
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>

#if defined(__ANDROID__)
extern "C" int __android_log_write(int prio, const char* tag, const char* text);
#endif

// Minimal logging shim for the NN demosaic core. ORT runs on Android without a
// stdout/stderr that logcat surfaces, so route through __android_log_write.
// On desktop platforms the path defaults to stderr, but the Rust host can
// redirect diagnostics into the app log file via set_log_file (so C++ NN
// messages land in the same app.log as the Rust tracing subscriber).
namespace nnlog {

// Thread-safe storage for an optional log-file path, using function-local
// statics so this header stays header-only (no .cpp / single-definition issue).
inline std::mutex& logMutexRef() {
    static std::mutex m;
    return m;
}

inline std::string& logFilePathRef() {
    static std::string s;
    return s;
}

/// Redirect subsequent info() output to `path` (opened in append mode). Pass
/// NULL to revert to stderr. Deep-copies the path under the log mutex.
inline void set_log_file(const char* path) {
    std::lock_guard<std::mutex> lk(logMutexRef());
    logFilePathRef() = path ? std::string(path) : std::string();
}

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
    // Snapshot the configured path under the mutex, then write outside the
    // lock so a slow disk doesn't serialize concurrent loggers. Fall back to
    // stderr if no path is set or the file can't be opened.
    std::string path;
    {
        std::lock_guard<std::mutex> lk(logMutexRef());
        path = logFilePathRef();
    }
    if (!path.empty()) {
        std::ofstream f(path, std::ios::app);
        if (f) {
            f << "[CameraFTP-NN] " << buf << "\n";
            return;
        }
    }
    std::fprintf(stderr, "[CameraFTP-NN] %s\n", buf);
#endif
}

}  // namespace nnlog
