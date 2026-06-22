/**
 * @file win_unicode.cpp
 * @brief UTF-8 to UTF-16 (wchar_t) path conversion - Windows implementation.
 *
 * Owns the sole #define NOMINMAX + #include <windows.h> boilerplate in the
 * project so every call site shares one MultiByteToWideChar(CP_UTF8, ...)
 * conversion. Compiled only on Windows; other platforms link nothing from
 * this translation unit.
 */

#ifdef _WIN32

#define NOMINMAX
#include <windows.h>

#include "win_unicode.h"

namespace rawalchemy {

std::wstring utf8_to_wide(const std::string& utf8) {
    if (utf8.empty()) return std::wstring();
    int size = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
    if (size <= 0) return std::wstring();
    std::wstring wide(static_cast<size_t>(size - 1), 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, &wide[0], size);
    return wide;
}

}  // namespace rawalchemy

#endif  // _WIN32
