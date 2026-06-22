#pragma once
/**
 * @file win_unicode.h
 * @brief UTF-8 to UTF-16 (wchar_t) path conversion for Windows wide-char C APIs.
 *
 * Centralizes the UTF-8 -> wchar_t conversion required by Windows file APIs
 * (_wfopen, TIFFOpenW, LibRaw::open_file, etc.) that accept only wide paths.
 * On non-Windows platforms the entire content compiles out; callers route
 * through the POSIX branches of the same #ifdef _WIN32 call sites and pass
 * UTF-8 paths directly.
 */

#ifdef _WIN32

#include <string>

namespace rawalchemy {

// Convert a UTF-8 string to UTF-16 (wchar_t) for Windows wide-char C APIs
// (_wfopen, TIFFOpenW, LibRaw::open_file, etc.). Returns empty on failure/empty input.
std::wstring utf8_to_wide(const std::string& utf8);

}  // namespace rawalchemy

#endif  // _WIN32
