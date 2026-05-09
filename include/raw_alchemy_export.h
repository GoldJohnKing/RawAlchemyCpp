#pragma once
/**
 * @file raw_alchemy_export.h
 * @brief Platform-specific export macros for shared library builds.
 *
 * RA_API   — Marks a symbol as exported from / imported to the shared library.
 * RA_CALL  — Calling convention (stdcall on Windows 32-bit, cdecl otherwise).
 *
 * When building the shared library, define RA_BUILDING_SHARED.
 * When consuming the shared library as a client, do NOT define it.
 */

// Calling convention
#ifdef _WIN32
    #ifdef _WIN64
        #define RA_CALL
    #else
        #define RA_CALL __stdcall
    #endif
#else
    #define RA_CALL
#endif

// Export / Import (Windows only — needed for __declspec)
// On non-Windows, all symbols are visible by default (CXX_VISIBILITY_PRESET default)
#if defined(RA_BUILDING_SHARED) && defined(_WIN32)
    #define RA_API __declspec(dllexport)
#elif defined(RA_SHARED) && defined(_WIN32)
    #define RA_API __declspec(dllimport)
#else
    #define RA_API
#endif
