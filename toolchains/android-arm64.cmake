# Android arm64-v8a preset
# Usage: cmake -B build-android-arm64 -C toolchains/android-arm64.cmake
#
# Requires: ANDROID_NDK environment variable or -DANDROID_NDK=/path/to/ndk

set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(ANDROID_ABI arm64-v8a)
set(ANDROID_PLATFORM android-21)

# NDK path
if(DEFINED ENV{ANDROID_NDK})
    set(CMAKE_ANDROID_NDK $ENV{ANDROID_NDK} CACHE PATH "Android NDK")
elseif(NOT DEFINED CMAKE_ANDROID_NDK)
    message(FATAL_ERROR "Set ANDROID_NDK env var or pass -DANDROID_NDK=/path/to/ndk")
endif()

set(CMAKE_ANDROID_ARCH_ABI ${ANDROID_ABI})
set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION clang)

# Project-specific options for Android
set(BUILD_SHARED ON CACHE BOOL "" FORCE)
set(BUILD_CAPI ON CACHE BOOL "" FORCE)
set(BUILD_CLI OFF CACHE BOOL "" FORCE)
set(ENABLE_LENS_CORRECTION ON CACHE BOOL "" FORCE)

# libjpeg-turbo: disable SIMD on Android (NASM may not be available)
set(WITH_SIMD OFF CACHE BOOL "" FORCE)
