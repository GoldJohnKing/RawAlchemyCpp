cmake_minimum_required(VERSION 3.18)
project(raw_alchemy
    VERSION 0.1.0
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Build options ----
option(BUILD_SHARED "Build raw_alchemy_core as a shared library (DLL/SO)" OFF)
option(BUILD_CAPI   "Build C API wrapper for FFI/shared library use" OFF)
option(BUILD_CLI    "Build CLI executable" ON)

# Automatically enable C API when building shared
if(BUILD_SHARED)
    set(BUILD_CAPI ON CACHE BOOL "" FORCE)
    # All static dependencies must be compiled with -fPIC for shared library linking
    set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
endif()

# ---- Dependencies (built from git submodules) ----

# LibRaw via LibRaw-cmake wrapper
set(LIBRAW_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/LibRaw" CACHE STRING "Path to LibRaw source")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ENABLE_LCMS     OFF CACHE BOOL "" FORCE)
set(ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ENABLE_JASPER   OFF CACHE BOOL "" FORCE)
set(ENABLE_RAWSPEED OFF CACHE BOOL "" FORCE)
set(LIBRAW_INSTALL  OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/LibRaw-cmake)

set(RA_LIBRAW_TARGET libraw::libraw_r)
set(RA_LIBRAW_SOURCE "submodule (LibRaw)")

# libexif — minimal build (EXIF APP1 serialization for JPEG output)
set(LIBEXIF_VERSION "0.6.26")
# Detect endianness for libexif config.h
if(DEFINED CMAKE_C_BYTE_ORDER)
    if(CMAKE_C_BYTE_ORDER STREQUAL "BIG")
        set(endianness_def "#define WORDS_BIGENDIAN 1")
    else()
        set(endianness_def "/* #undef WORDS_BIGENDIAN */")
    endif()
else()
    # Fallback: check via test
    include(TestBigEndian)
    TestBigEndian(IS_BIG_ENDIAN)
    if(IS_BIG_ENDIAN)
        set(endianness_def "#define WORDS_BIGENDIAN 1")
    else()
        set(endianness_def "/* #undef WORDS_BIGENDIAN */")
    endif()
endif()
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libexif_config/config.h.in
    ${CMAKE_BINARY_DIR}/libexif_generated/config.h @ONLY
)

set(LIBEXIF_SOURCES
    third_party/libexif/libexif/exif-byte-order.c
    third_party/libexif/libexif/exif-content.c
    third_party/libexif/libexif/exif-data.c
    third_party/libexif/libexif/exif-entry.c
    third_party/libexif/libexif/exif-format.c
    third_party/libexif/libexif/exif-gps-ifd.c
    third_party/libexif/libexif/exif-ifd.c
    third_party/libexif/libexif/exif-loader.c
    third_party/libexif/libexif/exif-log.c
    third_party/libexif/libexif/exif-mem.c
    third_party/libexif/libexif/exif-mnote-data.c
    third_party/libexif/libexif/exif-tag.c
    third_party/libexif/libexif/exif-utils.c
    # MakerNote parsers (needed for exif-data.c to compile even though we skip MakerNote)
    third_party/libexif/libexif/canon/exif-mnote-data-canon.c
    third_party/libexif/libexif/canon/mnote-canon-entry.c
    third_party/libexif/libexif/canon/mnote-canon-tag.c
    third_party/libexif/libexif/fuji/exif-mnote-data-fuji.c
    third_party/libexif/libexif/fuji/mnote-fuji-entry.c
    third_party/libexif/libexif/fuji/mnote-fuji-tag.c
    third_party/libexif/libexif/olympus/exif-mnote-data-olympus.c
    third_party/libexif/libexif/olympus/mnote-olympus-entry.c
    third_party/libexif/libexif/olympus/mnote-olympus-tag.c
    third_party/libexif/libexif/pentax/exif-mnote-data-pentax.c
    third_party/libexif/libexif/pentax/mnote-pentax-entry.c
    third_party/libexif/libexif/pentax/mnote-pentax-tag.c
    third_party/libexif/libexif/apple/exif-mnote-data-apple.c
    third_party/libexif/libexif/apple/mnote-apple-entry.c
    third_party/libexif/libexif/apple/mnote-apple-tag.c
)

add_library(libexif STATIC ${LIBEXIF_SOURCES})
target_include_directories(libexif PUBLIC
    ${CMAKE_BINARY_DIR}/libexif_generated                # config.h
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libexif      # libexif/exif-data.h
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libexif/libexif  # exif-content.h etc.
)
# Compile as C99
target_compile_features(libexif PUBLIC c_std_99)
# Suppress warnings from third-party code
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(libexif PRIVATE -w)
endif()

set(RA_LIBEXIF_TARGET libexif)
set(RA_LIBEXIF_SOURCE "submodule (libexif ${LIBEXIF_VERSION})")

# libtiff — minimal build (uncompressed 16-bit TIFF output only)
set(BUILD_SHARED_LIBS  OFF CACHE BOOL "" FORCE)
set(tiff-tools      OFF CACHE BOOL "" FORCE)
set(tiff-tests      OFF CACHE BOOL "" FORCE)
set(tiff-contrib    OFF CACHE BOOL "" FORCE)
set(tiff-docs       OFF CACHE BOOL "" FORCE)
set(tiff-install    OFF CACHE BOOL "" FORCE)
set(tiff-cxx        OFF CACHE BOOL "" FORCE)
set(tiff-opengl     OFF CACHE BOOL "" FORCE)
set(jpeg            OFF CACHE BOOL "" FORCE)
set(jbig            OFF CACHE BOOL "" FORCE)
set(lerc            OFF CACHE BOOL "" FORCE)
set(lzma            OFF CACHE BOOL "" FORCE)
set(zstd            OFF CACHE BOOL "" FORCE)
set(webp            OFF CACHE BOOL "" FORCE)
set(zlib            OFF CACHE BOOL "" FORCE)
set(ccitt           OFF CACHE BOOL "" FORCE)
set(packbits        OFF CACHE BOOL "" FORCE)
set(lzw             OFF CACHE BOOL "" FORCE)
set(thunder         OFF CACHE BOOL "" FORCE)
set(next            OFF CACHE BOOL "" FORCE)
set(logluv          OFF CACHE BOOL "" FORCE)
set(mdi             OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/libtiff)

set(RA_LIBTIFF_TARGET tiff)
set(RA_LIBTIFF_SOURCE "submodule (libtiff)")

# libjpeg-turbo — build from submodule via ExternalProject
include(ExternalProject)

set(LIBJPEG_TURBO_SRC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjpeg-turbo")
set(LIBJPEG_TURBO_BUILD "${CMAKE_BINARY_DIR}/third_party/libjpeg-turbo")
set(LIBJPEG_TURBO_INSTALL "${CMAKE_BINARY_DIR}/third_party/libjpeg-turbo-install")

# Create the include directory so CMake doesn't complain about non-existent paths
file(MAKE_DIRECTORY ${LIBJPEG_TURBO_INSTALL}/include)

# libjpeg-turbo names the static library differently per platform:
#   Windows: turbojpeg-static.lib
#   Linux/macOS/Android: libturbojpeg.a
if(WIN32)
    set(TURBOJPEG_LIB_NAME "turbojpeg-static")
else()
    set(TURBOJPEG_LIB_NAME "turbojpeg")
endif()

ExternalProject_Add(libjpeg-turbo_external
    SOURCE_DIR   ${LIBJPEG_TURBO_SRC}
    BINARY_DIR   ${LIBJPEG_TURBO_BUILD}
    INSTALL_DIR  ${LIBJPEG_TURBO_INSTALL}
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${LIBJPEG_TURBO_INSTALL}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DENABLE_SHARED=OFF
        -DENABLE_STATIC=ON
        -DWITH_SIMD=ON
        -DWITH_ARITH_DEC=OFF
        -DWITH_ARITH_ENC=OFF
        -DWITH_JAVA=OFF
        -DWITH_JPEG7=OFF
        -DWITH_JPEG8=OFF
        -DWITH_TOOLS=OFF
        -DWITH_TESTS=OFF
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    BUILD_BYPRODUCTS ${LIBJPEG_TURBO_INSTALL}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${TURBOJPEG_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}
)

# Create an imported static library target for libjpeg-turbo
# Uses config-specific IMPORTED_LOCATION for MSVC multi-config generator compatibility
add_library(turbojpeg STATIC IMPORTED GLOBAL)
set(TURBOJPEG_LIB "${LIBJPEG_TURBO_INSTALL}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${TURBOJPEG_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}")
set_target_properties(turbojpeg PROPERTIES
    IMPORTED_LOCATION "${TURBOJPEG_LIB}"
    IMPORTED_LOCATION_RELEASE "${TURBOJPEG_LIB}"
    IMPORTED_LOCATION_DEBUG "${TURBOJPEG_LIB}"
    IMPORTED_LOCATION_MINSIZEREL "${TURBOJPEG_LIB}"
    IMPORTED_LOCATION_RELWITHDEBINFO "${TURBOJPEG_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${LIBJPEG_TURBO_INSTALL}/include"
)
add_dependencies(turbojpeg libjpeg-turbo_external)

set(RA_LIBJPEG_TARGET turbojpeg)
set(RA_LIBJPEG_SOURCE "submodule (libjpeg-turbo)")

# ---- Lensfun (lens correction via Lensfun + GLib2 shim) ----
# Compiled directly from submodule sources with a GLib2 compatibility shim.
# No external GLib2 dependency — all GLib2 calls are replaced by C++17 equivalents
# via a drop-in header + implementation that maps to std::string, std::filesystem,
# std::mutex, std::vector, and pugixml (for XML parsing).
option(ENABLE_LENS_CORRECTION "Enable lens correction via Lensfun library" ON)
set(RA_LENSFUN_ENABLED OFF)

if(ENABLE_LENS_CORRECTION)
    # --- pugixml (XML parser, used by GLib2 shim for GMarkupParser bridge) ---
    set(PUGIXML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(PUGIXML_INSTALL OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/pugixml)

    # --- Generate Lensfun config.h ---
    # Every Lensfun source file does #include "config.h" (14 files).
    set(LENSFUN_VERSION "0.3.99.0")
    set(LENSFUN_DB_VERSION "2")
    include(CheckIncludeFiles)
    CHECK_INCLUDE_FILES(endian.h HAVE_ENDIAN_H)

    # Detect SSE support for x86
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "[XxIi][0-9]?86|[Aa][Mm][Dd]64")
        set(VECTORIZATION_SSE 1)
        set(VECTORIZATION_SSE2 1)
    endif()

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lensfun_config/config.h.in
        ${CMAKE_BINARY_DIR}/lensfun_generated/config.h @ONLY
    )

    # --- Generate lensfun.h from template ---
    set(VERSION_MAJOR 0)
    set(VERSION_MINOR 3)
    set(VERSION_MICRO 99)
    set(VERSION_BUGFIX 0)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lensfun/include/lensfun/lensfun.h.in
        ${CMAKE_BINARY_DIR}/lensfun_generated/lensfun/lensfun.h @ONLY
    )

    # --- GLib2 compatibility shim library ---
    # Provides glib.h + glib/gstdio.h that Lensfun source files include.
    # All implementations map GLib2 calls to C++17 standard library equivalents.
    add_library(glib_shim STATIC
        third_party/glib_shim/src/glib_shim.cpp
    )
    target_include_directories(glib_shim PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glib_shim/include
    )
    target_link_libraries(glib_shim PUBLIC pugixml::pugixml)
    target_compile_features(glib_shim PUBLIC cxx_std_17)
    # std::mutex needs pthread on Linux
    find_package(Threads)
    if(Threads_FOUND)
        target_link_libraries(glib_shim PUBLIC Threads::Threads)
    endif()

    # --- Lensfun source files (compiled directly into our project) ---
    set(LENSFUN_SOURCES
        third_party/lensfun/libs/lensfun/database.cpp
        third_party/lensfun/libs/lensfun/auxfun.cpp
        third_party/lensfun/libs/lensfun/camera.cpp
        third_party/lensfun/libs/lensfun/lens.cpp
        third_party/lensfun/libs/lensfun/mount.cpp
        third_party/lensfun/libs/lensfun/modifier.cpp
        third_party/lensfun/libs/lensfun/cpuid.cpp
        third_party/lensfun/libs/lensfun/mod-color.cpp
        third_party/lensfun/libs/lensfun/mod-color-sse.cpp
        third_party/lensfun/libs/lensfun/mod-color-sse2.cpp
        third_party/lensfun/libs/lensfun/mod-coord.cpp
        third_party/lensfun/libs/lensfun/mod-coord-sse.cpp
        third_party/lensfun/libs/lensfun/mod-pc.cpp
        third_party/lensfun/libs/lensfun/mod-subpix.cpp
    )

    add_library(lensfun STATIC ${LENSFUN_SOURCES})

    # Include path order matters for correct header resolution:
    # 1. Generated config.h and lensfun.h (for #include "config.h" and #include "lensfun.h")
    # 2. Lensfun source directory (for #include "lensfunprv.h")
    # 3. GLib2 shim (for #include <glib.h> from lensfunprv.h) — resolved via glib_shim link
    target_include_directories(lensfun PRIVATE
        ${CMAKE_BINARY_DIR}/lensfun_generated
        ${CMAKE_BINARY_DIR}/lensfun_generated/lensfun
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lensfun/libs/lensfun
    )
    # GLib2 shim comes from linking glib_shim (its PUBLIC include dirs propagate)
    target_link_libraries(lensfun PRIVATE glib_shim)

    # Compile definitions for Lensfun
    target_compile_definitions(lensfun PUBLIC CONF_LENSFUN_STATIC)

    # SSE/SSE2 flags for x86 optimization
    if(VECTORIZATION_SSE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(
            third_party/lensfun/libs/lensfun/mod-color-sse.cpp
            third_party/lensfun/libs/lensfun/mod-coord-sse.cpp
            PROPERTIES COMPILE_FLAGS "-msse"
        )
    endif()
    if(VECTORIZATION_SSE2 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(
            third_party/lensfun/libs/lensfun/mod-color-sse2.cpp
            PROPERTIES COMPILE_FLAGS "-msse2"
        )
    endif()

    # Suppress warnings from Lensfun source (third-party code)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(lensfun PRIVATE -w)
    endif()

    set(RA_LENSFUN_ENABLED ON)
    set(RA_LENSFUN_SOURCE "submodule (Lensfun 0.3.99 + GLib2 shim)")
endif()

# ---- Library target ----
set(RA_LIBRARY_SOURCES
    src/raw_decoder.cpp
    src/tiff_writer.cpp
    src/jpeg_writer.cpp
    src/log_transform.cpp
    src/lut_applier.cpp
    src/metering.cpp
    src/stylize.cpp
    src/lens_correction.cpp
    src/exif_injector.cpp
)

# Add C API sources when building the C API
if(BUILD_CAPI)
    list(APPEND RA_LIBRARY_SOURCES src/raw_alchemy_capi.cpp)
endif()

if(BUILD_SHARED)
    add_library(raw_alchemy_core SHARED ${RA_LIBRARY_SOURCES})
    # All code linked into a shared library must be position-independent
    set_target_properties(raw_alchemy_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
    add_library(raw_alchemy_core STATIC ${RA_LIBRARY_SOURCES})
endif()

# Shared library specific settings
if(BUILD_SHARED)
    target_compile_definitions(raw_alchemy_core
        PRIVATE RA_BUILDING_SHARED
        PUBLIC RA_SHARED
    )
    # All symbols visible by default — C++ symbols for internal linking, C API for FFI
    set_target_properties(raw_alchemy_core PROPERTIES
        CXX_VISIBILITY_PRESET default
        VISIBILITY_INLINES_HIDDEN OFF
    )
    # Windows: generate .lib import library alongside .dll
    if(WIN32)
        set_target_properties(raw_alchemy_core PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS ON
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
            ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        )
    endif()
    # Android: set library output name
    if(ANDROID)
        set_target_properties(raw_alchemy_core PROPERTIES
            OUTPUT_NAME "raw_alchemy"
        )
    endif()
    # Version the shared library (Linux/macOS)
    if(UNIX AND NOT ANDROID)
        set_target_properties(raw_alchemy_core PROPERTIES
            VERSION ${PROJECT_VERSION}
            SOVERSION ${PROJECT_VERSION_MAJOR}
        )
    endif()
endif()

target_include_directories(raw_alchemy_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(raw_alchemy_core PUBLIC
    ${RA_LIBRAW_TARGET}
    ${RA_LIBTIFF_TARGET}
    ${RA_LIBJPEG_TARGET}
    ${RA_LIBEXIF_TARGET}
)

# Lensfun linkage
if(RA_LENSFUN_ENABLED)
    target_compile_definitions(raw_alchemy_core PUBLIC RA_LENSFUN_ENABLED)
    target_include_directories(raw_alchemy_core PUBLIC
        ${CMAKE_BINARY_DIR}/lensfun_generated  # lensfun.h for public API
    )
    target_link_libraries(raw_alchemy_core PUBLIC lensfun)
endif()

# Link math library on Linux
if(UNIX)
    target_link_libraries(raw_alchemy_core PUBLIC m)
endif()

# OpenMP for parallel processing (optional)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(raw_alchemy_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(raw_alchemy_core PUBLIC RA_USE_OPENMP)
    message(STATUS "OpenMP enabled")
endif()

# ---- CLI executable ----
if(BUILD_CLI)
    add_executable(raw_alchemy_cli src/main.cpp)
    target_link_libraries(raw_alchemy_cli PRIVATE raw_alchemy_core)

    # ---- Verification tool ----
    add_executable(raw_alchemy_verify src/verify.cpp)
    target_link_libraries(raw_alchemy_verify PRIVATE raw_alchemy_core)

    # Windows: place executables next to the DLL so they can find it at runtime
    if(WIN32 AND BUILD_SHARED)
        set_target_properties(raw_alchemy_cli raw_alchemy_verify PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )
    endif()

    # Compiler warnings for CLI targets
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(raw_alchemy_cli PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

# ---- Compiler warnings ----
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(raw_alchemy_core PRIVATE -Wall -Wextra -Wpedantic -O2)
endif()

# ---- Status ----
message(STATUS "")
message(STATUS "=== Raw Alchemy (C++ Edition) ===")
message(STATUS "  Version:    ${PROJECT_VERSION}")
message(STATUS "  Shared:     ${BUILD_SHARED}")
message(STATUS "  C API:      ${BUILD_CAPI}")
message(STATUS "  CLI:        ${BUILD_CLI}")
message(STATUS "  LibRaw:     ${RA_LIBRAW_SOURCE}")
message(STATUS "  libtiff:    ${RA_LIBTIFF_SOURCE}")
message(STATUS "  JPEG:       ${RA_LIBJPEG_SOURCE}")
message(STATUS "  libexif:    ${RA_LIBEXIF_SOURCE}")
message(STATUS "  Lensfun:    ${RA_LENSFUN_ENABLED}")
    if(RA_LENSFUN_ENABLED)
        message(STATUS "             ${RA_LENSFUN_SOURCE}")
    endif()
message(STATUS "  OpenMP:     ${OpenMP_CXX_FOUND}")
if(ANDROID)
    message(STATUS "  Android:    ${ANDROID_ABI} (API ${ANDROID_PLATFORM_LEVEL})")
endif()
message(STATUS "=================================")
message(STATUS "")
