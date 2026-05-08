cmake_minimum_required(VERSION 3.16)
project(RawAlchemy
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Dependencies ----
find_package(PkgConfig)

# LibRaw
# Manual override: -DLIBRAW_ROOT=/path/to/libraw
pkg_check_modules(LIBRAW IMPORTED_TARGET libraw>=0.20.0)
if(NOT LIBRAW_FOUND)
    find_library(LIBRAW_LIBRARY NAMES raw raw_r
        HINTS ${LIBRAW_ROOT}/lib ${LIBRAW_ROOT}/lib64 ${LIBRAW_ROOT})
    find_path(LIBRAW_INCLUDE_DIR NAMES libraw/libraw.h
        HINTS ${LIBRAW_ROOT}/include ${LIBRAW_ROOT})
    if(LIBRAW_LIBRARY AND LIBRAW_INCLUDE_DIR)
        set(LIBRAW_FOUND TRUE)
        set(LIBRAW_LIBRARIES ${LIBRAW_LIBRARY})
        set(LIBRAW_INCLUDE_DIRS ${LIBRAW_INCLUDE_DIR})
    endif()
endif()

if(NOT LIBRAW_FOUND)
    message(FATAL_ERROR
        "LibRaw not found. Install it via:\n"
        "  Debian/Ubuntu: sudo apt install libraw-dev\n"
        "  macOS:         brew install libraw\n"
        "  Windows:       vcpkg install libraw\n"
        "  Or build from source: cmake -DLIBRAW_ROOT=/path/to/libraw ..\n"
    )
endif()

# libtiff
# Manual override: -DLIBTIFF_ROOT=/path/to/libtiff
pkg_check_modules(LIBTIFF IMPORTED_TARGET libtiff-4)
if(NOT LIBTIFF_FOUND)
    find_package(TIFF QUIET)
    if(TIFF_FOUND)
        set(LIBTIFF_FOUND TRUE)
        set(LIBTIFF_LIBRARIES TIFF::TIFF)
    else()
        find_library(LIBTIFF_LIBRARY NAMES tiff libtiff
            HINTS ${LIBTIFF_ROOT}/lib ${LIBTIFF_ROOT}/lib64 ${LIBTIFF_ROOT})
        find_path(LIBTIFF_INCLUDE_DIR NAMES tiffio.h
            HINTS ${LIBTIFF_ROOT}/include ${LIBTIFF_ROOT})
        if(LIBTIFF_LIBRARY AND LIBTIFF_INCLUDE_DIR)
            set(LIBTIFF_FOUND TRUE)
            set(LIBTIFF_LIBRARIES ${LIBTIFF_LIBRARY})
            set(LIBTIFF_INCLUDE_DIRS ${LIBTIFF_INCLUDE_DIR})
        endif()
    endif()
endif()

if(NOT LIBTIFF_FOUND)
    message(FATAL_ERROR
        "libtiff not found. Install it via:\n"
        "  Debian/Ubuntu: sudo apt install libtiff-dev\n"
        "  macOS:         brew install libtiff\n"
        "  Windows:       vcpkg install tiff\n"
        "  Or build from source: cmake -DLIBTIFF_ROOT=/path/to/libtiff ..\n"
    )
endif()

# ---- Library target (for future integration) ----
add_library(raw_alchemy_lib STATIC
    src/raw_decoder.cpp
    src/tiff_writer.cpp
)

target_include_directories(raw_alchemy_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LIBRAW_INCLUDE_DIRS}
    ${LIBTIFF_INCLUDE_DIRS}
)

target_link_libraries(raw_alchemy_lib PUBLIC
    ${LIBRAW_LIBRARIES}
    ${LIBTIFF_LIBRARIES}
)

# Link math library on Linux
if(UNIX)
    target_link_libraries(raw_alchemy_lib PUBLIC m)
endif()

# OpenMP for parallel processing (optional)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(raw_alchemy_lib PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(raw_alchemy_lib PUBLIC RA_USE_OPENMP)
    message(STATUS "OpenMP enabled")
endif()

# ---- CLI executable ----
add_executable(raw_alchemy src/main.cpp)
target_link_libraries(raw_alchemy PRIVATE raw_alchemy_lib)

# ---- Verification tool ----
add_executable(verify src/verify.cpp)
target_link_libraries(verify PRIVATE raw_alchemy_lib)

# ---- Compiler warnings ----
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(raw_alchemy_lib PRIVATE -Wall -Wextra -Wpedantic -O2)
    target_compile_options(raw_alchemy PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---- Status ----
message(STATUS "")
message(STATUS "=== Raw Alchemy (C++ Edition) ===")
message(STATUS "  Version:    ${PROJECT_VERSION}")
message(STATUS "  LibRaw:     ${LIBRAW_INCLUDE_DIRS}")
message(STATUS "  libtiff:    ${LIBTIFF_INCLUDE_DIRS}")
message(STATUS "  OpenMP:     ${OpenMP_CXX_FOUND}")
message(STATUS "=================================")
message(STATUS "")
