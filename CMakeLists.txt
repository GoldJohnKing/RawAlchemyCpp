cmake_minimum_required(VERSION 3.18)
project(RawAlchemy
    VERSION 0.1.0
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
    BUILD_BYPRODUCTS ${LIBJPEG_TURBO_INSTALL}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}turbojpeg${CMAKE_STATIC_LIBRARY_SUFFIX}
)

# Create an imported static library target for libjpeg-turbo
add_library(turbojpeg STATIC IMPORTED GLOBAL)
set_target_properties(turbojpeg PROPERTIES
    IMPORTED_LOCATION ${LIBJPEG_TURBO_INSTALL}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}turbojpeg${CMAKE_STATIC_LIBRARY_SUFFIX}
    INTERFACE_INCLUDE_DIRECTORIES ${LIBJPEG_TURBO_INSTALL}/include
)
add_dependencies(turbojpeg libjpeg-turbo_external)

set(RA_LIBJPEG_TARGET turbojpeg)
set(RA_LIBJPEG_SOURCE "submodule (libjpeg-turbo)")

# ---- Library target ----
add_library(raw_alchemy_lib STATIC
    src/raw_decoder.cpp
    src/tiff_writer.cpp
    src/jpeg_writer.cpp
    src/log_transform.cpp
    src/lut_applier.cpp
    src/metering.cpp
    src/stylize.cpp
)

target_include_directories(raw_alchemy_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(raw_alchemy_lib PUBLIC
    ${RA_LIBRAW_TARGET}
    ${RA_LIBTIFF_TARGET}
    ${RA_LIBJPEG_TARGET}
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
message(STATUS "  LibRaw:     ${RA_LIBRAW_SOURCE}")
message(STATUS "  libtiff:    ${RA_LIBTIFF_SOURCE}")
message(STATUS "  JPEG:       ${RA_LIBJPEG_SOURCE}")
message(STATUS "  OpenMP:     ${OpenMP_CXX_FOUND}")
message(STATUS "=================================")
message(STATUS "")
