@echo off
REM Build Raw Alchemy as a shared library (DLL) on Windows using LLVM/Clang (clang-cl).
REM
REM Requires (NO VS Clang component or MSBuild LLVM integration needed):
REM   - Visual Studio 2019+ with the "Desktop development with C++" workload
REM     (provides MSVC headers/libs, Windows SDK, and a bundled Ninja).
REM   - LLVM/Clang from llvm.org (or: winget install LLVM.LLVM). clang-cl.exe in PATH.
REM
REM Uses the Ninja generator + clang-cl directly, so no VS platform toolset
REM registration is needed. Output is MSVC-ABI and links cleanly against the Rust
REM msvc toolchain used by the host app, while shipping the latest libomp
REM (OpenMP 5.x) runtime from LLVM.
REM
REM Linker: clang-cl's driver selects lld-link (shipped alongside clang-cl by
REM llvm.org) when linking with -flto=thin. This is required because only lld-link
REM can consume LLVM bitcode objects (MSVC link.exe cannot). CMakeLists.txt passes
REM an explicit .def file — .def processing is object-format-independent, so it
REM keeps ThinLTO working while pinning the export surface to the C API.
REM
REM Usage:
REM   build_windows.bat                 -- builds DLL (Release, neural)
REM   build_windows.bat Debug           -- builds DLL (Debug)
REM   build_windows.bat Release legacy  -- builds DLL (Release, legacy — no NN)

setlocal enableextensions

set "BUILD_TYPE=Release"
if not "%~1"=="" set "BUILD_TYPE=%~1"

REM Validate BUILD_TYPE: build.rs looks for the DLL under bin/<BUILD_TYPE>/, so a
REM typo (e.g. "RelWithDebInfo") would silently produce a layout the host can't find.
if /I not "%BUILD_TYPE%"=="Debug" if /I not "%BUILD_TYPE%"=="Release" (
    echo ERROR: BUILD_TYPE must be Debug or Release, got "%BUILD_TYPE%".
    exit /b 1
)

set "VARIANT=neural"
if not "%~2"=="" set "VARIANT=%~2"

REM Validate VARIANT: neural builds the NN-linked DLL, legacy omits NN linkage.
if /I not "%VARIANT%"=="neural" if /I not "%VARIANT%"=="legacy" (
    echo ERROR: VARIANT must be neural or legacy, got "%VARIANT%".
    exit /b 1
)

set "BUILD_DIR=build-windows-dll"
set "NN_FLAG=ON"
if /I "%VARIANT%"=="legacy" (
    set "BUILD_DIR=build-windows-dll-legacy"
    set "NN_FLAG=OFF"
)

REM Locate the latest Visual Studio installation via vswhere (bundled with VS Installer).
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -property installationPath`) do set "VS_INSTALL=%%i"

if not exist "%VS_INSTALL%\VC\Auxiliary\Build\vcvars64.bat" (
    echo ERROR: Visual Studio with C++ build tools not found.
    echo        Install the "Desktop development with C++" workload.
    exit /b 1
)

REM Set up the MSVC environment (INCLUDE/LIB for Windows SDK + MSVC STL headers),
REM and put the VS-bundled Ninja on PATH.
call "%VS_INSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=%VS_INSTALL%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"

echo === Building Raw Alchemy Shared Library with clang-cl (%BUILD_TYPE%, %VARIANT%) ===

REM BUILD_CLI=OFF matches the Android FFI build and keeps the DLL surface to the
REM C API only (see CMakeLists.txt for the LTO / .def export details). CMake
REM auto-selects lld-link here; CMakeLists passes an explicit .def (highest export
REM priority in lld-link) so ThinLTO works AND the C API is exported.
cmake -B "%BUILD_DIR%" -G Ninja ^
    -DCMAKE_C_COMPILER=clang-cl ^
    -DCMAKE_CXX_COMPILER=clang-cl ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DBUILD_SHARED=ON ^
    -DBUILD_CAPI=ON ^
    -DBUILD_CLI=OFF ^
    -DRA_ENABLE_NN_DEMOSAIC=%NN_FLAG%

if %ERRORLEVEL% neq 0 (
    echo ERROR: CMake configuration failed.
    echo        Ensure LLVM is installed and clang-cl.exe is in PATH.
    exit /b 1
)

cmake --build "%BUILD_DIR%" -j%NUMBER_OF_PROCESSORS%

if %ERRORLEVEL% neq 0 (
    echo ERROR: Build failed.
    exit /b 1
)

echo.
echo === Build complete ===
echo DLL: %BUILD_DIR%\bin\%BUILD_TYPE%\raw_alchemy_core.dll
echo LIB: %BUILD_DIR%\lib\%BUILD_TYPE%\raw_alchemy_core.lib

endlocal
