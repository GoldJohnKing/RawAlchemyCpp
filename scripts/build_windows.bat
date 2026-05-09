@echo off
REM Build Raw Alchemy as a shared library (DLL) on Windows.
REM Requires: Visual Studio 2019+ with CMake, or MinGW.
REM
REM Usage:
REM   build_windows.bat          -- builds DLL with MSVC (Release)
REM   build_windows.bat Debug    -- builds DLL with MSVC (Debug)
REM   build_windows.bat Release MinGW -- builds DLL with MinGW

setlocal

set "BUILD_TYPE=Release"
if not "%~1"=="" set "BUILD_TYPE=%~1"

set "GENERATOR="
if /I "%~2"=="MinGW" set "GENERATOR=-G "MinGW Makefiles""

set "BUILD_DIR=build-windows-dll"

echo === Building Raw Alchemy Shared Library (%BUILD_TYPE%) ===

cmake -B "%BUILD_DIR%" %GENERATOR% ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DBUILD_SHARED=ON ^
    -DBUILD_CAPI=ON ^
    -DBUILD_CLI=ON

if %ERRORLEVEL% neq 0 (
    echo ERROR: CMake configuration failed.
    exit /b 1
)

cmake --build "%BUILD_DIR%" --config %BUILD_TYPE% -j%NUMBER_OF_PROCESSORS%

if %ERRORLEVEL% neq 0 (
    echo ERROR: Build failed.
    exit /b 1
)

echo.
echo === Build complete ===
if "%GENERATOR%"=="" (
    echo DLL: %BUILD_DIR%\bin\%BUILD_TYPE%\raw_alchemy_core.dll
    echo LIB: %BUILD_DIR%\lib\%BUILD_TYPE%\raw_alchemy_core.lib
    echo CLI: %BUILD_DIR%\bin\%BUILD_TYPE%\raw_alchemy_cli.exe
    echo Verify: %BUILD_DIR%\bin\%BUILD_TYPE%\raw_alchemy_verify.exe
) else (
    echo DLL: %BUILD_DIR%\bin\raw_alchemy_core.dll
    echo LIB: %BUILD_DIR%\lib\raw_alchemy_core.lib
    echo CLI: %BUILD_DIR%\bin\raw_alchemy_cli.exe
    echo Verify: %BUILD_DIR%\bin\raw_alchemy_verify.exe
)

endlocal
