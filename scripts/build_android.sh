#!/usr/bin/env bash
# Build Raw Alchemy as a shared library (.so) for Android.
#
# Prerequisites:
#   - Android NDK installed (set ANDROID_NDK env var)
#   - CMake 3.18+
#
# Usage:
#   ANDROID_NDK=/path/to/ndk ./scripts/build_android.sh          # arm64 only
#   ANDROID_NDK=/path/to/ndk ./scripts/build_android.sh arm64 armv7 x86_64

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check NDK
if [ -z "${ANDROID_NDK:-}" ]; then
    echo "ERROR: Set ANDROID_NDK environment variable to your NDK path."
    echo "  export ANDROID_NDK=/path/to/android-ndk"
    exit 1
fi

ABIS=("${@:-arm64}")
BUILD_TYPE="${BUILD_TYPE:-Release}"

for ABI in "${ABIS[@]}"; do
    case "$ABI" in
        arm64)   PRESET="android-arm64" ;;
        armv7)   PRESET="android-armv7" ;;
        x86_64)  PRESET="android-x86_64" ;;
        *)       echo "Unknown ABI: $ABI (use: arm64, armv7, x86_64)"; exit 1 ;;
    esac

    BUILD_DIR="$PROJECT_DIR/build-android-$ABI"
    echo "=== Building for Android $ABI ($BUILD_TYPE) ==="

    cmake -B "$BUILD_DIR" \
        -C "$PROJECT_DIR/toolchains/${PRESET}.cmake" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DANDROID_NDK="$ANDROID_NDK"

    cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || echo 4)"

    echo "  -> $BUILD_DIR/libraw_alchemy.so"
    echo ""
done

echo "=== Android builds complete ==="
for ABI in "${ABIS[@]}"; do
    echo "  $ABI: build-android-$ABI/libraw_alchemy.so"
done
