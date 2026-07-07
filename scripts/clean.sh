#!/usr/bin/env bash
# Clean all RawAlchemyCpp build artifacts.
#
# Removes the build directories declared in the project .gitignore:
#   build/, build-windows-dll*/ (incl. build-windows-dll_nn-demosaic), build-android-*/, cmake-build-*/
#
# The patterns mirror .gitignore so this stays in sync as build variants evolve;
# keep the two lists aligned when adding new build directories.
#
# A locked file in one directory does NOT abort the whole clean — remaining
# directories are still processed and a non-zero exit reports the partial failure.
#
# Usage:
#   ./scripts/clean.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

removed=0
failed=0
for pattern in build build-windows-dll* build-android-* cmake-build-*; do
    # Non-matching globs expand to the literal pattern; skip those.
    [ -d "$pattern" ] || continue
    if rm -rf "$pattern" 2>/dev/null; then
        echo "  removed $pattern/"
        removed=$((removed + 1))
    else
        echo "  WARNING: could not remove $pattern/" >&2
        echo "    A process may hold files open. Close it or reboot, then re-run clean." >&2
        failed=$((failed + 1))
    fi
done

if [ "$removed" -eq 0 ] && [ "$failed" -eq 0 ]; then
    echo "  nothing to clean (no build directories found)"
fi

# Non-zero exit if any directory could not be removed completely.
[ "$failed" -eq 0 ]
