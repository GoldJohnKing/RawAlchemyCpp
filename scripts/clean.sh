#!/usr/bin/env bash
# Clean all RawAlchemyCpp build artifacts.
#
# Removes the build directories declared in the project .gitignore:
#   build/, build-windows-dll/, build-android-*/, cmake-build-*/
#
# The patterns mirror .gitignore so this stays in sync as build variants evolve;
# keep the two lists aligned when adding new build directories.
#
# When a directory cannot be removed because a lingering Windows build process
# (ninja/cmake/MSVC tools from a prior failed build) holds files open, the
# script force-kills those processes and retries before reporting failure.
#
# Usage:
#   ./scripts/clean.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Force-kill lingering Windows build processes that may hold build-directory
# files open (common after a failed or cancelled MSVC/Ninja build). Targets the
# known build-tool image names. Returns 0 if at least one process was killed,
# 1 if cmd.exe is unavailable or nothing matched.
# NOTE: this is broad — it kills any ninja/cmake/MSVC process system-wide. Do
# not run clean while another project's build is active.
kill_lingering_build_processes() {
    command -v cmd.exe >/dev/null 2>&1 || return 1
    # taskkill returns 0 when it terminates at least one of the named images.
    cmd.exe /C "taskkill /f /im ninja.exe /im cmake.exe /im cl.exe /im rc.exe /im link.exe /im lib.exe" >/dev/null 2>&1
}

removed=0
failed=0
for pattern in build build-windows-dll build-android-* cmake-build-*; do
    # Non-matching globs expand to the literal pattern; skip those.
    [ -d "$pattern" ] || continue
    if rm -rf "$pattern" 2>/dev/null; then
        echo "  removed $pattern/"
        removed=$((removed + 1))
    else
        # rm failed — files are likely locked by a lingering build process.
        # Kill the offenders and retry once before giving up.
        echo "  $pattern/ locked — killing build processes and retrying..." >&2
        if kill_lingering_build_processes; then
            echo "    killed lingering build process(es)" >&2
            sleep 1  # let Windows release the file handles
        else
            echo "    no killable Windows build process found (non-Windows lock?)" >&2
        fi
        if rm -rf "$pattern" 2>/dev/null; then
            echo "  removed $pattern/ (after retry)" >&2
            removed=$((removed + 1))
        else
            echo "  WARNING: could not remove $pattern/ even after killing build processes" >&2
            echo "    Reboot or manually close programs holding these files, then re-run clean." >&2
            failed=$((failed + 1))
        fi
    fi
done

if [ "$removed" -eq 0 ] && [ "$failed" -eq 0 ]; then
    echo "  nothing to clean (no build directories found)"
fi

# Non-zero exit if any directory could not be removed completely.
[ "$failed" -eq 0 ]
