"""
Cross-validation: Phase 1 RAW mosaic decode + black-subtract + hot-pixel fix.

Compares the C++ `raw_alchemy_cli --dump-mosaic` output against a numpy
re-implementation of `raw_alchemy.core.subtract_black_level` (core.py:20-31)
and `raw_alchemy.core.fix_hot_pixels` (core.py:34-46), sourcing the raw
mosaic + metadata from rawpy (the LibRaw python binding).

Run with the Raw-Alchemy venv (has rawpy + numpy + scipy):
    /mnt/d/GitRepos/Raw-Alchemy/.venv/bin/python \\
        /mnt/d/GitRepos/RawAlchemyCpp/Test/cross_validate_blackpixel.py
"""
import os
import re
import subprocess
import sys

import numpy as np
import rawpy
from scipy.ndimage import median_filter

# ---- Paths ----
REPO = "/mnt/d/GitRepos/RawAlchemyCpp"
RAW = os.path.join(REPO, "Test", "Sample.NEF")
CLI = os.path.join(REPO, "build", "raw_alchemy_cli")
CPP_MOSAIC = "/tmp/cpp_mosaic.bin"

# PASS threshold (uint16 quantization + tiny median-border differences).
PASS_MAX_ABS = 0.002


# ---- Reference implementations (ports of raw_alchemy.core) ----
def subtract_black_level(sensor_raw, bl, wl, cfa_pattern):
    """Port of raw_alchemy.core.subtract_black_level (core.py:20-31)."""
    pat_size = cfa_pattern.shape[0]
    result = np.empty_like(sensor_raw)
    for r in range(pat_size):
        for c in range(pat_size):
            color = cfa_pattern[r, c]
            bl_c = float(bl[min(color, len(bl) - 1)])
            result[r::pat_size, c::pat_size] = np.maximum(
                sensor_raw[r::pat_size, c::pat_size] - bl_c, 0
            ) / (wl - bl_c)
    return result


def fix_hot_pixels(raw_norm, cfa_pattern, threshold=4.0):
    """Port of raw_alchemy.core.fix_hot_pixels (core.py:34-46).

    Oracle uses scipy.ndimage.median_filter(size=3, mode='nearest')
    (== BORDER_REPLICATE) in place of cv2.medianBlur.
    """
    pat_size = cfa_pattern.shape[0]
    for r in range(pat_size):
        for c in range(pat_size):
            plane = raw_norm[r::pat_size, c::pat_size]
            plane32 = plane.astype(np.float32) if plane.dtype != np.float32 else plane
            med = median_filter(plane32, size=3, mode='nearest')
            diff = np.abs(plane - med)
            std = max(np.std(diff), 1e-6)
            hot = diff > threshold * std
            plane[hot] = med[hot]


def main():
    print(f"=== Phase 1 Cross-Validation: {os.path.basename(RAW)} ===\n")

    # ---- Reference: rawpy + numpy oracle ----
    print("[Python oracle] decoding via rawpy...")
    with rawpy.imread(RAW) as raw:
        sensor_raw = raw.raw_image_visible.astype(np.float32)
        bl = np.array(raw.black_level_per_channel, dtype=np.float32)
        wl = float(raw.white_level)
        cfa_pattern = raw.raw_pattern.copy()
        num_colors = raw.num_colors

    print(f"  mosaic shape: {sensor_raw.shape}  dtype={sensor_raw.dtype}")
    print(f"  CFA pattern (raw_pattern):\n{cfa_pattern}")
    print(f"  black_per_channel: {bl}")
    print(f"  white_level: {wl}")
    print(f"  num_colors: {num_colors}")
    print()

    # Note on Bayer color ordering:
    #   rawpy `raw_pattern` uses LibRaw's color indices 0=R, 1=G, 2=B, 3=G2
    #   (matching `cdesc`). The C++ side uses LibRaw's FC macro which
    #   produces the SAME indices for the same `filters` value. We verified
    #   empirically that for Sample.NEF: rawpy = [[0,1],[3,2]] and
    #   LibRaw FC(2x2) = [[0,1],[3,2]] — identical. No mapping required.
    print("  CFA color-order mapping: rawpy == LibRaw FC (no remap needed)\n")

    print("[Python oracle] subtract_black_level + fix_hot_pixels...")
    raw_norm = subtract_black_level(sensor_raw, bl, wl, cfa_pattern)
    fix_hot_pixels(raw_norm, cfa_pattern)
    print()

    # ---- C++ ----
    print(f"[C++] {CLI} --dump-mosaic {CPP_MOSAIC}")
    proc = subprocess.run(
        [CLI, RAW, "/tmp/dummy.tiff", "--dump-mosaic", CPP_MOSAIC],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print("C++ failed (returncode {}):\n{}".format(proc.returncode, proc.stderr))
        sys.exit(1)

    # Parse "MOSAIC W H" from stdout.
    m = re.search(r"^MOSAIC\s+(\d+)\s+(\d+)\s*$", proc.stdout, re.MULTILINE)
    if not m:
        print("Could not find 'MOSAIC W H' in C++ stdout:\n" + proc.stdout)
        sys.exit(1)
    W, H = int(m.group(1)), int(m.group(2))
    print(f"  C++ mosaic dims: {W} x {H}")

    if (H, W) != sensor_raw.shape:
        print(f"  WARNING: dim mismatch with rawpy oracle {sensor_raw.shape}")
        print("  (top/left margin handling may differ — continuing anyway)")

    cpp = np.fromfile(CPP_MOSAIC, dtype="<u2").reshape(H, W).astype(np.float32) / 65535.0
    print()

    # ---- Compare ----
    # Align shapes (defensive: if dims differ, crop to min).
    H_min = min(cpp.shape[0], raw_norm.shape[0])
    W_min = min(cpp.shape[1], raw_norm.shape[1])
    a = cpp[:H_min, :W_min]
    b = raw_norm[:H_min, :W_min]
    diff = np.abs(a - b)

    print("=== Comparison (cpp vs rawpy oracle) ===")
    print(f"  compared shape: {H_min} x {W_min}")
    print(f"  max-abs-diff:   {diff.max():.6f}")
    print(f"  mean-abs-diff:  {diff.mean():.6f}")
    print(f"  P99 abs-diff:   {np.percentile(diff, 99):.6f}")
    print(f"  rms-diff:       {np.sqrt((diff ** 2).mean()):.6f}")

    # Per-CFA-plane statistics: index planes by the same (r, c) offsets the
    # oracle uses, group by color index.
    print("\n=== Per-CFA-plane mean (cpp / oracle / diff) ===")
    pat_size = cfa_pattern.shape[0]
    for r in range(pat_size):
        for c in range(pat_size):
            color = int(cfa_pattern[r, c])
            name = {0: "R", 1: "G", 2: "B", 3: "G2"}.get(color, f"C{color}")
            cpp_plane = a[r::pat_size, c::pat_size]
            ora_plane = b[r::pat_size, c::pat_size]
            d = np.abs(cpp_plane - ora_plane)
            print(f"  plane (r={r},c={c}) [{name}]: "
                  f"cpp_mean={cpp_plane.mean():.6f}  "
                  f"ora_mean={ora_plane.mean():.6f}  "
                  f"max_diff={d.max():.6f}  mean_diff={d.mean():.6f}")

    # ---- PASS / FAIL ----
    max_abs = float(diff.max())
    print()
    if max_abs < PASS_MAX_ABS:
        print(f"PASS  (max-abs-diff={max_abs:.6f} < {PASS_MAX_ABS})")
        sys.exit(0)
    else:
        print(f"FAIL  (max-abs-diff={max_abs:.6f} >= {PASS_MAX_ABS})")
        sys.exit(1)


if __name__ == "__main__":
    main()
