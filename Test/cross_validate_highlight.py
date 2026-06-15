"""
Cross-validation: Phase 2 highlight reconstruction (segmentation-based
"inpaint-opposed").

Two tests:
  - Test A (synthetic, deterministic): build a 64x80 RGGB Bayer mosaic in
    numpy (same gradient, same clipped block, same cam_mul as
    src/highlight_test.cpp), reimplement compute_hl_refavg +
    highlight_inpaint_opposed via scipy.ndimage, run the C++ test binary,
    compare. PASS if max-abs-diff < 0.005. This GUARANTEES the
    reconstruction path is exercised (Sample.NEF may have no natural clipping).
  - Test B (Sample.NEF, natural): decode via rawpy -> visible mosaic ->
    Phase-1 oracle (subtract_black_level + fix_hot_pixels) -> reimplemented
    highlight. Run the C++ CLI with --dump-mosaic-stage highlight, compare.
    May be ~0 (no natural clipping — valid no-op pass; Test A is the real
    validation).

Run with the Raw-Alchemy venv (has numpy + scipy + rawpy):
    /mnt/d/GitRepos/Raw-Alchemy/.venv/bin/python \\
        /mnt/d/GitRepos/RawAlchemyCpp/Test/cross_validate_highlight.py
"""
import os
import re
import subprocess
import sys

import numpy as np
from scipy import ndimage as ndi

REPO = "/mnt/d/GitRepos/RawAlchemyCpp"
RAW = os.path.join(REPO, "Test", "Sample.NEF")
CLI = os.path.join(REPO, "build", "raw_alchemy_cli")
HL_TEST = os.path.join(REPO, "build", "raw_alchemy_highlight_test")

CPP_BIN_A = "/tmp/cpp_hl.bin"
CPP_BIN_B = "/tmp/cpp_hl_real.bin"

PASS_MAX_ABS_SYNTH = 0.005    # Test A — generous for CC label ordering + float
PASS_MAX_ABS_REAL  = 0.002    # Test B — uint16 quantization (matches Phase 1)

CLIP = 0.987

# ============================================================
#           Pure-numpy reimplementation of the oracle
# ============================================================

def compute_hl_refavg(raw_data, color_map, wb_gains, raw_clips):
    """Port of math_ops.py:1182-1230 (the Taichi kernel).

    Per-pixel opposing-channel reference average + clip flag.
    """
    H, W = raw_data.shape
    refavg = np.zeros((H, W), dtype=np.float32)
    clipped = np.zeros((H, W), dtype=bool)

    # 3x3 neighborhood indices with BORDER_REPLICATE.
    ys = np.clip(np.arange(H)[:, None] + np.array([-1, 0, 1])[None, :], 0, H - 1)
    xs = np.clip(np.arange(W)[:, None] + np.array([-1, 0, 1])[None, :], 0, W - 1)
    # ys[H, 3], xs[W, 3]

    # Per-color mean: for each pixel, accumulate over its 3x3 window.
    raw_pos = np.maximum(raw_data, 0.0)

    mean = np.zeros((3, H, W), dtype=np.float32)
    cnt  = np.zeros((3, H, W), dtype=np.float32)

    for dy in range(3):
        for dx in range(3):
            ny = ys[:, dy]                         # [H]
            nx = xs[:, dx]                         # [W]
            nv = raw_pos[ny[:, None], nx[None, :]] # [H, W]
            nc = color_map[ny[:, None], nx[None, :]]  # [H, W]
            for c in range(3):
                mask = (nc == c)
                mean[c] += np.where(mask, nv, 0.0)
                cnt[c]  += mask.astype(np.float32)

    cbrt_mean = np.zeros((3, H, W), dtype=np.float32)
    for c in range(3):
        with np.errstate(invalid="ignore", divide="ignore"):
            arg = wb_gains[c] * mean[c] / np.maximum(cnt[c], 1e-30)
        arg = np.where(cnt[c] > 0, arg, 0.0)
        arg = np.maximum(arg, 0.0)
        cbrt_mean[c] = np.where(cnt[c] > 0, np.cbrt(arg), 0.0)

    color = color_map  # [H, W] in {0,1,2}
    opp_cbrt = np.zeros((H, W), dtype=np.float32)
    for c in range(3):
        # Sum of cbrt_mean over the OTHER two colors, then halve.
        others = [k for k in range(3) if k != c]
        opp_cbrt_c = 0.5 * (cbrt_mean[others[0]] + cbrt_mean[others[1]])
        opp_cbrt = np.where(color == c, opp_cbrt_c, opp_cbrt)

    ref = opp_cbrt ** 3
    ref = np.where(wb_gains[color] > 1e-6,
                   ref / np.maximum(wb_gains[color], 1e-6),
                   ref)
    refavg = ref.astype(np.float32)

    # is_clipped = val >= raw_clips[color]
    val = raw_data
    thr = raw_clips[color]
    clipped = (val >= thr)

    return refavg, clipped


def highlight_inpaint_opposed(raw_data, color_map, wb):
    """Port of core.py:53-136.

    `wb` is the 4-element camera WB array (only [0],[1],[2] used).
    Modifies raw_data in place.
    """
    H, W = raw_data.shape
    g = max(float(wb[1]), 1e-6)

    wb_gains = np.array([wb[0] / g, 1.0, wb[2] / g], dtype=np.float32)
    raw_clips = np.array([CLIP / max(wg, 1e-6) for wg in wb_gains],
                          dtype=np.float32)

    refavg, clipped = compute_hl_refavg(raw_data, color_map, wb_gains, raw_clips)
    if not np.any(clipped):
        return 0  # no-op; report 0 clipped pixels

    diff = raw_data - refavg

    # 7x7 flat rect structuring element.
    se = np.ones((7, 7), dtype=bool)

    for c in range(3):
        clipped_c = clipped & (color_map == c)
        if not np.any(clipped_c):
            continue

        # Morphological close (dilate then erode), border_value=0 for BOTH.
        # scipy.binary_closing passes border_value to both internal calls.
        closed = ndi.binary_closing(clipped_c, structure=se, border_value=0)

        # 8-connectivity CC.
        labels, num_seg_plus_one = ndi.label(closed, structure=np.ones((3, 3)))
        num_seg = num_seg_plus_one  # ndi.label returns num_features (= max label)
        if num_seg == 0:
            continue

        # 7x7 grey dilation (max-filter) on the label image as float.
        # OOB treated as 0 (default mode='constant', cval=0.0).
        expanded = ndi.grey_dilation(labels.astype(np.float32),
                                     size=(7, 7), mode='constant', cval=0.0)

        lo = float(raw_clips[c]) * 0.2
        unclipped_valid = (color_map == c) & (~clipped) & (raw_data > lo)

        border = (expanded > 0) & (labels == 0) & unclipped_valid
        border_labels = expanded[border].astype(np.int64)
        border_diffs = diff[border]

        seg_sum = np.bincount(border_labels, weights=border_diffs,
                              minlength=num_seg + 1).astype(np.float64)
        seg_cnt = np.bincount(border_labels, minlength=num_seg + 1)

        global_chroma = 0.0
        total_cnt = int(seg_cnt[1:].sum())
        if total_cnt > 100:
            global_chroma = float(seg_sum[1:].sum()) / total_cnt

        seg_chroma = np.where(seg_cnt > 10,
                              seg_sum / np.maximum(seg_cnt, 1),
                              global_chroma).astype(np.float32)

        target = clipped_c & (labels > 0)
        target_labels = labels[target]
        raw_data[target] = np.maximum(
            raw_data[target],
            refavg[target] + seg_chroma[target_labels]
        )

    return int(np.sum(clipped))


# ============================================================
#                  Phase-1 oracle primitives
# ============================================================
def subtract_black_level(sensor_raw, bl, wl, cfa_pattern):
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
    pat_size = cfa_pattern.shape[0]
    for r in range(pat_size):
        for c in range(pat_size):
            plane = raw_norm[r::pat_size, c::pat_size]
            med = ndi.median_filter(plane, size=3, mode='nearest')
            diff = np.abs(plane - med)
            std = max(np.std(diff), 1e-6)
            hot = diff > threshold * std
            plane[hot] = med[hot]


def build_color_map(cfa_pattern, H, W):
    """Tile CFA pattern, collapse >=3 to 1 (X-Trans secondary greens / Bayer G2)."""
    pat_size = cfa_pattern.shape[0]
    color_map = np.tile(cfa_pattern,
                        ((H + pat_size - 1) // pat_size,
                         (W + pat_size - 1) // pat_size))[:H, :W]
    color_map = np.where(color_map >= 3, 1, color_map).astype(np.int32)
    return color_map


# ============================================================
#                       Test A (synthetic)
# ============================================================
def build_synthetic_mosaic():
    """Mirror src/highlight_test.cpp::buildSyntheticMosaic EXACTLY."""
    H, W = 64, 80
    filters = 0x94949494  # RGGB Bayer
    cam_mul = np.array([2.0, 1.0, 1.5, 1.0], dtype=np.float32)
    cfa_pattern = np.array([[0, 1], [3, 2]], dtype=np.int32)  # RGGB raw_pattern

    data = np.zeros((H, W), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            t = float(y * W + x) / float(H * W)
            wave = 0.04 * np.sin(0.15 * y) * np.cos(0.12 * x)
            data[y, x] = 0.05 + 0.25 * t + wave

    # 12x12 clipped block at 1.0.
    data[20:32, 30:42] = 1.0

    # Scattered clipped pixels.
    data[5, 7] = 1.0
    data[50, 60] = 1.0
    data[10, 70] = 1.0
    data[55, 15] = 1.0

    return data, cfa_pattern, cam_mul, filters


def run_test_a():
    print("=" * 64)
    print("Test A (synthetic 64x80 RGGB): reconstruction path cross-validation")
    print("=" * 64)

    data, cfa_pattern, cam_mul, filters = build_synthetic_mosaic()
    H, W = data.shape
    color_map = build_color_map(cfa_pattern, H, W)

    # Phase-1 preprocess (no-op since cblack=0, maximum=1, but follow the path).
    bl = np.array([0, 0, 0, 0], dtype=np.float32)
    wl = 1.0
    data = subtract_black_level(data, bl, wl, cfa_pattern)
    fix_hot_pixels(data, cfa_pattern)

    n_clipped = highlight_inpaint_opposed(data, color_map, cam_mul)

    # Run the C++ test binary.
    proc = subprocess.run([HL_TEST, CPP_BIN_A], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"C++ test binary failed:\n{proc.stderr}")
        return False
    m = re.match(r"^\s*(\d+)\s+(\d+)\s*$", proc.stdout.strip())
    if not m:
        print(f"Could not parse H W from C++ stdout: {proc.stdout!r}")
        return False
    H_cpp, W_cpp = int(m.group(1)), int(m.group(2))
    if (H_cpp, W_cpp) != (H, W):
        print(f"Dimension mismatch: cpp=({H_cpp},{W_cpp}) oracle=({H},{W})")
        return False

    cpp = np.fromfile(CPP_BIN_A, dtype="<f4").reshape(H_cpp, W_cpp).astype(np.float32)

    diff = np.abs(cpp - data)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    n_changed = int(np.sum(diff > 1e-6))

    print(f"  clipped pixels:           {n_clipped}")
    print(f"  pixels changed by recon:  {n_changed}")
    print(f"  max-abs-diff (cpp vs py): {max_abs:.6f}")
    print(f"  mean-abs-diff:            {mean_abs:.6f}")

    if max_abs < PASS_MAX_ABS_SYNTH:
        print(f"PASS  (max-abs-diff={max_abs:.6f} < {PASS_MAX_ABS_SYNTH})")
        return True
    else:
        print(f"FAIL  (max-abs-diff={max_abs:.6f} >= {PASS_MAX_ABS_SYNTH})")
        # Diagnostic dump for debugging.
        np.save("/tmp/py_hl_test_a.npy", data)
        np.save("/tmp/cpp_hl_test_a.npy", cpp)
        return False


# ============================================================
#                       Test B (Sample.NEF)
# ============================================================
def run_test_b():
    print()
    print("=" * 64)
    print(f"Test B ({os.path.basename(RAW)}): natural-clipping cross-validation")
    print("=" * 64)

    import rawpy
    with rawpy.imread(RAW) as raw:
        sensor_raw = raw.raw_image_visible.astype(np.float32)
        bl = np.array(raw.black_level_per_channel, dtype=np.float32)
        wl = float(raw.white_level)
        cfa_pattern = raw.raw_pattern.copy()
        cam_mul = np.array(raw.camera_whitebalance, dtype=np.float32)

    print(f"  mosaic shape:  {sensor_raw.shape}")
    print(f"  cam_mul:       {cam_mul.tolist()}")
    print(f"  white_level:   {wl}")

    H, W = sensor_raw.shape
    color_map = build_color_map(cfa_pattern, H, W)

    data = subtract_black_level(sensor_raw, bl, wl, cfa_pattern)
    fix_hot_pixels(data, cfa_pattern)
    n_clipped = highlight_inpaint_opposed(data, color_map, cam_mul)

    proc = subprocess.run(
        [CLI, RAW, "/tmp/dummy.tiff",
         "--dump-mosaic-stage", "highlight",
         "--dump-mosaic", CPP_BIN_B],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"C++ CLI failed:\n{proc.stderr}")
        return False
    m = re.search(r"^MOSAIC\s+(\d+)\s+(\d+)\s*$", proc.stdout, re.MULTILINE)
    if not m:
        print(f"Could not parse 'MOSAIC W H' from stdout:\n{proc.stdout}")
        return False
    W_cpp, H_cpp = int(m.group(1)), int(m.group(2))
    print(f"  C++ mosaic:    {W_cpp} x {H_cpp}")

    H_min = min(H_cpp, H)
    W_min = min(W_cpp, W)
    cpp = np.fromfile(CPP_BIN_B, dtype="<u2").reshape(H_cpp, W_cpp).astype(np.float32) / 65535.0
    diff = np.abs(cpp[:H_min, :W_min] - data[:H_min, :W_min])
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())

    print(f"  # clipped pixels (oracle): {n_clipped}")
    print(f"  max-abs-diff:              {max_abs:.6f}")
    print(f"  mean-abs-diff:             {mean_abs:.6f}")

    if max_abs < PASS_MAX_ABS_REAL:
        print(f"PASS  (max-abs-diff={max_abs:.6f} < {PASS_MAX_ABS_REAL})")
        return True
    else:
        print(f"FAIL  (max-abs-diff={max_abs:.6f} >= {PASS_MAX_ABS_REAL})")
        return False


# ============================================================
def main():
    if not os.path.isfile(HL_TEST):
        print(f"ERROR: {HL_TEST} not found — build first.")
        sys.exit(2)
    if not os.path.isfile(CLI):
        print(f"ERROR: {CLI} not found — build first.")
        sys.exit(2)

    ok_a = run_test_a()
    ok_b = run_test_b()

    print()
    print("=" * 64)
    print(f"Summary:  Test A = {'PASS' if ok_a else 'FAIL'}    "
          f"Test B = {'PASS' if ok_b else 'FAIL'}")
    print("=" * 64)
    sys.exit(0 if (ok_a and ok_b) else 1)


if __name__ == "__main__":
    main()
