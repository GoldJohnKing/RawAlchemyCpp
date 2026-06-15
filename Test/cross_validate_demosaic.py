"""
Cross-validation: Phase 3 RCD (Ratio Corrected Demosaicing).

Three tests:
  - Test A (synthetic, known-truth — CORRECTNESS GATE): build a smooth 256x256
    RGB gradient in numpy, subsample it to RGGB Bayer the same way
    src/rcd_test.cpp does, run the C++ test binary, read the demosaiced
    float32 RGB, and compare to the ORIGINAL (un-mosaiced) gradient.
    PASS if max-abs-diff < 0.03 (RCD reconstructs smooth content within
    ~1-2%; 0.03 gives margin for the 4-px bilinear border).
    This validates algorithm correctness INDEPENDENTLY of the Taichi
    reference (which needs a GPU and is unavailable here).
  - Test B (high-freq edges — DIRECTION-SELECTION GATE): for each of three
    step-edge patterns (edge-h, edge-v, edge-d), run the C++ test binary with
    the pattern name and assert STRUCTURAL sanity (no exact truth compare —
    demosaicing is imperfect at edges; the goal is to catch direction-
    selection bugs). Checks: edge contrast preserved, edge position
    localized, no zipper artifact (catches V/H swap bugs), color balance on
    the diagonal edge (catches P/Q swap bugs).
    RATIONALE: the smooth gradient in Test A is DEGENERATE for RCD's
    direction-selection logic — VH_Disc and PQ_Disc are all ~0.5 on smooth
    data, so a V/H or P/Q swap bug is invisible there. The high-frequency
    edges here are the only thing that exercises that logic. (Stage-review
    Q4 found the original validation insufficient for exactly this reason.)
  - Test B-smoke (Sample.NEF — SMOKE TEST): run the CLI with --dump-demosaic
    on a real 45 MP image. Confirm it completes without crash/OOM, prints
    'DEMOSAIC W H', and the output is finite and >= 0. Report per-channel
    min/mean/max. No rawpy comparison (different demosaic algo + WB/matrix
    mismatch). PASS if it completes and values are finite and >= 0.

Run with the Raw-Alchemy venv (has numpy):
    /mnt/d/GitRepos/Raw-Alchemy/.venv/bin/python \\
        /mnt/d/GitRepos/RawAlchemyCpp/Test/cross_validate_demosaic.py
"""
import os
import re
import subprocess
import sys

import numpy as np

REPO = "/mnt/d/GitRepos/RawAlchemyCpp"
RAW = os.path.join(REPO, "Test", "Sample.NEF")
CLI = os.path.join(REPO, "build", "raw_alchemy_cli")
RCD_TEST = os.path.join(REPO, "build", "raw_alchemy_rcd_test")

CPP_BIN_A = "/tmp/cpp_rcd.bin"
CPP_BIN_B = "/tmp/cpp_rcd_real.bin"

# Test A — RCD reconstructs smooth content within ~1-2%; 0.03 gives margin
# for the 4-px bilinear border.
PASS_MAX_ABS_SYNTH = 0.03

# Test B structural-check thresholds (Bayer CFA period is 2, so direction
# selection should be much cleaner than X-Trans — thresholds are tighter
# than Test/cross_validate_demosaic_xtrans.py).
EDGE_CONTRAST_MIN = 0.5       # |bright_mean - dark_mean| across the edge
EDGE_POSITION_TOL = 2         # px from col=128 / row=128 (interior idx center)
ZIPPER_MAX = 0.05             # max adj-px oscillation ALONG edge, far from edge
DIAG_STD_MIN = 0.3            # diagonal edge reproduced (not smeared flat)
DIAG_CH_MEANDIFF_MAX = 0.05   # per-channel mean diff for grey diagonal edge

FILTERS_RGGB = 0x94949494


# ============================================================
#                  CFA helper (mirror of C++)
# ============================================================
def fc(row, col, filters=FILTERS_RGGB):
    """dcraw/LibRaw FC macro. Returns 0=R, 1=G, 2=B for standard Bayer codes."""
    return (filters >> (((((row << 1) & 14) + (col & 1)) << 1))) & 3


# ============================================================
#                       Test A (synthetic)
# ============================================================
def build_synthetic_gradient():
    """Mirror src/rcd_test.cpp::buildSyntheticRggbMosaic EXACTLY (smooth case).

    Builds a 256x256 RGB gradient, subsamples to RGGB Bayer.
    Returns (bayer_2d, truth_rgb).
    """
    H = W = 256
    cols = np.arange(W, dtype=np.float32)
    rows = np.arange(H, dtype=np.float32)
    R = np.broadcast_to(cols[None, :] / 255.0, (H, W)).astype(np.float32)   # [H, W]
    G = np.broadcast_to(rows[:, None] / 255.0, (H, W)).astype(np.float32)   # [H, W]
    B = (rows[:, None] + cols[None, :]) / 510.0
    B = np.ascontiguousarray(B, dtype=np.float32)                           # [H, W]
    truth = np.stack([R, G, B], axis=-1)                                    # [H, W, 3]

    # Subsample to RGGB: keep only the CFA color's value at each pixel.
    bayer = np.zeros((H, W), dtype=np.float32)
    for row in range(H):
        for col in range(W):
            c = fc(row, col)
            if c == 0:
                bayer[row, col] = R[row, col]
            elif c == 1:
                bayer[row, col] = G[row, col]
            else:  # c == 2
                bayer[row, col] = B[row, col]
    return bayer, truth


def run_test_a():
    print("=" * 64)
    print("Test A (synthetic 256x256 RGGB): RCD known-truth cross-validation")
    print("=" * 64)

    bayer, truth = build_synthetic_gradient()
    H, W = bayer.shape

    # Run the C++ test binary. Legacy single-arg form (out path only) is
    # equivalent to the new two-arg form with pattern="smooth".
    proc = subprocess.run([RCD_TEST, CPP_BIN_A], capture_output=True, text=True)
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

    cpp = np.fromfile(CPP_BIN_A, dtype="<f4").reshape(H_cpp, W_cpp, 3).astype(np.float32)

    # Compare to the ORIGINAL (un-mosaiced) gradient — this is the whole
    # point: RCD should reconstruct the ground truth from the mosaic.
    diff = np.abs(cpp - truth)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    per_channel_mean = [float(diff[:, :, c].mean()) for c in range(3)]

    print(f"  gradient shape:           {H}x{W}")
    print(f"  max-abs-diff (cpp vs truth): {max_abs:.6f}")
    print(f"  mean-abs-diff:               {mean_abs:.6f}")
    print(f"  per-channel mean-abs-diff:   "
          f"R={per_channel_mean[0]:.6f}  "
          f"G={per_channel_mean[1]:.6f}  "
          f"B={per_channel_mean[2]:.6f}")

    if max_abs < PASS_MAX_ABS_SYNTH:
        print(f"PASS  (max-abs-diff={max_abs:.6f} < {PASS_MAX_ABS_SYNTH})")
        return True
    else:
        print(f"FAIL  (max-abs-diff={max_abs:.6f} >= {PASS_MAX_ABS_SYNTH})")
        np.save("/tmp/rcd_truth.npy", truth)
        np.save("/tmp/rcd_cpp.npy", cpp)
        print("  (diagnostic dumps: /tmp/rcd_truth.npy, /tmp/rcd_cpp.npy)")
        return False


# ============================================================
#                  Test B (high-freq edges)
# ============================================================
def _run_pattern(pattern, out_path):
    """Run raw_alchemy_rcd_test {pattern} {out_path}; return (H, W) or None."""
    proc = subprocess.run(
        [RCD_TEST, pattern, out_path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"  C++ test binary failed for '{pattern}':\n{proc.stderr}")
        return None
    m = re.match(r"^\s*(\d+)\s+(\d+)\s*$", proc.stdout.strip())
    if not m:
        print(f"  Could not parse H W from stdout: {proc.stdout!r}")
        return None
    return int(m.group(1)), int(m.group(2))


def _load_rgb(path, H, W):
    return np.fromfile(path, dtype="<f4").reshape(H, W, 3).astype(np.float32)


def _step_edge_checks(cpp, axis, label):
    """Structural sanity checks for a step edge.

    axis=1 -> edge-h: vertical edge line at col=128 (column discontinuity;
              the edge LINE is vertical). Tests horizontal-direction demosaic.
    axis=0 -> edge-v: horizontal edge line at row=128. Tests vertical-direction
              demosaic.

    Checks (all on the interior, skipping the 4-px bilinear border + margin):
      1. finite + non-negative.
      2. contrast: |bright_mean - dark_mean| >= EDGE_CONTRAST_MIN (the step
         is preserved, not blurred away).
      3. edge position: the steepest transition in the perpendicular profile
         is within +-EDGE_POSITION_TOL of the image center (col=128 / row=128).
      4. zipper: max adjacent-pixel oscillation ALONG the edge-line direction,
         in the far-from-edge region. A V/H direction-selection swap bug
         produces a high-frequency zipper (sawtooth) artifact along the edge
         line; a correct RCD implementation has near-zero oscillation here.
         The smooth Test A is blind to this — VH_Disc is ~0.5 everywhere on
         smooth data, so a swap bug doesn't change the output. ONLY a high-
         frequency edge exercises V/H selection, which is why this test exists.
    """
    b = 6  # RCD border is 4 px; +2 px margin
    interior = cpp[b:-b, b:-b, :]
    H_int, W_int, _ = interior.shape
    # Interior index center maps back to image col/row 128:
    #   interior idx i  <->  image idx (i + b);  center_int = (W_int // 2)
    #   -> image idx = b + W_int//2 = 6 + 122 = 128 (for 256-wide image).
    center = (H_int if axis == 0 else W_int) // 2
    ok = True

    # 1. finite + non-negative.
    finite = bool(np.all(np.isfinite(interior)))
    nonneg = bool(np.all(interior >= -1e-6))
    if not (finite and nonneg):
        print(f"  [{label}] finite={finite} nonneg={nonneg}  FAIL")
        return False

    # 2-3. contrast + edge position. Profile = average over the axis PARALLEL
    # to the edge line (collapse the edge-line direction + channels),
    # producing a 1-D profile PERPENDICULAR to the edge.
    if axis == 0:
        profile = interior.mean(axis=(1, 2))  # avg over cols + channels -> [H_int]
    else:
        profile = interior.mean(axis=(0, 2))  # avg over rows + channels -> [W_int]
    dark_mean = float(profile[:center].mean())
    bright_mean = float(profile[center:].mean())
    contrast = abs(bright_mean - dark_mean)
    edge_pos = int(np.argmax(np.abs(np.diff(profile))))
    dist = abs(edge_pos - center)
    c_ok = contrast >= EDGE_CONTRAST_MIN
    p_ok = dist <= EDGE_POSITION_TOL
    print(f"  [{label}] contrast={contrast:.3f} ({'OK' if c_ok else 'BAD'}, "
          f"min={EDGE_CONTRAST_MIN}); dark_mean={dark_mean:.3f} "
          f"bright_mean={bright_mean:.3f}")
    print(f"  [{label}] edge_pos@int_idx={edge_pos} (center={center}, "
          f"dist={dist}, {'OK' if p_ok else 'BAD'}, tol={EDGE_POSITION_TOL})")
    ok &= c_ok and p_ok

    # 4. zipper: in the far-from-edge region (|perp_coord - center| > 20),
    # measure max adjacent-pixel diff ALONG the edge-line direction, per
    # channel. For edge-h (axis=1) the edge line runs vertically (along
    # rows), so we look at far-from-edge columns and measure adjacent-row
    # differences. For edge-v (axis=0) the edge line runs horizontally
    # (along cols), so we look at far-from-edge rows and measure adjacent-
    # column differences.
    far = np.ones(profile.shape[0], dtype=bool)
    far[max(0, center - 20):center + 20] = False
    if axis == 0:
        slab = interior[far, :, :]               # [n_far, W_int, 3]
        adj_diff = np.abs(np.diff(slab, axis=1))  # along cols (edge-line dir)
    else:
        slab = interior[:, far, :]                # [H_int, n_far, 3]
        adj_diff = np.abs(np.diff(slab, axis=0))  # along rows (edge-line dir)
    zipper_max = float(adj_diff.max()) if adj_diff.size > 0 else 0.0
    z_ok = zipper_max < ZIPPER_MAX
    print(f"  [{label}] zipper_max={zipper_max:.6f} "
          f"({'OK' if z_ok else 'BAD'}, tol={ZIPPER_MAX})")
    ok &= z_ok

    return bool(ok)


def _diagonal_checks(cpp, label="edge-d"):
    """Structural sanity for the 45-degree diagonal edge.

    The diagonal edge passes through every row and column, so we cannot
    average it down to a 1-D profile (unlike the axis-aligned edges). Instead:
      1. finite + non-negative.
      2. spatial std > DIAG_STD_MIN (the edge was reproduced, not smeared
         flat — a P/Q direction-selection failure often smears the diagonal).
      3. per-channel mean diff < DIAG_CH_MEANDIFF_MAX: the diagonal edge is
         color-balanced (R=G=B in the input), so the per-channel means of the
         output must agree. A P/Q swap bug breaks diagonal discrimination
         asymmetrically across channels.
    """
    b = 6
    interior = cpp[b:-b, b:-b, :]
    finite = bool(np.all(np.isfinite(interior)))
    nonneg = bool(np.all(interior >= -1e-6))
    if not (finite and nonneg):
        print(f"  [{label}] finite={finite} nonneg={nonneg}  FAIL")
        return False
    std = float(interior.std())
    ch_means = [float(interior[:, :, c].mean()) for c in range(3)]
    ch_meandiff = max(
        abs(ch_means[0] - ch_means[1]),
        abs(ch_means[1] - ch_means[2]),
        abs(ch_means[0] - ch_means[2]),
    )
    s_ok = std > DIAG_STD_MIN
    d_ok = ch_meandiff < DIAG_CH_MEANDIFF_MAX
    print(f"  [{label}] std={std:.4f} ({'OK' if s_ok else 'BAD'}, "
          f"min={DIAG_STD_MIN})")
    print(f"  [{label}] ch_means=R={ch_means[0]:.4f} "
          f"G={ch_means[1]:.4f} B={ch_means[2]:.4f}")
    print(f"  [{label}] ch_meandiff={ch_meandiff:.5f} "
          f"({'OK' if d_ok else 'BAD'}, tol={DIAG_CH_MEANDIFF_MAX})")
    return bool(s_ok and d_ok)


def run_test_b():
    print()
    print("=" * 64)
    print("Test B (high-freq edges): RCD direction-selection structural gate")
    print("=" * 64)
    print("  (the smooth Test A is degenerate for VH_Disc/PQ_Disc direction")
    print("   selection — only high-frequency edges exercise that logic.)")
    all_ok = True

    # (a) edge-h: vertical edge line at col=128 -> tests horizontal demosaic.
    dims = _run_pattern("edge-h", "/tmp/rcd_edge-h.bin")
    if dims is None:
        return False
    cpp = _load_rgb("/tmp/rcd_edge-h.bin", dims[0], dims[1])
    all_ok &= _step_edge_checks(cpp, axis=1, label="edge-h")

    # (b) edge-v: horizontal edge line at row=128 -> tests vertical demosaic.
    dims = _run_pattern("edge-v", "/tmp/rcd_edge-v.bin")
    if dims is None:
        return False
    cpp = _load_rgb("/tmp/rcd_edge-v.bin", dims[0], dims[1])
    all_ok &= _step_edge_checks(cpp, axis=0, label="edge-v")

    # (c) edge-d: 45-degree diagonal -> tests P/Q diagonal discrimination.
    dims = _run_pattern("edge-d", "/tmp/rcd_edge-d.bin")
    if dims is None:
        return False
    cpp = _load_rgb("/tmp/rcd_edge-d.bin", dims[0], dims[1])
    all_ok &= _diagonal_checks(cpp, label="edge-d")

    if all_ok:
        print("PASS  (edges reproduced; no zipper/color-balance artifacts)")
    else:
        print("FAIL  (structural sanity check failed)")
    return bool(all_ok)


# ============================================================
#                  Test B-smoke (Sample.NEF)
# ============================================================
def run_test_b_smoke():
    print()
    print("=" * 64)
    print(f"Test B-smoke ({os.path.basename(RAW)}): "
          f"RCD smoke test on real 45 MP image")
    print("=" * 64)

    if not os.path.isfile(RAW):
        print(f"SKIP  (Sample.NEF not found at {RAW})")
        return True  # not a failure — just nothing to test

    proc = subprocess.run(
        [CLI, RAW, "/tmp/dummy_demosaic.tiff",
         "--dump-demosaic", CPP_BIN_B],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"C++ CLI failed (returncode={proc.returncode}):\n{proc.stderr}")
        return False

    m = re.search(r"^DEMOSAIC\s+(\d+)\s+(\d+)\s*$", proc.stdout, re.MULTILINE)
    if not m:
        print(f"Could not parse 'DEMOSAIC W H' from stdout:\n{proc.stdout}")
        return False
    W_cpp, H_cpp = int(m.group(1)), int(m.group(2))
    print(f"  C++ demosaic output: {W_cpp} x {H_cpp}")

    cpp = np.fromfile(CPP_BIN_B, dtype="<f4").reshape(H_cpp, W_cpp, 3)
    finite = bool(np.all(np.isfinite(cpp)))
    nonneg = bool(np.all(cpp >= 0.0))

    per_channel = []
    for c, name in enumerate("RGB"):
        ch = cpp[:, :, c]
        per_channel.append((name, float(ch.min()), float(ch.mean()), float(ch.max())))

    print(f"  finite: {finite}   non-negative: {nonneg}")
    for name, lo, mu, hi in per_channel:
        print(f"  {name}: min={lo:.6f}  mean={mu:.6f}  max={hi:.6f}")

    if finite and nonneg:
        print(f"PASS  (completed; values finite and >= 0)")
        return True
    else:
        print(f"FAIL  (finite={finite} nonneg={nonneg})")
        return False


# ============================================================
def main():
    if not os.path.isfile(RCD_TEST):
        print(f"ERROR: {RCD_TEST} not found — build first.")
        sys.exit(2)

    ok_a = run_test_a()
    ok_b = run_test_b()

    # Test B-smoke depends on the Phase-5 CLI (raw_alchemy_cli), which may
    # not be built yet if Phase 5 has the tree half-edited. Soft-skip with a
    # clear warning so Test A + Test B (Phase-3 scope) can still be verified.
    ok_b_smoke = True
    if not os.path.isfile(CLI):
        print()
        print("=" * 64)
        print(f"Test B-smoke: SKIP  ({CLI} not built — Phase 5 in progress?)")
        print("=" * 64)
        print("  (Test A + Test B are Phase-3 scope and still ran above.)")
    else:
        ok_b_smoke = run_test_b_smoke()

    print()
    print("=" * 64)
    print(f"Summary:  Test A = {'PASS' if ok_a else 'FAIL'}    "
          f"Test B = {'PASS' if ok_b else 'FAIL'}    "
          f"Test B-smoke = {'PASS' if ok_b_smoke else 'FAIL'}")
    print("=" * 64)
    sys.exit(0 if (ok_a and ok_b and ok_b_smoke) else 1)


if __name__ == "__main__":
    main()
