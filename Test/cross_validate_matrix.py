"""
Cross-validation: Phase 5 — Camera->ProPhoto matrix derivation + full custom
demosaic pipeline wiring.

Two tests:

  - Test A (Matrix EXACT ORACLE — CORRECTNESS GATE): load
    colorspace_matrices.py standalone (via importlib — it only needs numpy, NOT
    the raw_alchemy package which needs loguru/taichi). For Sample.NEF, get
    rgb_xyz_matrix (4x3) from rawpy. Call the reference
    cam_to_prophoto_matrix(xyz_to_cam_4x3) -> reference 3x3. Run the C++ matrix
    derivation (--dump-matrix PATH) and compare. PASS if max-abs relative
    error < 1e-6 (both float64-derived; the final float32 cast is well within
    this).

  - Test B (End-to-end integration SANITY): run the CLI with the custom RCD
    path (--demosaic rcd) and the dcraw/AHD path (--demosaic ahd) on
    Sample.NEF through the FULL output pipeline (--log-space F-Log2 -> 16-bit
    TIFF). Decode both outputs and compare per-channel mean/std. They won't
    match exactly (RCD vs AHD, custom matrix vs LibRaw matrix, custom
    highlight vs LibRaw blend) but PASS if per-channel means agree within ~5%
    (statistical equivalence — confirms the custom path produces correct
    ProPhoto RGB, not garbage).

Run with the Raw-Alchemy venv (has numpy + rawpy + tifffile):
    /mnt/d/GitRepos/Raw-Alchemy/.venv/bin/python \\
        /mnt/d/GitRepos/RawAlchemyCpp/Test/cross_validate_matrix.py
"""
import importlib.util
import os
import re
import subprocess
import sys

import numpy as np

REPO = "/mnt/d/GitRepos/RawAlchemyCpp"
RAW = os.path.join(REPO, "Test", "Sample.NEF")
CLI = os.path.join(REPO, "build", "raw_alchemy_cli")

CSM_PATH = "/mnt/d/GitRepos/Raw-Alchemy/src/raw_alchemy/colorspace_matrices.py"
CPP_MATRIX_PATH = "/tmp/cpp_matrix.txt"

RCD_OUT = "/tmp/rcd_out.tiff"
AHD_OUT = "/tmp/ahd_out.tiff"

# Test A: max-abs relative error threshold. Both paths compute in float64;
# the C++ result is cast to float32 at the very end (cameraToProphotoMatrix
# returns std::array<float,3,3>). float32 truncation is ~1e-7 relative, and
# the internal Gauss-Jordan/analytical-inverse math matches numpy closely, so
# 1e-6 is a generous gate.
PASS_MATRIX_MAX_REL = 1e-6

# Test B: per-channel mean agreement threshold for RCD-vs-AHD statistical
# equivalence. RCD vs AHD, custom matrix vs LibRaw, custom highlight vs
# LibRaw blend all differ; 5% confirms we're in the right ballpark (not
# garbage / not a different color space).
PASS_E2E_MEAN_PCT = 0.05


# ============================================================
#                Load the reference module standalone
# ============================================================
def load_reference_module():
    """Load colorspace_matrices.py via importlib (avoids importing the full
    raw_alchemy package, which needs loguru/taichi)."""
    spec = importlib.util.spec_from_file_location("csm_ref", CSM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================
#                       Test A (matrix exact)
# ============================================================
def run_test_a(csm_ref):
    print("=" * 64)
    print("Test A (matrix exact oracle): C++ vs reference Camera->ProPhoto")
    print("=" * 64)

    import rawpy

    # Reference: rawpy rgb_xyz_matrix -> cam_to_prophoto_matrix.
    with rawpy.imread(RAW) as raw:
        xyz_to_cam = np.array(raw.rgb_xyz_matrix, dtype=np.float64)
    ref_mat = csm_ref.cam_to_prophoto_matrix(xyz_to_cam)

    # C++: --dump-matrix PATH.
    if os.path.exists(CPP_MATRIX_PATH):
        os.remove(CPP_MATRIX_PATH)
    proc = subprocess.run(
        [CLI, RAW, "/tmp/_matrix_dummy.tiff", "--dump-matrix", CPP_MATRIX_PATH],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"C++ CLI failed:\n{proc.stderr}")
        return False
    if "MATRIX" not in proc.stdout:
        print(f"No 'MATRIX' marker in stdout:\n{proc.stdout}")
        return False

    # Parse the 3x3 floats (3 lines, whitespace-separated).
    cpp_mat = np.zeros((3, 3), dtype=np.float64)
    with open(CPP_MATRIX_PATH) as f:
        for i, line in enumerate(f):
            vals = line.split()
            if len(vals) != 3:
                print(f"Bad matrix line {i}: {line!r}")
                return False
            for j, v in enumerate(vals):
                cpp_mat[i, j] = float(v)

    np.set_printoptions(precision=10, suppress=True)
    print(f"  Reference matrix (float64):")
    print(repr(ref_mat))
    print(f"  C++ matrix (float32-promoted):")
    print(repr(cpp_mat))

    # Max-abs relative error. Guard against zero denominators.
    abs_diff = np.abs(cpp_mat - ref_mat)
    denom = np.maximum(np.abs(ref_mat), 1e-12)
    rel_err = abs_diff / denom
    max_abs = float(abs_diff.max())
    max_rel = float(rel_err.max())

    print(f"  max-abs-error:  {max_abs:.3e}")
    print(f"  max-rel-error:  {max_rel:.3e}")

    if max_rel < PASS_MATRIX_MAX_REL:
        print(f"PASS  (max-rel-error={max_rel:.3e} < {PASS_MATRIX_MAX_REL})")
        return True
    else:
        print(f"FAIL  (max-rel-error={max_rel:.3e} >= {PASS_MATRIX_MAX_REL})")
        return False


# ============================================================
#                  Test B (end-to-end sanity)
# ============================================================
def read_tiff16(path):
    """Read a 16-bit RGB TIFF as float32 [0,1] HWC via tifffile."""
    import tifffile
    arr = tifffile.imread(path)  # (H, W, 3) uint16
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"unexpected TIFF shape {arr.shape} from {path}")
    return arr.astype(np.float64) / 65535.0


def channel_stats(img):
    """Per-channel (mean, std) for an HWC float image."""
    means = img.reshape(-1, 3).mean(axis=0)
    stds = img.reshape(-1, 3).std(axis=0)
    return means, stds


def run_test_b():
    print()
    print("=" * 64)
    print("Test B (end-to-end sanity): custom RCD vs dcraw AHD -> F-Log2 TIFF")
    print("=" * 64)

    if not os.path.isfile(RAW):
        print(f"SKIP  (Sample.NEF not found at {RAW})")
        return True

    # Run both paths through the full pipeline.
    common = ["--log-space", "F-Log2"]
    for tag, out, mode in [("rcd", RCD_OUT, "rcd"), ("ahd", AHD_OUT, "ahd")]:
        proc = subprocess.run(
            [CLI, RAW, out, "--demosaic", mode] + common,
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"C++ CLI ({tag}) failed:\n{proc.stderr}")
            return False
        if not os.path.isfile(out):
            print(f"Output {out} not created for {tag}")
            return False

    rcd = read_tiff16(RCD_OUT)
    ahd = read_tiff16(AHD_OUT)

    if rcd.shape != ahd.shape:
        print(f"Shape mismatch: rcd={rcd.shape} ahd={ahd.shape}")
        return False
    print(f"  output shape: {rcd.shape}")

    rcd_mean, rcd_std = channel_stats(rcd)
    ahd_mean, ahd_std = channel_stats(ahd)

    print(f"  per-channel stats (R / G / B):")
    print(f"    rcd mean: {rcd_mean}")
    print(f"    ahd mean: {ahd_mean}")
    print(f"    rcd std:  {rcd_std}")
    print(f"    ahd std:  {ahd_std}")

    # Per-channel relative mean difference (vs ahd as reference).
    denom = np.maximum(np.abs(ahd_mean), 1e-6)
    rel_mean_diff = np.abs(rcd_mean - ahd_mean) / denom
    max_rel_mean = float(rel_mean_diff.max())

    print(f"  per-channel |rcd-ahd|/|ahd| mean diff: {rel_mean_diff}")
    print(f"  max per-channel mean rel-diff: {max_rel_mean:.4f} ({max_rel_mean*100:.2f}%)")

    if max_rel_mean < PASS_E2E_MEAN_PCT:
        print(f"PASS  (max mean rel-diff={max_rel_mean*100:.2f}% < {PASS_E2E_MEAN_PCT*100:.2f}%)")
        return True
    else:
        print(f"FAIL  (max mean rel-diff={max_rel_mean*100:.2f}% >= {PASS_E2E_MEAN_PCT*100:.2f}%)")
        return False


# ============================================================
def main():
    if not os.path.isfile(CLI):
        print(f"ERROR: {CLI} not found — build first.")
        sys.exit(2)

    csm_ref = load_reference_module()
    ok_a = run_test_a(csm_ref)
    ok_b = run_test_b()

    print()
    print("=" * 64)
    print(f"Summary:  Test A (matrix) = {'PASS' if ok_a else 'FAIL'}    "
          f"Test B (e2e) = {'PASS' if ok_b else 'FAIL'}")
    print("=" * 64)
    sys.exit(0 if (ok_a and ok_b) else 1)


if __name__ == "__main__":
    main()
