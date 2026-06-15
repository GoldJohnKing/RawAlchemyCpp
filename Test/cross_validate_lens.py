"""
Cross-validation: Phase 6 lens-correction edge robustness.

Two-tier validation:

  PRIMARY (synthetic, deterministic): run ./build/raw_alchemy_lens_test, which
  constructs synthetic coord buffers with known OOB / near-border cases and
  asserts the two-stage safety scale + uniform OOB mask behaves per
  lensfun_wrapper.py:851-903. PASS if all 4 cases (auto-crop, interp_margin,
  uniform anti-fringing, identity) report PASS. This GUARANTEES the
  postProcessLensCoords path is exercised regardless of Lensfun DB state.

  SECONDARY (Sample.NEF smoke): run the C++ CLI on Sample.NEF with lens
  correction ON and OFF, and check:
    - lens correction either applied (DB has Nikon Z 8 + lens) or was a clean
      no-op (lens not found) — both are acceptable,
    - if applied: no black borders introduced at image edges (the uniform OOB
      mask + stage-A/B must not mass-zero borders) and no per-channel color
      fringing at edges (R ~= G ~= B where the mask would have fired).
  If the Lensfun DB lacks the camera/lens (correction is a no-op), we report
  that and rely on the synthetic test (PRIMARY).

Legacy mode: with two TIFF args, runs the original before/after comparison.

Usage:
  python3 cross_validate_lens.py                          # synthetic + smoke
  python3 cross_validate_lens.py <off.tif> <on.tif>       # legacy comparison

Run with the Raw-Alchemy venv:
    /mnt/d/GitRepos/Raw-Alchemy/.venv/bin/python \\
        /mnt/d/GitRepos/RawAlchemyCpp/Test/cross_validate_lens.py
"""
import os
import re
import struct
import subprocess
import sys

REPO = "/mnt/d/GitRepos/RawAlchemyCpp"
RAW = os.path.join(REPO, "Test", "Sample.NEF")
CLI = os.path.join(REPO, "build", "raw_alchemy_cli")
LENS_TEST = os.path.join(REPO, "build", "raw_alchemy_lens_test")

# Edge-quality thresholds for the smoke test.
BLACK_BORDER_MAX_MEAN = 0.02   # <2% of white → effectively black border
FRINGE_MAX_CHANNEL_SPREAD = 0.20  # max |c1-c2|/max(c) at a border pixel
BORDER_PIXELS = 3               # interp_margin = 2; check a 3px rim


# ============================================================
#           TIFF reader (16-bit RGB, uncompressed)
# ============================================================

def read_tiff_pixels(path):
    """Read a simple 16-bit RGB TIFF file; return (width, height, pixels).

    pixels is a flat list of uint16 values in RGB order.
    """
    with open(path, "rb") as f:
        byte_order = f.read(2)
        if byte_order == b"II":
            endian = "<"
        elif byte_order == b"MM":
            endian = ">"
        else:
            raise ValueError(f"Not a TIFF file: {path}")

        magic = struct.unpack(endian + "H", f.read(2))[0]
        assert magic == 42, f"Bad TIFF magic: {magic}"
        ifd_offset = struct.unpack(endian + "I", f.read(4))[0]

        f.seek(ifd_offset)
        num_entries = struct.unpack(endian + "H", f.read(2))[0]

        tags = {}
        for _ in range(num_entries):
            tag_id = struct.unpack(endian + "H", f.read(2))[0]
            type_id = struct.unpack(endian + "H", f.read(2))[0]
            count = struct.unpack(endian + "I", f.read(4))[0]
            value_raw = f.read(4)
            type_sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8}
            val_size = type_sizes.get(type_id, 1) * count
            if val_size <= 4:
                if type_id == 3:
                    tags[tag_id] = struct.unpack(endian + "H", value_raw[:2])[0]
                elif type_id == 4:
                    tags[tag_id] = struct.unpack(endian + "I", value_raw)[0]
                elif type_id == 1:
                    tags[tag_id] = value_raw[0]
            else:
                offset = struct.unpack(endian + "I", value_raw)[0]
                tags[tag_id] = ("ptr", offset, count, type_id)

        width = tags.get(256, 0)
        height = tags.get(257, 0)
        bits_per_sample = tags.get(258, 16)
        compression = tags.get(259, 1)
        samples_per_pixel = tags.get(277, 3)
        strip_offsets = tags.get(273)
        strip_byte_counts = tags.get(279)

        assert width > 0 and height > 0, f"Invalid dimensions: {width}x{height}"
        assert bits_per_sample == 16, f"Expected 16-bit, got {bits_per_sample}"
        assert compression == 1, f"Expected uncompressed, got compression={compression}"
        assert samples_per_pixel == 3, f"Expected RGB (3), got {samples_per_pixel}"

        if isinstance(strip_offsets, tuple):
            _, soff, scount, _ = strip_offsets
            f.seek(soff)
            strip_off_list = [struct.unpack(endian + "I", f.read(4))[0] for _ in range(scount)]
        else:
            strip_off_list = [strip_offsets]

        if isinstance(strip_byte_counts, tuple):
            _, sboff, sbcount, _ = strip_byte_counts
            f.seek(sboff)
            strip_bc_list = [struct.unpack(endian + "I", f.read(4))[0] for _ in range(sbcount)]
        else:
            strip_bc_list = [strip_byte_counts]

        all_data = bytearray()
        for off, bc in zip(strip_off_list, strip_bc_list):
            f.seek(off)
            all_data.extend(f.read(bc))

        num_pixels = width * height * 3
        fmt = endian + ("H" * num_pixels)
        pixels = struct.unpack(fmt, bytes(all_data[: num_pixels * 2]))
        return width, height, pixels


def pixel_at(pixels, width, row, col, channel):
    return pixels[(row * width + col) * 3 + channel]


# ============================================================
#           PRIMARY: synthetic coord test
# ============================================================

def run_synthetic_test():
    print("=" * 64)
    print("PRIMARY: synthetic postProcessLensCoords test")
    print("=" * 64)
    if not os.path.isfile(LENS_TEST):
        print(f"  ERROR: {LENS_TEST} not found — build first.")
        return False

    proc = subprocess.run([LENS_TEST], capture_output=True, text=True)
    print(proc.stdout.rstrip())
    if proc.stderr.strip():
        print("STDERR:", proc.stderr.rstrip())

    if proc.returncode != 0:
        print("FAIL  (test binary exited non-zero)")
        return False

    # The binary prints "RESULT: PASS" on success.
    if "RESULT: PASS" in proc.stdout:
        print("\nPRIMARY: PASS (all 4 synthetic cases passed)")
        return True
    print("\nPRIMARY: FAIL")
    return False


# ============================================================
#           SECONDARY: Sample.NEF smoke (edge robustness)
# ============================================================

def _run_cli(output_path, lens_on):
    """Run the CLI on Sample.NEF; return (ok, stdout)."""
    args = [CLI, RAW, output_path, "--log-space", "F-Log2C", "--exposure", "0"]
    if lens_on:
        args.append("--lens-correction")
    else:
        args.append("--no-lens-correction")
    proc = subprocess.run(args, capture_output=True, text=True)
    return proc.returncode == 0, proc.stdout + proc.stderr


def _is_lens_applied(cli_stdout):
    """Detect whether lens correction actually ran (vs no-op)."""
    # applyLensCorrection returns true and prints "Correction complete." when
    # it ran; the CLI prints "-> Skipped" when it returned false.
    if "Correction complete." in cli_stdout:
        return True
    if "Skipping correction" in cli_stdout or "-> Skipped" in cli_stdout:
        return False
    # Ambiguous: assume not applied.
    return False


def _border_pixel_mean(pixels, W, H):
    """Mean (0..1) of the BORDER_PIXELS rim, all channels."""
    s = 0
    n = 0
    for r in range(H):
        for c in range(W):
            on_border = (r < BORDER_PIXELS or r >= H - BORDER_PIXELS or
                         c < BORDER_PIXELS or c >= W - BORDER_PIXELS)
            if on_border:
                for ch in range(3):
                    s += pixel_at(pixels, W, r, c, ch)
                    n += 1
    return (s / n) / 65535.0 if n else 0.0


def _border_fringe_metric(pixels, W, H):
    """Max per-pixel channel spread at the border (0..1).

    For each border pixel, spread = (max(R,G,B) - min(R,G,B)) / max(R,G,B, 1).
    A masked (uniformly-zeroed) pixel has spread 0; a fringed pixel has a
    large spread. We report the 95th-percentile spread to ignore outliers.
    """
    spreads = []
    for r in range(H):
        for c in range(W):
            on_border = (r < BORDER_PIXELS or r >= H - BORDER_PIXELS or
                         c < BORDER_PIXELS or c >= W - BORDER_PIXELS)
            if not on_border:
                continue
            vals = [pixel_at(pixels, W, r, c, ch) / 65535.0 for ch in range(3)]
            mx = max(vals)
            if mx < 1e-3:
                spreads.append(0.0)  # uniformly near-zero (masked or dark) — no fringe
            else:
                spreads.append((mx - min(vals)) / mx)
    if not spreads:
        return 0.0
    spreads.sort()
    return spreads[int(0.95 * (len(spreads) - 1))]


def run_smoke_test():
    print()
    print("=" * 64)
    print("SECONDARY: Sample.NEF lens-correction smoke (edge robustness)")
    print("=" * 64)
    if not os.path.isfile(CLI):
        print(f"  ERROR: {CLI} not found — build first.")
        return False
    if not os.path.isfile(RAW):
        print(f"  ERROR: {RAW} not found.")
        return False

    on_path = "/tmp/cpp_lens_on.tif"
    off_path = "/tmp/cpp_lens_off.tif"

    print("  Running CLI (lens ON)...")
    ok_on, out_on = _run_cli(on_path, lens_on=True)
    if not ok_on:
        print("  FAIL: CLI (lens ON) exited non-zero.")
        return False

    applied = _is_lens_applied(out_on)
    if not applied:
        print("  Lens correction was a NO-OP (Lensfun DB lacks the camera/lens")
        print("  or no distortion/TCA data). This is acceptable — the PRIMARY")
        print("  synthetic test is the authoritative validation.")
        print("\nSECONDARY: PASS (clean no-op; rely on PRIMARY)")
        return True

    print("  Lens correction APPLIED. Running CLI (lens OFF) for comparison...")
    ok_off, _ = _run_cli(off_path, lens_on=False)
    if not ok_off:
        print("  FAIL: CLI (lens OFF) exited non-zero.")
        return False

    w_on, h_on, px_on = read_tiff_pixels(on_path)
    w_off, h_off, px_off = read_tiff_pixels(off_path)
    if (w_on, h_on) != (w_off, h_off):
        print(f"  FAIL: output size mismatch ON {w_on}x{h_on} vs OFF {w_off}x{h_off}.")
        return False
    W, H = w_on, h_on
    print(f"  Output: {W} x {H}")

    # 1. No black borders introduced by the OOB mask. The lens-ON border mean
    #    must not collapse relative to lens-OFF (a mass-zeroing mask failure
    #    would crater the border mean toward 0).
    border_on = _border_pixel_mean(px_on, W, H)
    border_off = _border_pixel_mean(px_off, W, H)
    ratio = border_on / border_off if border_off > 1e-6 else 1.0
    print(f"  Border mean: ON={border_on:.4f}  OFF={border_off:.4f}  ratio={ratio:.3f}")
    no_black_border = border_on >= BLACK_BORDER_MAX_MEAN or ratio > 0.5

    # 2. No per-channel fringing at the border. The lens-ON 95th-percentile
    #    channel spread at the border must stay bounded (a per-channel
    #    constant-0 fringing failure would spike it).
    fringe_on = _border_fringe_metric(px_on, W, H)
    fringe_off = _border_fringe_metric(px_off, W, H)
    print(f"  Border 95th-pct channel spread: ON={fringe_on:.4f}  OFF={fringe_off:.4f}")
    no_fringe = fringe_on <= max(FRINGE_MAX_CHANNEL_SPREAD, fringe_off * 1.25 + 0.05)

    print(f"  no-black-border: {'PASS' if no_black_border else 'FAIL'}")
    print(f"  no-fringing:     {'PASS' if no_fringe else 'FAIL'}")

    ok = no_black_border and no_fringe
    print(f"\nSECONDARY: {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================
#           LEGACY: before/after TIFF comparison
# ============================================================

def run_legacy_compare(off_path, on_path):
    import math
    print("=== Legacy Lens Correction Comparison ===\n")
    w1, h1, px_off = read_tiff_pixels(off_path)
    w2, h2, px_on = read_tiff_pixels(on_path)
    assert w1 == w2 and h1 == h2, f"Size mismatch: {w1}x{h1} vs {w2}x{h2}"
    W, H = w1, h1

    total = W * H
    diff_sq_sum = 0.0
    for i in range(len(px_off)):
        d = abs(px_off[i] - px_on[i])
        diff_sq_sum += d * d
    mse = diff_sq_sum / len(px_off)
    psnr = 10.0 * math.log10((65535.0 ** 2) / mse) if mse > 0 else float("inf")
    print(f"  Image: {W} x {H}, PSNR: {psnr:.2f} dB")
    ok = 15.0 < psnr < 50.0
    print(f"  Legacy: {'PASS' if ok else 'FAIL'} (15 < PSNR < 50)")
    return ok


# ============================================================
def main():
    if len(sys.argv) >= 3:
        ok = run_legacy_compare(sys.argv[1], sys.argv[2])
        sys.exit(0 if ok else 1)

    ok_primary = run_synthetic_test()
    ok_secondary = run_smoke_test() if ok_primary else False

    print()
    print("=" * 64)
    print(f"Summary:  PRIMARY (synthetic) = {'PASS' if ok_primary else 'FAIL'}    "
          f"SECONDARY (smoke) = {'PASS' if ok_secondary else 'FAIL'}")
    print("=" * 64)
    # PRIMARY is authoritative; SECONDARY may be a clean no-op pass.
    sys.exit(0 if ok_primary else 1)


if __name__ == "__main__":
    main()
