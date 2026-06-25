#!/usr/bin/env python3
"""
Cross-validate our X-Trans wavelet denoise against darktable's rawdenoise iop.

Prerequisites:
  - darktable CLI installed (darktable-cli)
  - raw_alchemy_cli built with BUILD_CLI=ON
  - Sample.RAF (Fujifilm X-Trans) in Test/
  - pip install numpy pillow scikit-image   (exifread optional, for ISO traceability)

Usage:
  python3 cross_validate_denoise_xtrans.py [--sample Sample.RAF] [--cli path/to/cli]

Domain note (why we expect close agreement): both pipelines denoise
pre-white-balance in the same domain. darktable's rawdenoise iop runs in its
raw pipeline pre-WB; ours runs in LibRaw's pre_scalecolors_cb hook pre-WB.
Residual differences come from (a) our single-threaded sensel extraction vs
darktable's chunked one, and (b) LibRaw vs darktable black-level/highlight
handling. Track the PSNR number; if it regresses or shows systematic
under/over-denoising, tune computeXtransDenoiseThreshold (denoise_xtrans.cpp).

NOTE: This script could not be runtime-validated in this environment
(darktable-cli + Sample.RAF absent). It mirrors cross_validate_demosaic_xtrans.py
structurally (self-contained PSNR/SSIM/diff helpers — the shared
cross_validate.py predates those helpers and exports none of them). Run after
installing darktable, building the CLI, and supplying Sample.RAF.

darktable rawdenoise enablement: rawdenoise is not in darktable's default
pipeline, so --conf alone cannot switch it on. We enable it via a generated
XMP sidecar passed to darktable-cli --apply. The rawdenoise params blob is a
darktable-version-specific binary struct, so the sidecar carries empty params
and relies on the module's ISO-adaptive defaults. If your darktable build
ignores an empty-params history entry, generate the reference sidecar from
darktable-gui (load Sample.RAF, enable rawdenoise, export the .xmp) and pass
it via --apply instead of regenerating here.
"""
import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None

TEST_DIR = Path(__file__).parent
DEFAULT_SAMPLE = TEST_DIR / "Sample.RAF"
DEFAULT_CLI = TEST_DIR.parent / "build-windows-dll" / "bin" / "raw_alchemy_cli.exe"

# Denoise is inherently lossy and the two implementations differ in chunking
# and black-level handling, so the floor is looser than the demosaic cross-val's
# 40 dB. 30 dB is the realistic "close agreement" bar for a pre-WB denoise.
PSNR_FLOOR_DB = 30.0


def read_exif_iso(path):
    """Read ISOSpeedRatings from a RAF; return None if exifread missing or tag absent."""
    try:
        import exifread
    except ImportError:
        return None
    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=False)
    for key in ("EXIF ISOSpeedRatings", "EXIF PhotographicSensitivity"):
        if key in tags:
            try:
                return int(str(tags[key]))
            except ValueError:
                return None
    return None


def iso_to_rawdenoise_threshold(iso):
    """Mirror rawalchemy's computeXtransDenoiseThreshold ISO curve (denoise_xtrans.cpp).

    Returned value documents what OUR pipeline applies internally; darktable's
    rawdenoise uses its own (empty-params) defaults, so this is not injected
    into the darktable sidecar — it is printed for traceability only.
    """
    if not iso or iso <= 0:
        return 0.02  # EXIF ISO unavailable; pick a mid-curve default for reporting
    if iso <= 100:
        return 0.0
    if iso <= 400:
        return 0.01
    log_low = 8.644    # log2(400)
    log_high = 13.644  # log2(12800)
    t = (math.log2(iso) - log_low) / (log_high - log_low)
    t = max(0.0, min(1.0, t))
    if t >= 1.0:
        return 0.05
    return 0.01 + t * 0.04


def write_rawdenoise_sidecar(out_path, threshold, iso):
    """Write an XMP sidecar enabling darktable's rawdenoise iop.

    Enabling a non-default iop via darktable-cli requires a history entry in an
    XMP sidecar (--conf only tunes always-on modules). We add rawdenoise with
    empty params so the module initializes its own defaults. See module docstring
    for the GUI fallback if darktable rejects the empty-params entry.
    """
    iso_label = iso if iso else "unknown"
    label = "iso{}_thr{:.4f}".format(iso_label, threshold)
    xmp = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="rawalchemy-crossval">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description\n'
        '    xmlns:darktable="http://darktable.org/xmp/1.0/"\n'
        '    darktable:version="3.0"\n'
        '    darktable:iop_order_version="2"\n'
        '    darktable:history_end="1"\n'
        '    darktable:label="{}">\n'
        '   <darktable:history rdf:parseType="Resource">\n'
        '    <rdf:Seq>\n'
        '     <rdf:li\n'
        '      darktable:num="0"\n'
        '      darktable:operation="rawdenoise"\n'
        '      darktable:module_name="rawdenoise"\n'
        '      darktable:enabled="1"\n'
        '      darktable:modversion="2"\n'
        '      darktable:params=""\n'
        '      darktable:multi_name="{}"\n'
        '      darktable:multi_priority="0"\n'
        '      darktable:iop_order="20.000000"/>\n'
        '    </rdf:Seq>\n'
        '   </darktable:history>\n'
        '   <darktable:masks rdf:parseType="Resource"/>\n'
        '  </rdf:Description>\n'
        ' </rdf:RDF>\n'
        '</x:xmpmeta>\n'
    ).format(label, label)
    out_path.write_text(xmp, encoding="utf-8")
    return out_path


def run_darktable_reference(sample, output_tiff, sidecar):
    # --apply enables rawdenoise via the sidecar; --conf selects Markesteijn-3
    # to match our --demosaic markesteijn path, isolating the denoise delta.
    cmd = [
        "darktable-cli",
        str(sample),
        str(output_tiff),
        "--apply", str(sidecar),
        "--core",
        "--conf", "plugins/darkroom/demosaic/method_xtrans=markesteijn3",
    ]
    print("Running darktable: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_raw_alchemy(sample, output_tiff, cli_path, demosaic):
    # X-Trans denoise is automatic (pre_scalecolors_cb hook); no explicit flag.
    # --demosaic markesteijn mirrors darktable's ref; switch to "auto" to
    # exercise the production default at the cost of a confounding variable.
    cmd = [
        str(cli_path),
        "--input", str(sample),
        "--output", str(output_tiff),
        "--demosaic", demosaic,
        "--format", "tiff16",
    ]
    print("Running raw_alchemy: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def compute_psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def main():
    parser = argparse.ArgumentParser(description="Cross-validate X-Trans denoise vs darktable")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE,
                        help="Path to Fujifilm .RAF sample")
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI,
                        help="Path to raw_alchemy_cli executable")
    parser.add_argument("--demosaic", default="markesteijn",
                        choices=["markesteijn", "auto", "rcd", "libraw"],
                        help="Our demosaic algo (markesteijn matches darktable's ref)")
    args = parser.parse_args()

    if not args.sample.exists():
        print("[skip] {} not found; supply a Fujifilm .RAF to run this check".format(args.sample))
        return 0
    if not args.cli.exists():
        print("Error: raw_alchemy_cli not found: {}".format(args.cli), file=sys.stderr)
        print("Build with BUILD_CLI=ON (see build-windows-dll).", file=sys.stderr)
        return 1

    iso = read_exif_iso(args.sample)
    threshold = iso_to_rawdenoise_threshold(iso)
    print("EXIF ISO: {}  -> our internal threshold: {:.4f}".format(
        iso if iso else "unknown", threshold))

    dt_output = TEST_DIR / "dt_xtrans_denoise_reference.tiff"
    ra_output = TEST_DIR / "ra_xtrans_denoise.tiff"
    sidecar = TEST_DIR / "dt_rawdenoise_sidecar.xmp"
    diff_heatmap = TEST_DIR / "diff_heatmap_xtrans_denoise.tiff"

    try:
        write_rawdenoise_sidecar(sidecar, threshold, iso)
        run_darktable_reference(args.sample, dt_output, sidecar)
        run_raw_alchemy(args.sample, ra_output, args.cli, args.demosaic)

        dt_img = np.array(Image.open(dt_output), dtype=np.float64) / 65535.0
        ra_img = np.array(Image.open(ra_output), dtype=np.float64) / 65535.0

        if dt_img.shape != ra_img.shape:
            print("Error: shape mismatch {} vs {}".format(dt_img.shape, ra_img.shape),
                  file=sys.stderr)
            return 1

        psnr = compute_psnr(dt_img, ra_img)

        # Diff heatmap (10x amplified, 16-bit) — PIL/numpy only, no skimage dep.
        diff = np.abs(dt_img - ra_img)
        diff_scaled = (diff * 10 * 65535).clip(0, 65535).astype(np.uint16)
        Image.fromarray(diff_scaled).save(diff_heatmap)

        line = "PSNR: {:.2f} dB".format(psnr)
        if ssim is not None:
            s = ssim(dt_img, ra_img, channel_axis=2, data_range=1.0)
            line += "   SSIM: {:.4f}".format(s)
        else:
            line += "   SSIM: (scikit-image not installed; skipped)"
        line += "   diff: {}".format(diff_heatmap)
        print(line)

        if psnr >= PSNR_FLOOR_DB:
            print("PASS")
            return 0
        print("FAIL: PSNR < {} dB".format(PSNR_FLOOR_DB))
        return 1
    finally:
        # Keep only the diff heatmap (gitignored via *.tiff); regenerate
        # intermediates + sidecar on each run.
        for f in [dt_output, ra_output, sidecar]:
            if f.exists():
                f.unlink()


if __name__ == "__main__":
    sys.exit(main())
