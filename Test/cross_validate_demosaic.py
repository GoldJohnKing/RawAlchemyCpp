#!/usr/bin/env python3
"""
Cross-validate our RCD implementation against darktable's reference output.

Prerequisites:
  - darktable CLI installed (darktable-cli)
  - raw_alchemy_cli built with BUILD_CLI=ON
  - Sample.NEF in Test/
  - pip install numpy pillow scikit-image

Workflow:
  1. Run darktable on Sample.NEF with demosaic=rcd, all other iops disabled
  2. Run raw_alchemy_cli on Sample.NEF with --demosaic rcd
  3. Compute PSNR + SSIM between the two outputs
  4. Acceptance: PSNR >= 40 dB

Usage:
  python3 cross_validate_demosaic.py [--sample Sample.NEF] [--cli path/to/cli]

Note: This script was written during the demosaic port but could not be runtime-validated
because darktable-cli is not installed in the development environment. Run it after
installing darktable and building raw_alchemy_cli with BUILD_CLI=ON.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None
    print("Warning: scikit-image not installed, SSIM will be skipped", file=sys.stderr)

TEST_DIR = Path(__file__).parent
DEFAULT_SAMPLE = TEST_DIR / "Sample.NEF"
DEFAULT_CLI = TEST_DIR.parent / "build-windows-dll" / "bin" / "raw_alchemy_cli.exe"


def run_darktable_reference(sample: Path, output_tiff: Path) -> None:
    """Run darktable CLI with RCD demosaic only, no other processing."""
    xmp = sample.with_suffix(".xmp")
    xmp.write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        '  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '    <rdf:Description rdf:about=""\n'
        '      xmlns:darktable="http://darktable.sf.net/">\n'
        '      <darktable:masks_history><rdf:Seq/></darktable:masks_history>\n'
        '      <darktable:history>\n'
        '        <rdf:Seq>\n'
        '          <rdf:li darktable:operation="demosaic"\n'
        '                  darktable:enabled="1"\n'
        '                  darktable:params="0000000005000000"/>\n'
        '        </rdf:Seq>\n'
        '      </darktable:history>\n'
        '    </rdf:Description>\n'
        '  </rdf:RDF>\n'
        '</x:xmpmeta>\n'
    )
    cmd = [
        "darktable-cli",
        str(sample),
        str(xmp),
        str(output_tiff),
        "--width", "0",
        "--height", "0",
        "--core",
        "--conf", "plugins/darkroom/demosaic/method_bayer=rcd",
    ]
    print(f"Running darktable: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_raw_alchemy(sample: Path, output_tiff: Path, cli_path: Path) -> None:
    """Run our raw_alchemy_cli with RCD demosaic."""
    cmd = [
        str(cli_path),
        "--input", str(sample),
        "--output", str(output_tiff),
        "--demosaic", "rcd",
        "--format", "tiff16",
    ]
    print(f"Running raw_alchemy: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute PSNR between two float64 [0,1] RGB images."""
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-validate RCD vs darktable")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    args = parser.parse_args()

    if not args.sample.exists():
        print(f"Error: sample file not found: {args.sample}", file=sys.stderr)
        return 1
    if not args.cli.exists():
        print(f"Error: raw_alchemy_cli not found: {args.cli}", file=sys.stderr)
        print("Build with BUILD_CLI=ON to enable the CLI.", file=sys.stderr)
        return 1

    dt_output = TEST_DIR / "dt_reference.tiff"
    ra_output = TEST_DIR / "ra_rcd.tiff"

    try:
        run_darktable_reference(args.sample, dt_output)
        run_raw_alchemy(args.sample, ra_output, args.cli)

        dt_img = np.array(Image.open(dt_output), dtype=np.float64) / 65535.0
        ra_img = np.array(Image.open(ra_output), dtype=np.float64) / 65535.0

        if dt_img.shape != ra_img.shape:
            print(
                f"Error: shape mismatch dt={dt_img.shape} vs ra={ra_img.shape}",
                file=sys.stderr,
            )
            return 1

        psnr = compute_psnr(dt_img, ra_img)
        print(f"PSNR: {psnr:.2f} dB")

        if ssim is not None:
            s = ssim(dt_img, ra_img, channel_axis=2, data_range=1.0)
            print(f"SSIM: {s:.4f}")

        if psnr >= 40.0:
            print("PASS: PSNR >= 40 dB (visually indistinguishable)")
            return 0
        else:
            print("FAIL: PSNR < 40 dB — investigate differences")
            diff = np.abs(dt_img - ra_img)
            diff_scaled = (diff * 10 * 65535).clip(0, 65535).astype(np.uint16)
            Image.fromarray(diff_scaled).save(TEST_DIR / "diff_heatmap.tiff")
            print(f"Diff heatmap saved to {TEST_DIR / 'diff_heatmap.tiff'}")
            return 1
    finally:
        for f in [dt_output, ra_output]:
            if f.exists():
                f.unlink()


if __name__ == "__main__":
    sys.exit(main())
