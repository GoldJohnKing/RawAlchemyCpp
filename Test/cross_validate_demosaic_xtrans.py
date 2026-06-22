#!/usr/bin/env python3
"""
Cross-validate our 3-pass Markesteijn implementation against darktable's reference.

Prerequisites:
  - darktable CLI installed (darktable-cli)
  - raw_alchemy_cli built with BUILD_CLI=ON
  - Sample.RAF (Fujifilm X-Trans) in Test/
  - pip install numpy pillow scikit-image

Usage:
  python3 cross_validate_demosaic_xtrans.py [--sample Sample.RAF] [--cli path/to/cli]

Note: This script was written during the demosaic port but could not be runtime-validated
because darktable-cli is not installed. Run after installing darktable and building CLI.
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

TEST_DIR = Path(__file__).parent
DEFAULT_SAMPLE = TEST_DIR / "Sample.RAF"
DEFAULT_CLI = TEST_DIR.parent / "build-windows-dll" / "bin" / "raw_alchemy_cli.exe"


def run_darktable_reference(sample: Path, output_tiff: Path) -> None:
    cmd = [
        "darktable-cli",
        str(sample),
        str(output_tiff),
        "--core",
        "--conf", "plugins/darkroom/demosaic/method_xtrans=markesteijn3",
    ]
    print(f"Running darktable: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_raw_alchemy(sample: Path, output_tiff: Path, cli_path: Path) -> None:
    cmd = [
        str(cli_path),
        "--input", str(sample),
        "--output", str(output_tiff),
        "--demosaic", "markesteijn",
        "--format", "tiff16",
    ]
    print(f"Running raw_alchemy: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-validate Markesteijn vs darktable")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    args = parser.parse_args()

    if not args.sample.exists():
        print(f"Error: sample RAF file not found: {args.sample}", file=sys.stderr)
        print("Place a Fujifilm .RAF file in Test/ to enable X-Trans validation.", file=sys.stderr)
        return 1
    if not args.cli.exists():
        print(f"Error: raw_alchemy_cli not found: {args.cli}", file=sys.stderr)
        return 1

    dt_output = TEST_DIR / "dt_xtrans_reference.tiff"
    ra_output = TEST_DIR / "ra_markesteijn.tiff"

    try:
        run_darktable_reference(args.sample, dt_output)
        run_raw_alchemy(args.sample, ra_output, args.cli)

        dt_img = np.array(Image.open(dt_output), dtype=np.float64) / 65535.0
        ra_img = np.array(Image.open(ra_output), dtype=np.float64) / 65535.0

        if dt_img.shape != ra_img.shape:
            print(f"Error: shape mismatch", file=sys.stderr)
            return 1

        psnr = compute_psnr(dt_img, ra_img)
        print(f"PSNR: {psnr:.2f} dB")

        if ssim is not None:
            s = ssim(dt_img, ra_img, channel_axis=2, data_range=1.0)
            print(f"SSIM: {s:.4f}")

        if psnr >= 40.0:
            print("PASS")
            return 0
        else:
            print("FAIL: PSNR < 40 dB")
            diff = np.abs(dt_img - ra_img)
            diff_scaled = (diff * 10 * 65535).clip(0, 65535).astype(np.uint16)
            Image.fromarray(diff_scaled).save(TEST_DIR / "diff_heatmap_xtrans.tiff")
            return 1
    finally:
        for f in [dt_output, ra_output]:
            if f.exists():
                f.unlink()


if __name__ == "__main__":
    sys.exit(main())
