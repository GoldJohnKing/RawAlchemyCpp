#!/usr/bin/env python3
"""
Benchmark our demosaic implementations against LibRaw's built-in path.

Acceptance (6 MP RAW, single-threaded, OMP_NUM_THREADS=1):
  RCD:                < 1.5 seconds
  Markesteijn 3-pass: < 5.0 seconds

Usage:
  OMP_NUM_THREADS=1 python3 benchmark_demosaic.py [--cli path/to/cli]

Note: Requires raw_alchemy_cli built with BUILD_CLI=ON.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

TEST_DIR = Path(__file__).parent
DEFAULT_CLI = TEST_DIR.parent / "build-windows-dll" / "bin" / "raw_alchemy_cli.exe"


def benchmark(cli_path: Path, sample: Path, demosaic: str, runs: int = 3) -> float:
    """Return median wall-clock time in seconds."""
    times = []
    env = dict(os.environ, OMP_NUM_THREADS="1")
    for _ in range(runs):
        out = Path("/tmp/bench_out.tiff")
        start = time.perf_counter()
        subprocess.run(
            [str(cli_path), "--input", str(sample), "--output", str(out),
             "--demosaic", demosaic, "--format", "tiff16"],
            check=True, env=env, capture_output=True,
        )
        times.append(time.perf_counter() - start)
        out.unlink(missing_ok=True)
    times.sort()
    return times[len(times) // 2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark demosaic performance")
    parser.add_argument("--bayer-sample", type=Path, default=TEST_DIR / "Sample.NEF")
    parser.add_argument("--xtrans-sample", type=Path, default=TEST_DIR / "Sample.RAF")
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    args = parser.parse_args()

    if not args.cli.exists():
        print(f"Error: raw_alchemy_cli not found: {args.cli}", file=sys.stderr)
        return 1

    print(f"{'Algorithm':<25} {'Time (s)':>10} {'Limit (s)':>10} {'Status':>8}")
    print("-" * 55)

    all_pass = True

    if args.bayer_sample.exists():
        t = benchmark(args.cli, args.bayer_sample, "rcd")
        ok = t < 1.5
        all_pass &= ok
        print(f"{'RCD (Bayer)':<25} {t:>10.2f} {'1.50':>10} {'PASS' if ok else 'FAIL':>8}")
    else:
        print(f"{'RCD (Bayer)':<25} {'skipped (no sample)':>29}")

    if args.xtrans_sample.exists():
        t = benchmark(args.cli, args.xtrans_sample, "markesteijn")
        ok = t < 5.0
        all_pass &= ok
        print(f"{'Markesteijn 3-pass':<25} {t:>10.2f} {'5.00':>10} {'PASS' if ok else 'FAIL':>8}")
    else:
        print(f"{'Markesteijn 3-pass':<25} {'skipped (no RAF)':>29}")

    if args.bayer_sample.exists():
        t = benchmark(args.cli, args.bayer_sample, "libraw")
        print(f"{'LibRaw fallback':<25} {t:>10.2f} {'n/a':>10} {'baseline':>8}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
