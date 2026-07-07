#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate a synthetic 64x64 RGGB Bayer CFA fixture for the NN dispatch
integration test (Test/cpp/test_nn_dispatch.cpp, sub-test 3).

Writes Test/data/bayer_test_cfa.bin as little-endian float32 (64*64 = 4096
values, 16384 bytes). The pattern is a smooth diagonal gradient in [0, 1) —
enough per-pixel variation to exercise the normalize + tile paths without
modelling real sensor noise, and small enough to commit (16 KB).

The fixture deliberately stays below 1.0 to avoid clipped highlights, which
keeps x-veon's bounded residual activation inside its trained range (see the
highlight-recon deviation note in demosaic_nn_xveon.cpp).

Run from anywhere (resolves its own output path relative to this file):

    python3 Test/data/generate_fixture.py
"""
from pathlib import Path
import struct
import sys

W, H = 64, 64


def main() -> int:
    values = bytearray()
    for y in range(H):
        for x in range(W):
            # Diagonal gradient in [0, 1): per-pixel variation, no clipping.
            v = (x + y) / (W + H - 2)
            values += struct.pack("<f", v)

    out = Path(__file__).resolve().parent / "bayer_test_cfa.bin"
    out.write_bytes(values)
    print(f"wrote {out} ({len(values)} bytes, {W}x{H} float32)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
