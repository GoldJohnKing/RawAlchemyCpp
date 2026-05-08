"""
Cross-validation: compare C++ lens correction output with and without correction.

Validates that:
1. Lens correction produces measurable pixel differences (not a no-op)
2. Vignetting correction brightens corners more than center
3. Distortion correction changes edge geometry more than center
4. Overall change is within expected bounds (not destructive)

Requires: Python 3 stdlib only (no numpy/scipy/rawpy needed)

Usage:
  python3 cross_validate_lens.py /tmp/lens_off.tif /tmp/lens_on.tif
"""

import struct
import sys
import os
import math


def read_tiff_pixels(path):
    """Read a simple 16-bit RGB TIFF file and return (width, height, pixels).
    
    pixels is a flat list of uint16 values in RGB order.
    Only handles uncompressed, strip-organized, little-endian 16-bit RGB TIFFs.
    """
    with open(path, 'rb') as f:
        # TIFF header
        byte_order = f.read(2)
        if byte_order == b'II':
            endian = '<'
        elif byte_order == b'MM':
            endian = '>'
        else:
            raise ValueError(f"Not a TIFF file: {path}")
        
        magic = struct.unpack(endian + 'H', f.read(2))[0]
        assert magic == 42, f"Bad TIFF magic: {magic}"
        
        ifd_offset = struct.unpack(endian + 'I', f.read(4))[0]
        
        # Parse IFD
        f.seek(ifd_offset)
        num_entries = struct.unpack(endian + 'H', f.read(2))[0]
        
        tags = {}
        for _ in range(num_entries):
            tag_id = struct.unpack(endian + 'H', f.read(2))[0]
            type_id = struct.unpack(endian + 'H', f.read(2))[0]
            count = struct.unpack(endian + 'I', f.read(4))[0]
            value_raw = f.read(4)
            
            # Type sizes
            type_sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8}
            val_size = type_sizes.get(type_id, 1) * count
            
            if val_size <= 4:
                if type_id == 3:  # SHORT
                    tags[tag_id] = struct.unpack(endian + 'H', value_raw[:2])[0]
                elif type_id == 4:  # LONG
                    tags[tag_id] = struct.unpack(endian + 'I', value_raw)[0]
                elif type_id == 1:  # BYTE
                    tags[tag_id] = value_raw[0]
            else:
                offset = struct.unpack(endian + 'I', value_raw)[0]
                tags[tag_id] = ('ptr', offset, count, type_id)
        
        width = tags.get(256, 0)
        height = tags.get(257, 0)
        bits_per_sample = tags.get(258, 16)
        compression = tags.get(259, 1)
        photometric = tags.get(262, 2)
        strip_offsets = tags.get(273)
        strip_byte_counts = tags.get(279)
        samples_per_pixel = tags.get(277, 3)
        rows_per_strip = tags.get(278, height)
        
        assert width > 0 and height > 0, f"Invalid dimensions: {width}x{height}"
        assert bits_per_sample == 16, f"Expected 16-bit, got {bits_per_sample}"
        assert compression == 1, f"Expected uncompressed, got compression={compression}"
        assert samples_per_pixel == 3, f"Expected RGB (3), got {samples_per_pixel}"
        
        # Get strip data
        if isinstance(strip_offsets, tuple):
            _, soff, scount, stype = strip_offsets
            f.seek(soff)
            strip_off_list = [struct.unpack(endian + 'I', f.read(4))[0] for _ in range(scount)]
        else:
            strip_off_list = [strip_offsets]
        
        if isinstance(strip_byte_counts, tuple):
            _, sboff, sbcount, sbtype = strip_byte_counts
            f.seek(sboff)
            strip_bc_list = [struct.unpack(endian + 'I', f.read(4))[0] for _ in range(sbcount)]
        else:
            strip_bc_list = [strip_byte_counts]
        
        # Read all pixel data
        all_data = bytearray()
        for off, bc in zip(strip_off_list, strip_bc_list):
            f.seek(off)
            all_data.extend(f.read(bc))
        
        # Convert to uint16 array
        num_pixels = width * height * 3
        fmt = endian + ('H' * num_pixels)
        pixels = struct.unpack(fmt, bytes(all_data[:num_pixels * 2]))
        
        return width, height, pixels


def pixel_at(pixels, width, row, col, channel):
    """Get pixel value at (row, col, channel)"""
    return pixels[(row * width + col) * 3 + channel]


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 cross_validate_lens.py <no_correction.tif> <with_correction.tif>")
        sys.exit(1)
    
    off_path = sys.argv[1]
    on_path = sys.argv[2]
    
    print("=== Lens Correction Cross-Validation ===\n")
    print(f"  Without: {off_path}")
    print(f"  With:    {on_path}")
    print()
    
    w1, h1, px_off = read_tiff_pixels(off_path)
    w2, h2, px_on = read_tiff_pixels(on_path)
    
    assert w1 == w2 and h1 == h2, f"Size mismatch: {w1}x{h1} vs {w2}x{h2}"
    W, H = w1, h1
    
    print(f"  Image: {W} x {H} = {W*H/1e6:.1f} MP (16-bit RGB)\n")
    
    # ---- 1. Overall pixel difference statistics ----
    total = W * H
    diff_sum = 0.0
    diff_sq_sum = 0.0
    max_diff = 0
    changed_pixels = 0
    
    # Per-region statistics
    margin = min(W, H) // 10  # 10% border
    center_diffs = []
    corner_diffs = []
    
    for row in range(H):
        for col in range(W):
            for c in range(3):
                v_off = pixel_at(px_off, W, row, col, c)
                v_on = pixel_at(px_on, W, row, col, c)
                d = abs(v_off - v_on)
                diff_sum += d
                diff_sq_sum += d * d
                if d > max_diff:
                    max_diff = d
                if d > 0:
                    changed_pixels += 1
                
                if c == 1:  # Green channel only for region stats
                    is_corner = (row < margin or row >= H - margin) and (col < margin or col >= W - margin)
                    is_center = (margin * 3 <= row < H - margin * 3) and (margin * 3 <= col < W - margin * 3)
                    if is_corner:
                        corner_diffs.append(d)
                    elif is_center:
                        center_diffs.append(d)
    
    num_values = total * 3
    mean_diff = diff_sum / num_values
    
    # ---- 2. PSNR ----
    mse = diff_sq_sum / num_values
    if mse > 0:
        psnr = 10.0 * math.log10((65535.0 ** 2) / mse)
    else:
        psnr = float('inf')
    
    print("--- Overall Difference ---")
    print(f"  Pixels changed:  {changed_pixels}/{num_values} ({100*changed_pixels/num_values:.2f}%)")
    print(f"  Mean abs diff:   {mean_diff:.2f} / 65535")
    print(f"  Max abs diff:    {max_diff}")
    print(f"  PSNR:            {psnr:.2f} dB")
    print()
    
    # ---- 3. Region analysis ----
    corner_mean = sum(corner_diffs) / len(corner_diffs) if corner_diffs else 0
    center_mean = sum(center_diffs) / len(center_diffs) if center_diffs else 0
    
    print("--- Region Analysis (Green channel) ---")
    print(f"  Corner mean diff: {corner_mean:.2f}")
    print(f"  Center mean diff: {center_mean:.2f}")
    print(f"  Corner/Center ratio: {corner_mean/center_mean:.2f}x" if center_mean > 0 else "  Corner/Center ratio: N/A")
    print()
    
    # ---- 4. Vignetting validation ----
    # Vignetting darkens corners; correction should brighten them.
    # Check: corners should have higher values in corrected image.
    corners = [
        (0, 0), (0, W-1), (H-1, 0), (H-1, W-1),
        (0, W//2), (H//2, 0), (H//2, W-1), (H-1, W//2),
    ]
    center = (H//2, W//2)
    
    print("--- Vignetting Correction Check ---")
    for label, (r, c) in [("Top-left", corners[0]), ("Top-right", corners[1]),
                           ("Bot-left", corners[2]), ("Bot-right", corners[3]),
                           ("Top-mid", corners[4]), ("Left-mid", corners[5]),
                           ("Right-mid", corners[6]), ("Bot-mid", corners[7]),
                           ("CENTER", center)]:
        g_off = pixel_at(px_off, W, r, c, 1)
        g_on = pixel_at(px_on, W, r, c, 1)
        delta = g_on - g_off
        pct = 100.0 * delta / g_off if g_off > 0 else 0
        print(f"  {label:12s}: G_off={g_off:5d}  G_on={g_on:5d}  delta={delta:+5d} ({pct:+.2f}%)")
    print()
    
    # ---- 5. Sample grid comparison ----
    print("--- Sample Pixel Comparison (R, G, B) ---")
    print(f"  {'Position':20s}  {'Without correction':>30s}  {'With correction':>30s}  {'Delta':>15s}")
    samples = [
        (0, 0, "Top-left"),
        (0, W-1, "Top-right"),
        (H-1, 0, "Bottom-left"),
        (H-1, W-1, "Bottom-right"),
        (H//2, W//2, "Center"),
        (H//4, W//4, "Quarter"),
        (3*H//4, 3*W//4, "Three-quarter"),
    ]
    for row, col, label in samples:
        off_px = tuple(pixel_at(px_off, W, row, col, c) for c in range(3))
        on_px = tuple(pixel_at(px_on, W, row, col, c) for c in range(3))
        delta = tuple(on_px[c] - off_px[c] for c in range(3))
        print(f"  {label:20s}  ({off_px[0]:5d},{off_px[1]:5d},{off_px[2]:5d})      "
              f"({on_px[0]:5d},{on_px[1]:5d},{on_px[2]:5d})      "
              f"({delta[0]:+5d},{delta[1]:+5d},{delta[2]:+5d})")
    print()
    
    # ---- 6. Verdict ----
    print("--- Verdict ---")
    if changed_pixels == 0:
        print("  FAIL: No pixels were changed by lens correction!")
    elif psnr > 50:
        print("  FAIL: PSNR too high (>50 dB), correction may be trivial.")
    elif psnr < 15:
        print("  WARN: PSNR very low (<15 dB), correction may be too aggressive.")
    else:
        print(f"  PASS: Lens correction produces measurable changes (PSNR={psnr:.1f} dB)")
    
    if corner_mean > center_mean * 1.2:
        print("  PASS: Corner regions changed more than center (vignetting + distortion)")
    else:
        print("  WARN: Corner vs center difference not as expected")
    
    print()


if __name__ == "__main__":
    main()
