"""
Cross-validate Step 2 (Log Signal Preparation) against Python reference.
Decodes the same NEF with rawpy, applies the same gamut+log transform,
and compares pixel-level statistics with the C++ output.
"""
import rawpy
import numpy as np
import colour
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Raw-Alchemy', 'src'))
from raw_alchemy.config import LOG_TO_WORKING_SPACE, LOG_ENCODING_MAP
from raw_alchemy import utils

def main():
    raw_path = "/mnt/d/GitRepos/Raw_Log_Lut/Test/DSC_5065.NEF"
    log_space = "F-Log2"
    exposure = 5.0  # same as C++ test

    print(f"=== Python Reference: {log_space} with exposure +{exposure} ===\n")

    with rawpy.imread(raw_path) as raw:
        # Step 1: Decode to ProPhoto Linear
        prophoto_linear = raw.postprocess(
            gamma=(1, 1), no_auto_bright=True, use_camera_wb=True,
            output_bps=16, output_color=rawpy.ColorSpace.ProPhoto,
            bright=1.0, highlight_mode=2,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        )
        img = prophoto_linear.astype(np.float32) / 65535.0

    print(f"After decode: shape={img.shape}, mean={img.mean():.6f}")

    # Step 1.5: Exposure
    gain = 2.0 ** exposure
    img *= gain
    print(f"After exposure (+{exposure}): mean={img.mean():.6f}")

    # Step 2a: Gamut Transform
    log_color_space_name = LOG_TO_WORKING_SPACE.get(log_space)
    M = colour.matrix_RGB_to_RGB(
        colour.RGB_COLOURSPACES['ProPhoto RGB'],
        colour.RGB_COLOURSPACES[log_color_space_name],
    )
    print(f"\nGamut matrix (ProPhoto -> {log_color_space_name}):")
    for row in M:
        print(f"  [{row[0]:12.8f}, {row[1]:12.8f}, {row[2]:12.8f}]")

    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)
    utils.apply_matrix_inplace(img, M)
    print(f"After gamut: mean={img.mean():.6f}")

    # Step 2b: Log Encoding
    np.maximum(img, 1e-6, out=img)
    log_curve_name = LOG_ENCODING_MAP.get(log_space, log_space)
    img = colour.cctf_encoding(img, function=log_curve_name)
    print(f"After log curve ({log_curve_name}): mean={img.mean():.6f}")

    # Channel statistics
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    def stats(name, ch):
        print(f"  [{name}] range=[{ch.min():.6f}, {ch.max():.6f}] "
              f"mean={ch.mean():.6f} p50={np.median(ch):.6f} "
              f"p99={np.percentile(ch,99):.6f}")
    print("\n--- Python Log Output Statistics ---")
    stats("Red",   r)
    stats("Green", g)
    stats("Blue",  b)

    # Sample pixels at same positions
    h, w = img.shape[:2]
    samples = [
        (0, 0, "Top-left"),
        (0, w-1, "Top-right"),
        (h-1, 0, "Bottom-left"),
        (h//2, w//2, "Center"),
        (3*h//4, 3*w//4, "Three-quarter"),
    ]
    print("\n--- Sample Pixels ---")
    for row, col, label in samples:
        px = img[row, col]
        print(f"  {label:20s} ({col:4d},{row:4d}): R={px[0]:.4f} G={px[1]:.4f} B={px[2]:.4f}")

if __name__ == "__main__":
    main()
