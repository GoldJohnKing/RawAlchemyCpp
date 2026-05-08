"""
Cross-validate JPEG output: C++ pipeline vs Python reference pipeline.

Runs the same pipeline (highlight-safe metering, F-Log2, LUT) in Python,
saves as JPEG, then compares the two JPEGs at the pixel level.

Usage:
    python Test/cross_validate_jpeg.py
"""
import rawpy
import numpy as np
import colour
import sys
import os
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Raw-Alchemy', 'src'))
from raw_alchemy.config import LOG_TO_WORKING_SPACE, LOG_ENCODING_MAP
from raw_alchemy import utils

def python_pipeline():
    """Run the full Python reference pipeline with highlight-safe metering."""
    raw_path = os.path.join(os.path.dirname(__file__), "DSC_5065.NEF")
    lut_path = os.path.join(os.path.dirname(__file__), "FLog2_to_CLASSIC-Neg._65grid_V.1.00.cube")
    ref_jpeg_path = os.path.join(os.path.dirname(__file__), "output_highlight_safe_ref.jpg")
    log_space = "F-Log2"

    print("=" * 60)
    print("Python Reference Pipeline (highlight-safe metering)")
    print("=" * 60)

    # Step 1: Decode RAW
    with rawpy.imread(raw_path) as raw:
        img = raw.postprocess(
            gamma=(1, 1), no_auto_bright=True, use_camera_wb=True,
            output_bps=16, output_color=rawpy.ColorSpace.ProPhoto,
            bright=1.0, highlight_mode=2,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        ).astype(np.float32) / 65535.0
    print(f"  [Step 1] Decoded: {img.shape}, mean={img.mean():.6f}")

    # Step 2: Highlight-safe auto exposure
    sample = utils.get_subsampled_view(img)
    max_vals = np.max(sample, axis=2)
    p99 = np.percentile(max_vals, 99.0)
    target_high = 0.9
    gain = target_high / max(p99, 1e-6)
    print(f"  [Step 2] Highlight-safe gain: {gain:.4f} (p99={p99:.6f})")
    utils.apply_gain_inplace(img, float(gain))

    # Step 3: Camera-Match Boost
    source_cs = colour.RGB_COLOURSPACES['ProPhoto RGB']
    utils.apply_saturation_and_contrast(img, saturation=1.25, contrast=1.1, colourspace=source_cs)
    print(f"  [Step 3] After boost: mean={img.mean():.6f}")

    # Step 4: Gamut + Log
    gamut = LOG_TO_WORKING_SPACE[log_space]
    M = colour.matrix_RGB_to_RGB(
        colour.RGB_COLOURSPACES['ProPhoto RGB'],
        colour.RGB_COLOURSPACES[gamut],
    )
    utils.apply_matrix_inplace(img, M)
    np.maximum(img, 1e-6, out=img)
    curve = LOG_ENCODING_MAP.get(log_space, log_space)
    img = colour.cctf_encoding(img, function=curve)
    print(f"  [Step 4] After log: mean={img.mean():.6f}")

    # Step 5: LUT
    lut = colour.read_LUT(lut_path)
    if isinstance(lut, colour.LUT3D):
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if lut.table.dtype != np.float32:
            lut.table = lut.table.astype(np.float32)
        utils.apply_lut_inplace(img, lut.table, lut.domain[0], lut.domain[1])
    else:
        img = lut.apply(img)
    print(f"  [Step 5] After LUT: mean={img.mean():.6f}")

    # Step 6: Save as JPEG (matching C++ settings: quality=95)
    np.clip(img, 0.0, 1.0, out=img)
    img_uint8 = (img * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(
        ref_jpeg_path,
        quality=95,
        subsampling=0,       # 4:4:4 (Python uses this; C++ stb uses 4:2:0)
        optimize=True,
    )
    print(f"  [Step 6] Saved reference JPEG: {ref_jpeg_path}")

    # Return float data for comparison
    return img_uint8, img


def load_cpp_jpeg():
    """Load the C++ JPEG output."""
    jpeg_path = os.path.join(os.path.dirname(__file__), "output_highlight_safe.jpg")
    img = np.array(Image.open(jpeg_path))
    print(f"  C++ JPEG loaded: {jpeg_path} shape={img.shape}")
    return img


def compare_images(cpp_img, ref_img_uint8):
    """Compare C++ JPEG against Python reference JPEG."""
    print("\n" + "=" * 60)
    print("JPEG Cross-Validation Results")
    print("=" * 60)

    # Basic shape check
    assert cpp_img.shape == ref_img_uint8.shape, \
        f"Shape mismatch: C++ {cpp_img.shape} vs Python {ref_img_uint8.shape}"
    h, w, c = cpp_img.shape
    print(f"  Dimensions: {w}x{h}x{c} — MATCH")

    # Per-channel statistics
    print("\n--- Per-Channel Statistics ---")
    ch_names = ["Red", "Green", "Blue"]
    for i, name in enumerate(ch_names):
        cpp_ch = cpp_img[:, :, i].astype(np.float32)
        ref_ch = ref_img_uint8[:, :, i].astype(np.float32)
        diff = cpp_ch - ref_ch
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))
        max_diff = np.max(np.abs(diff))
        print(f"  [{name:5s}] MAE={mae:.4f}  RMSE={rmse:.4f}  "
              f"MaxDiff={max_diff:.0f}  "
              f"C++=[{cpp_ch.min():.0f},{cpp_ch.max():.0f}]  "
              f"Py=[{ref_ch.min():.0f},{ref_ch.max():.0f}]")

    # Overall pixel-level comparison
    diff = cpp_img.astype(np.float32) - ref_img_uint8.astype(np.float32)
    abs_diff = np.abs(diff)

    mae = np.mean(abs_diff)
    rmse = np.sqrt(np.mean(diff ** 2))
    max_diff = np.max(abs_diff)

    # Tolerance bands (JPEG compression introduces quantization artifacts)
    exact_match = np.sum(abs_diff == 0)
    within_1 = np.sum(abs_diff <= 1)
    within_3 = np.sum(abs_diff <= 3)
    within_5 = np.sum(abs_diff <= 5)
    total = abs_diff.size

    print(f"\n--- Overall Error Metrics ---")
    print(f"  MAE:     {mae:.4f} /255")
    print(f"  RMSE:    {rmse:.4f} /255")
    print(f"  MaxDiff: {max_diff:.0f} /255")
    print(f"\n--- Tolerance Distribution (over {total:,} values) ---")
    print(f"  Exact (diff=0):   {exact_match:>10,}  ({exact_match/total*100:.2f}%)")
    print(f"  |diff| <= 1:      {within_1:>10,}  ({within_1/total*100:.2f}%)")
    print(f"  |diff| <= 3:      {within_3:>10,}  ({within_3/total*100:.2f}%)")
    print(f"  |diff| <= 5:      {within_5:>10,}  ({within_5/total*100:.2f}%)")

    # Sample pixel comparison
    print(f"\n--- Sample Pixel Comparison ---")
    samples = [
        (0, 0, "Top-left"), (0, w-1, "Top-right"),
        (h-1, 0, "Bottom-left"), (h//2, w//2, "Center"),
        (3*h//4, 3*w//4, "3/4 point"),
    ]
    print(f"  {'Location':20s} {'C++ RGB':30s} {'Python RGB':30s} {'MaxDiff':>8s}")
    for row, col, label in samples:
        cpp_px = cpp_img[row, col]
        ref_px = ref_img_uint8[row, col]
        md = max(abs(int(cpp_px[i]) - int(ref_px[i])) for i in range(3))
        print(f"  {label:20s} ({cpp_px[0]:3d},{cpp_px[1]:3d},{cpp_px[2]:3d})"
              f"{'':>7s}({ref_px[0]:3d},{ref_px[1]:3d},{ref_px[2]:3d})"
              f"{'':>7s}{md:>5d}")

    # Verdict
    # JPEG is lossy; even with same input, different encoders produce different output.
    # With 4:4:4 vs 4:2:0 subsampling difference, MAE < 3 is acceptable.
    print(f"\n--- Verdict ---")
    if mae < 3.0:
        print(f"  PASS - MAE ({mae:.4f}) is within JPEG quantization tolerance.")
        print(f"  Note: Differences are expected due to:")
        print(f"    - JPEG lossy compression (both sides)")
        print(f"    - Chroma subsampling: Python 4:4:4 vs C++ 4:2:0")
        print(f"    - Different JPEG encoder implementations")
    elif mae < 5.0:
        print(f"  ACCEPTABLE - MAE ({mae:.4f}) is within expected JPEG variance.")
        print(f"  Differences attributed to encoder/subsampling differences.")
    else:
        print(f"  FAIL - MAE ({mae:.4f}) exceeds JPEG quantization tolerance.")
        print(f"  This may indicate a pipeline error.")


def main():
    # 1. Run Python reference pipeline and save JPEG
    ref_img_uint8, ref_float = python_pipeline()

    # 2. Load C++ JPEG output
    cpp_img = load_cpp_jpeg()

    # 3. Compare
    compare_images(cpp_img, ref_img_uint8)


if __name__ == "__main__":
    main()
