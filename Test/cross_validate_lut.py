"""
Cross-validate Step 3 (LUT Application) against Python reference.
Full pipeline: RAW -> ProPhoto Linear -> Exposure -> F-Log2 -> LUT -> stats
"""
import rawpy, numpy as np, colour, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Raw-Alchemy', 'src'))
from raw_alchemy.config import LOG_TO_WORKING_SPACE, LOG_ENCODING_MAP
from raw_alchemy import utils

def main():
    raw_path = "/mnt/d/GitRepos/Raw_Log_Lut/Test/DSC_5065.NEF"
    lut_path = "/mnt/d/GitRepos/Raw_Log_Lut/Test/FLog2_to_CLASSIC-Neg._65grid_V.1.00.cube"
    log_space = "F-Log2"
    exposure = 5.0

    print(f"=== Python Reference: {log_space} + LUT ===\n")

    with rawpy.imread(raw_path) as raw:
        img = raw.postprocess(
            gamma=(1,1), no_auto_bright=True, use_camera_wb=True,
            output_bps=16, output_color=rawpy.ColorSpace.ProPhoto,
            bright=1.0, highlight_mode=2, demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        ).astype(np.float32) / 65535.0

    # Exposure
    img *= 2.0 ** exposure
    print(f"After exposure: mean={img.mean():.6f}")

    # Gamut
    gamut = LOG_TO_WORKING_SPACE[log_space]
    M = colour.matrix_RGB_to_RGB(colour.RGB_COLOURSPACES['ProPhoto RGB'], colour.RGB_COLOURSPACES[gamut])
    utils.apply_matrix_inplace(img, M)

    # Log curve
    np.maximum(img, 1e-6, out=img)
    curve = LOG_ENCODING_MAP.get(log_space, log_space)
    img = colour.cctf_encoding(img, function=curve)
    print(f"After log: mean={img.mean():.6f}")

    # LUT
    lut = colour.read_LUT(lut_path)
    if isinstance(lut, colour.LUT3D):
        if lut.table.dtype != np.float32:
            lut.table = lut.table.astype(np.float32)
        utils.apply_lut_inplace(img, lut.table, lut.domain[0], lut.domain[1])
    else:
        img = lut.apply(img)
    print(f"After LUT: mean={img.mean():.6f}\n")

    # Stats
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    def stats(name, ch):
        print(f"  [{name}] range=[{ch.min():.6f}, {ch.max():.6f}] "
              f"mean={ch.mean():.6f} p50={np.median(ch):.6f}")
    print("--- Python LUT Output Statistics ---")
    stats("Red", r)
    stats("Green", g)
    stats("Blue", b)

    # Sample pixels
    h, w = img.shape[:2]
    samples = [(0,0,"Top-left"), (0,w-1,"Top-right"), (h-1,0,"Bottom-left"),
               (h//2,w//2,"Center"), (3*h//4,3*w//4,"Three-quarter")]
    print("\n--- Sample Pixels ---")
    for row, col, label in samples:
        px = img[row, col]
        print(f"  {label:20s} ({col:4d},{row:4d}): R={px[0]:.4f} G={px[1]:.4f} B={px[2]:.4f}")

if __name__ == "__main__":
    main()
