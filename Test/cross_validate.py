"""
Cross-validation: decode the same NEF file with Python rawpy (same params as C++)
and compare pixel statistics against the C++ output TIFF.
"""
import rawpy
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Raw-Alchemy', 'src'))
from raw_alchemy import utils

def main():
    raw_path = "/mnt/d/GitRepos/Raw_Log_Lut/Test/DSC_5065.NEF"
    
    print("=== Python rawpy Decoding (Reference) ===\n")
    
    # Exact same parameters as the C++ RawDecoder
    with rawpy.imread(raw_path) as raw:
        # Print metadata
        print(f"Camera: {raw.camera_params.make} {raw.camera_params.model}")
        print(f"Lens:   {raw.lens_params.make} {raw.lens_params.model}")
        print(f"Focal:  {raw.other_params.focal_len}")
        print(f"Aperture: f/{raw.other_params.aperture}")
        print(f"ISO:    {raw.other_params.iso_speed}")
        print()
        
        # Decode with SAME parameters as C++ code
        prophoto_linear = raw.postprocess(
            gamma=(1, 1),                    # Linear
            no_auto_bright=True,             # No auto brightness
            use_camera_wb=True,              # Camera WB
            output_bps=16,                   # 16-bit
            output_color=rawpy.ColorSpace.ProPhoto,  # ProPhoto RGB
            bright=1.0,                      # No brightness boost
            highlight_mode=2,                # Blend highlights
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,  # AHD
        )
        
        print(f"Output shape: {prophoto_linear.shape}")
        print(f"Output dtype: {prophoto_linear.dtype}")
        print()
        
        # Convert to float32 [0,1]
        img = prophoto_linear.astype(np.float32) / 65535.0
        
        # Channel statistics
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        
        def print_ch(name, ch):
            print(f"  [{name}]")
            print(f"    Range: [{ch.min():.6f}, {ch.max():.6f}]")
            print(f"    Mean:  {ch.mean():.6f}  StdDev: {ch.std():.6f}")
            print(f"    P50:   {np.median(ch):.6f}")
            print(f"    P99:   {np.percentile(ch, 99):.6f}")
            clipped = (ch >= 1.0).sum()
            print(f"    Clipped (>=1.0): {clipped} ({100*clipped/ch.size:.2f}%)")
            print()
        
        print("--- Python Channel Statistics ---")
        print_ch("Red",   r)
        print_ch("Green", g)
        print_ch("Blue",  b)
        
        # Luminance (ProPhoto RGB coefficients)
        source_cs = __import__('colour').RGB_COLOURSPACES['ProPhoto RGB']
        coeffs = utils.get_luminance_coeffs(source_cs)
        lum = np.dot(img, coeffs)
        
        print(f"  [Luma]")
        print(f"    Range: [{lum.min():.6f}, {lum.max():.6f}]")
        print(f"    Mean:  {lum.mean():.6f}  StdDev: {lum.std():.6f}")
        print(f"    P50:   {np.median(lum):.6f}")
        print(f"    P99:   {np.percentile(lum, 99):.6f}")
        print()
        
        # Sample pixels at same positions as C++ verify
        h, w = img.shape[:2]
        samples = [
            (0, 0, "Top-left"),
            (0, w-1, "Top-right"),
            (h-1, 0, "Bottom-left"),
            (h//2, w//2, "Center"),
            (h//4, w//4, "Quarter"),
            (3*h//4, 3*w//4, "Three-quarter"),
        ]
        print("--- Sample Pixels ---")
        for row, col, label in samples:
            px = img[row, col]
            l = coeffs[0]*px[0] + coeffs[1]*px[1] + coeffs[2]*px[2]
            print(f"  {label:20s} ({col:4d},{row:4d}): R={px[0]:.4f} G={px[1]:.4f} B={px[2]:.4f}  Luma={l:.4f}")
        
        print()
        print(f"Image shape: {h} x {w} = {h*w/1e6:.1f} MP")

if __name__ == "__main__":
    main()
