/**
 * cross_validate_lens.cpp — Compare lens correction ON vs OFF at pixel level.
 * Reads two TIFF files (16-bit RGB, uncompressed, any strip layout).
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

struct TiffImage {
    int width = 0, height = 0;
    std::vector<uint16_t> pixels; // RGB interleaved, row-major
};

static bool readTiff16(const char* path, TiffImage& img) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); return false; }

    bool le = (fgetc(f) == 'I'); fgetc(f); // byte order
    auto r16 = [&]() -> uint16_t { uint8_t b[2]; fread(b,1,2,f); return le ? b[0]|(uint16_t)(b[1]<<8) : (uint16_t)(b[0]<<8)|b[1]; };
    auto r32 = [&]() -> uint32_t { uint8_t b[4]; fread(b,1,4,f); return le ? b[0]|(b[1]<<8)|(b[2]<<16)|((uint32_t)b[3]<<24) : ((uint32_t)b[0]<<24)|(b[1]<<16)|(b[2]<<8)|b[3]; };

    uint16_t magic = r16();
    if (magic != 42) { fclose(f); return false; }
    uint32_t ifd_off = r32();

    fseek(f, ifd_off, SEEK_SET);
    uint16_t nentries = r16();

    int w=0, h=0, comp=1;
    uint32_t strip_off_ptr=0, strip_bc_ptr=0;
    uint32_t strip_off_count=0, strip_bc_count=0;

    for (int i = 0; i < nentries; i++) {
        uint16_t tag = r16();
        uint16_t type = r16();
        uint32_t count = r32();
        uint32_t val = r32();
        switch(tag) {
            case 256: w = val; break;
            case 257: h = val; break;
            case 258: /* BitsPerSample, assume 16 */ break;
            case 259: comp = val; break;
            case 273: strip_off_ptr = val; strip_off_count = count; break;
            case 279: strip_bc_ptr = val; strip_bc_count = count; break;
        }
    }

    if (w <= 0 || h <= 0 || comp != 1) {
        fprintf(stderr, "Unsupported TIFF: %s (w=%d h=%d comp=%d)\n", path, w, h, comp);
        fclose(f); return false;
    }

    img.width = w;
    img.height = h;
    img.pixels.resize((size_t)w * h * 3);

    // Read strip offsets
    std::vector<uint32_t> offsets(strip_off_count);
    if (strip_off_count == 1) {
        offsets[0] = strip_off_ptr;
    } else {
        fseek(f, strip_off_ptr, SEEK_SET);
        for (uint32_t i = 0; i < strip_off_count; i++) offsets[i] = r32();
    }

    // Read strip byte counts
    std::vector<uint32_t> bytecounts(strip_bc_count);
    if (strip_bc_count == 1) {
        bytecounts[0] = strip_bc_ptr;
    } else {
        fseek(f, strip_bc_ptr, SEEK_SET);
        for (uint32_t i = 0; i < strip_bc_count; i++) bytecounts[i] = r32();
    }

    // Read pixel data from strips
    size_t pixel_idx = 0;
    for (uint32_t s = 0; s < strip_off_count && pixel_idx < img.pixels.size(); s++) {
        fseek(f, offsets[s], SEEK_SET);
        size_t nPixels = bytecounts[s] / 2;
        size_t toRead = std::min(nPixels, img.pixels.size() - pixel_idx);
        fread(img.pixels.data() + pixel_idx, 2, toRead, f);
        pixel_idx += toRead;
    }

    fclose(f);

    // Byte swap if big-endian TIFF
    if (!le) for (auto& v : img.pixels) v = (v >> 8) | (v << 8);

    return pixel_idx == img.pixels.size();
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <off.tif> <on.tif>\n", argv[0]);
        return 1;
    }

    printf("=== Lens Correction Cross-Validation ===\n\n");

    TiffImage off, on;
    if (!readTiff16(argv[1], off)) return 1;
    if (!readTiff16(argv[2], on)) return 1;

    if (off.width != on.width || off.height != on.height) {
        printf("ERROR: Size mismatch %dx%d vs %dx%d\n", off.width, off.height, on.width, on.height);
        return 1;
    }

    int W = off.width, H = off.height;
    size_t N = (size_t)W * H * 3;

    printf("  Image: %d x %d = %.1f MP (16-bit RGB)\n\n", W, H, W*H/1e6);

    // 1. Overall statistics
    double diff_sum = 0, diff_sq = 0;
    int64_t max_diff = 0;
    size_t changed = 0;

    for (size_t i = 0; i < N; i++) {
        int64_t d = std::abs((int64_t)on.pixels[i] - (int64_t)off.pixels[i]);
        diff_sum += d;
        diff_sq += (double)d * d;
        if (d > max_diff) max_diff = d;
        if (d > 0) changed++;
    }

    double mean_diff = diff_sum / N;
    double mse = diff_sq / N;
    double psnr = (mse > 0) ? 10.0 * log10((65535.0 * 65535.0) / mse) : 999.0;

    printf("--- Overall Difference ---\n");
    printf("  Pixels changed:  %zu / %zu (%.2f%%)\n", changed, N, 100.0 * changed / N);
    printf("  Mean abs diff:   %.2f / 65535\n", mean_diff);
    printf("  Max abs diff:    %ld\n", (long)max_diff);
    printf("  PSNR:            %.2f dB\n\n", psnr);

    // 2. Region analysis (green channel)
    int margin = std::min(W, H) / 10;
    double corner_diff = 0, center_diff = 0;
    long corner_count = 0, center_count = 0;

    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            size_t idx = ((size_t)r * W + c) * 3 + 1; // Green
            int64_t d = std::abs((int64_t)on.pixels[idx] - (int64_t)off.pixels[idx]);

            bool is_corner = (r < margin || r >= H - margin) && (c < margin || c >= W - margin);
            bool is_center = (margin*3 <= r && r < H - margin*3) && (margin*3 <= c && c < W - margin*3);

            if (is_corner) { corner_diff += d; corner_count++; }
            else if (is_center) { center_diff += d; center_count++; }
        }
    }

    corner_diff /= corner_count;
    center_diff /= center_count;

    printf("--- Region Analysis (Green channel) ---\n");
    printf("  Corner mean diff: %.2f\n", corner_diff);
    printf("  Center mean diff: %.2f\n", center_diff);
    if (center_diff > 0)
        printf("  Corner/Center ratio: %.2fx\n\n", corner_diff / center_diff);

    // 3. Vignetting check
    struct Pt { int r, c; const char* name; };
    Pt pts[] = {
        {0, 0, "Top-left"}, {0, W-1, "Top-right"},
        {H-1, 0, "Bot-left"}, {H-1, W-1, "Bot-right"},
        {0, W/2, "Top-mid"}, {H/2, 0, "Left-mid"},
        {H/2, W-1, "Right-mid"}, {H-1, W/2, "Bot-mid"},
        {H/2, W/2, "CENTER"},
    };

    printf("--- Vignetting Correction Check (Green) ---\n");
    printf("  %-12s  %8s  %8s  %8s  %8s\n", "Position", "G_off", "G_on", "Delta", "Pct%");
    for (auto& p : pts) {
        size_t idx = ((size_t)p.r * W + p.c) * 3 + 1;
        int g_off = off.pixels[idx];
        int g_on  = on.pixels[idx];
        int delta = g_on - g_off;
        double pct = g_off > 0 ? 100.0 * delta / g_off : 0;
        printf("  %-12s  %8d  %8d  %+8d  %+.2f%%\n", p.name, g_off, g_on, delta, pct);
    }
    printf("\n");

    // 4. Verdict
    printf("--- Verdict ---\n");
    if (changed == 0) {
        printf("  FAIL: No pixels changed!\n");
    } else if (psnr > 50) {
        printf("  FAIL: PSNR too high, correction may be trivial\n");
    } else if (psnr < 15) {
        printf("  WARN: PSNR very low, correction very aggressive\n");
    } else {
        printf("  PASS: Measurable correction applied (PSNR=%.1f dB)\n", psnr);
    }

    if (corner_diff > center_diff * 1.2) {
        printf("  PASS: Corners changed more than center (vignetting + distortion OK)\n");
    } else {
        printf("  WARN: Corner/center ratio not as expected\n");
    }

    printf("\n");
    return 0;
}
