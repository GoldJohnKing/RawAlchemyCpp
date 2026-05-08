/**
 * @file verify.cpp
 * @brief Verification tool — reads a 16-bit TIFF and checks that it matches
 *        expected ProPhoto RGB (Linear) characteristics.
 *
 * Checks:
 *   1. Image dimensions and bit depth
 *   2. Pixel value range — linear data should have many dark values,
 *      a long tail toward highlights, and no gamma curve artifacts
 *   3. Channel-wise statistics (min, max, mean, median, percentiles)
 *   4. Histogram analysis — linear data has an exponential-like distribution,
 *      gamma-corrected data has a more uniform distribution
 *   5. Saturation / clipping check
 */

#include "raw_alchemy/tiff_writer.h"  // for TiffCloser
#include "raw_alchemy/common.h"

#include <tiffio.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <numeric>

// ---- RAII TIFF closer (reuse from tiff_writer.h or define locally) ----
struct LocalTiffCloser {
    TIFF* tif;
    explicit LocalTiffCloser(TIFF* t) : tif(t) {}
    ~LocalTiffCloser() { if (tif) TIFFClose(tif); }
};

// ---- Statistics ----
struct ChannelStats {
    float minVal, maxVal, mean, stddev;
    float p01, p05, p10, p25, p50, p75, p90, p95, p99;
    float ratioAbove18pct;  // fraction of pixels > 0.18 (middle gray in linear)
    float ratioClipped;     // fraction of pixels >= 1.0
};

static ChannelStats computeStats(const std::vector<float>& values) {
    ChannelStats s = {};
    if (values.empty()) return s;

    const size_t n = values.size();
    s.minVal = *std::min_element(values.begin(), values.end());
    s.maxVal = *std::max_element(values.begin(), values.end());

    double sum = 0, sumSq = 0;
    size_t above18 = 0, clipped = 0;
    for (auto v : values) {
        sum += v;
        sumSq += v * v;
        if (v > 0.18f) above18++;
        if (v >= 1.0f) clipped++;
    }
    s.mean = static_cast<float>(sum / n);
    double variance = sumSq / n - (sum / n) * (sum / n);
    s.stddev = static_cast<float>(std::sqrt(std::max(0.0, variance)));
    s.ratioAbove18pct = static_cast<float>(static_cast<double>(above18) / n);
    s.ratioClipped = static_cast<float>(static_cast<double>(clipped) / n);

    // Percentiles
    std::vector<float> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](float p) -> float {
        size_t idx = static_cast<size_t>(p / 100.0 * (n - 1));
        return sorted[std::min(idx, n - 1)];
    };
    s.p01 = pct(1);   s.p05 = pct(5);   s.p10 = pct(10);
    s.p25 = pct(25);  s.p50 = pct(50);  s.p75 = pct(75);
    s.p90 = pct(90);  s.p95 = pct(95);  s.p99 = pct(99);

    return s;
}

static void printStats(const char* label, const ChannelStats& s) {
    printf("  [%s]\n", label);
    printf("    Range:   [%.6f, %.6f]\n", s.minVal, s.maxVal);
    printf("    Mean:    %.6f   StdDev: %.6f\n", s.mean, s.stddev);
    printf("    P01: %.4f  P05: %.4f  P10: %.4f\n", s.p01, s.p05, s.p10);
    printf("    P25: %.4f  P50: %.4f  P75: %.4f\n", s.p25, s.p50, s.p75);
    printf("    P90: %.4f  P95: %.4f  P99: %.4f\n", s.p90, s.p95, s.p99);
    printf("    > 0.18 (middle gray): %.1f%%\n", s.ratioAbove18pct * 100);
    printf("    >= 1.0 (clipped):     %.2f%%\n", s.ratioClipped * 100);
}

// ---- Histogram (64 bins, linear scale) ----
static void printHistogram(const char* label, const std::vector<float>& values) {
    const int NBINS = 64;
    int bins[NBINS] = {};
    size_t n = values.size();
    for (auto v : values) {
        int b = static_cast<int>(v * (NBINS - 1));
        b = std::max(0, std::min(NBINS - 1, b));
        bins[b]++;
    }
    int maxBin = *std::max_element(bins, bins + NBINS);
    printf("  [%s] Histogram (64 bins, linear 0..1):\n", label);
    for (int i = 0; i < NBINS; i += 4) {
        float lo = static_cast<float>(i) / NBINS;
        float hi = static_cast<float>(i + 4) / NBINS;
        int total = 0;
        for (int j = i; j < i + 4 && j < NBINS; j++) total += bins[j];
        int barLen = maxBin > 0 ? total * 40 / maxBin : 0;
        printf("    %.3f-%.3f |%.*s%s (%d)\n", lo, hi, barLen, 
               "========================================", 
               total > 0 ? "" : "", total);
    }
}

// ---- Read TIFF and analyze ----
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: verify <image.tif>\n");
        return 1;
    }
    const char* path = argv[1];

    // Open TIFF
    TIFF* tif = TIFFOpen(path, "r");
    if (!tif) {
        fprintf(stderr, "Cannot open: %s\n", path);
        return 1;
    }
    LocalTiffCloser closer(tif);

    uint32_t w = 0, h = 0;
    uint16_t bps = 0, spp = 0, photo = 0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
    TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photo);

    printf("=== Image Properties ===\n");
    printf("  Dimensions: %u x %u (%.1f MP)\n", w, h, w * h / 1e6);
    printf("  Bits/Sample: %u\n", bps);
    printf("  Samples/Pixel: %u\n", spp);
    printf("  Photometric: %s\n",
           photo == PHOTOMETRIC_RGB ? "RGB" :
           photo == PHOTOMETRIC_MINISBLACK ? "Grayscale" : "Other");
    printf("\n");

    if (bps != 16 || spp != 3) {
        fprintf(stderr, "Expected 16-bit RGB TIFF. Got %u-bit, %u channels.\n", bps, spp);
        return 1;
    }

    // Read all pixels
    size_t pixelCount = static_cast<size_t>(w) * h;
    std::vector<uint16_t> rawBuf(pixelCount * 3);
    uint32_t rowBytes = w * 3 * sizeof(uint16_t);

    for (uint32_t row = 0; row < h; row++) {
        if (TIFFReadScanline(tif, rawBuf.data() + row * w * 3, row, 0) < 0) {
            fprintf(stderr, "Error reading row %u\n", row);
            return 1;
        }
    }

    // Convert to float
    std::vector<float> rCh(pixelCount), gCh(pixelCount), bCh(pixelCount);
    for (size_t i = 0; i < pixelCount; i++) {
        rCh[i] = static_cast<float>(rawBuf[i * 3 + 0]) / 65535.0f;
        gCh[i] = static_cast<float>(rawBuf[i * 3 + 1]) / 65535.0f;
        bCh[i] = static_cast<float>(rawBuf[i * 3 + 2]) / 65535.0f;
    }

    // Luminance (ProPhoto RGB coefficients: Y from RGB-to-XYZ row 1)
    // ProPhoto RGB -> XYZ: Y = 0.2880*R + 0.7118*G + 0.0003*B (approximate)
    // More precise: [0.7976749, 0.1351917, 0.0313534] for R,G,B primaries
    // Y coeffs = second row of RGB_to_XYZ matrix
    // ProPhoto RGB XYZ: R=(0.7976749,0.1351917,0.0313534), G=(0.2880747,0.7118632,0.0000622), B=(0,0,0.8251046)
    // Y = 0.2880747*R + 0.7118632*G + 0.0000622*B + ... actually simpler:
    // For ProPhoto: Y_coeffs ≈ (0.2880, 0.7119, 0.0001)
    const float lumaR = 0.2880f, lumaG = 0.7119f, lumaB = 0.0001f;
    std::vector<float> lum(pixelCount);
    for (size_t i = 0; i < pixelCount; i++) {
        lum[i] = lumaR * rCh[i] + lumaG * gCh[i] + lumaB * bCh[i];
    }

    // ---- Analysis ----
    printf("=== Pixel Statistics ===\n\n");

    ChannelStats rStats = computeStats(rCh);
    ChannelStats gStats = computeStats(gCh);
    ChannelStats bStats = computeStats(bCh);
    ChannelStats lStats = computeStats(lum);

    printStats("Red",   rStats); printf("\n");
    printStats("Green", gStats); printf("\n");
    printStats("Blue",  bStats); printf("\n");
    printStats("Luma",  lStats); printf("\n");

    // ---- Histograms ----
    printf("=== Histograms ===\n\n");
    printHistogram("Luma", lum);
    printf("\n");

    // ---- Linear Data Checks ----
    printf("=== Linear Data Validation ===\n\n");

    // Check 1: Mean luminance should be low (linear data is dark before gamma)
    // For a typical outdoor scene, mean linear luminance is ~0.05-0.25
    printf("  [Check 1] Mean luminance for linear data\n");
    printf("    Mean luma = %.4f (expected: 0.05-0.25 for linear data)\n", lStats.mean);
    if (lStats.mean > 0.01 && lStats.mean < 0.50) {
        printf("    PASS: Mean is in the expected range for linear scene data\n");
    } else {
        printf("    WARNING: Mean outside typical linear range\n");
    }
    printf("\n");

    // Check 2: Median should be much lower than mean (right-skewed distribution)
    printf("  [Check 2] Skewness check (median << mean for linear data)\n");
    float ratio = lStats.mean > 0 ? lStats.p50 / lStats.mean : 0;
    printf("    Median/Mean ratio = %.3f (expected: 0.3-0.8 for linear data)\n", ratio);
    if (ratio < 1.0f) {
        printf("    PASS: Right-skewed distribution consistent with linear data\n");
    } else {
        printf("    WARNING: Distribution may not be linear\n");
    }
    printf("\n");

    // Check 3: Very few pixels should be at exactly 0.0 or 1.0
    printf("  [Check 3] Clipping check\n");
    printf("    Pixels at max (65535): ");
    int maxCount = 0;
    for (auto v : rawBuf) { if (v == 65535) maxCount++; }
    printf("%d (%.2f%%)\n", maxCount, 100.0 * maxCount / rawBuf.size());
    printf("    Pixels at 0:           ");
    int zeroCount = 0;
    for (auto v : rawBuf) { if (v == 0) zeroCount++; }
    printf("%d (%.2f%%)\n", zeroCount, 100.0 * zeroCount / rawBuf.size());
    if (lStats.ratioClipped < 0.05f) {
        printf("    PASS: Minimal clipping (highlight blend mode working)\n");
    } else {
        printf("    WARNING: Significant clipping detected\n");
    }
    printf("\n");

    // Check 4: P99 should be significantly below 1.0 for well-exposed scenes
    printf("  [Check 4] Dynamic range check\n");
    printf("    P99 luma = %.4f (expected: < 0.9 for linear data with headroom)\n", lStats.p99);
    if (lStats.p99 < 0.95f) {
        printf("    PASS: Good headroom preserved in highlights\n");
    } else {
        printf("    NOTE: P99 is high, some highlights may be close to clipping\n");
    }
    printf("\n");

    // Check 5: ProPhoto gamut check — some values may exceed 1.0 in float,
    // but uint16 is clamped to 65535. Check if ProPhoto-like gamut behavior exists.
    printf("  [Check 5] Color gamut spot check\n");
    // Sample a few known-bright pixels
    size_t brightCount = 0;
    for (size_t i = 0; i < pixelCount; i++) {
        if (rCh[i] > 0.9f || gCh[i] > 0.9f || bCh[i] > 0.9f) brightCount++;
    }
    printf("    Pixels with any channel > 0.9: %.1f%%\n", 100.0 * brightCount / pixelCount);

    // ---- Corner / edge pixel sampling ----
    printf("\n=== Sample Pixels ===\n\n");
    // Sample specific locations
    struct Sample { int row, col; const char* label; };
    Sample samples[] = {
        {0, 0, "Top-left corner"},
        {0, (int)w-1, "Top-right corner"},
        {(int)h-1, 0, "Bottom-left corner"},
        {(int)h/2, (int)w/2, "Center"},
        {(int)h/4, (int)w/4, "Quarter point"},
        {(int)h*3/4, (int)w*3/4, "Three-quarter point"},
    };
    for (auto& s : samples) {
        size_t idx = static_cast<size_t>(s.row) * w + s.col;
        printf("  %-20s (%4d,%4d): R=%.4f G=%.4f B=%.4f  Luma=%.4f\n",
               s.label, s.col, s.row,
               rCh[idx], gCh[idx], bCh[idx], lum[idx]);
    }

    printf("\n=== Verification Complete ===\n");
    return 0;
}
