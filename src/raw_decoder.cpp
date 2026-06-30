/**
 * @file raw_decoder.cpp
 * @brief LibRaw-based RAW decoder — Standardized Decoding implementation.
 *
 * Core Philosophy Step 1:
 *   Decode RAW files from any camera into a standardized, wide-gamut
 *   intermediate space — ProPhoto RGB (Linear) at 16-bit depth,
 *   then provide as float32 [0.0, 1.0].
 */

#include "raw_decoder.h"
#include "exif_injector.h"
#include "win_unicode.h"

#include <libraw/libraw.h>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>

#include "aligned_allocator.h"
#include "cfa_lookup.h"
#include "demosaic.h"

// Forward declarations
#include "demosaic_rcd.h"
#include "demosaic_markesteijn.h"
#include "denoise_xtrans.h"

// NN demosaic path (Plan B Task 3): dispatch + session + camRGB->ProPhoto adapter
#include "demosaic_dispatch.h"
#include "nn_session.h"
#include "nn_color_adapt.h"
#include "color_convert.h"  // computeProPhotoMatrix: LibRaw out_cam = prophoto_rgb · rgb_cam

namespace rawalchemy {

// ---- X-Trans denoise config (set before dcraw_process, read in the callback) ----
// denoiseXtransPreWB (the pre_scalecolors_cb hook) is a plain C callback
// void(void*); it cannot receive DecodeParams. decodeRaw fills this
// synchronously right before dcraw_process() fires the callback, so it is
// scoped to a single decode.
struct XtransDenoiseConfig {
    float threshold = 0.0f;  // darktable domain; 0.0 = skip denoise
};
namespace {
thread_local XtransDenoiseConfig tl_xtransDenoise;
} // namespace

// ---- Error helper ----
static void throwLibRawError(int ret, const char* context) {
    throw std::runtime_error(
        std::string("[RawDecoder] ") + context + " failed: " +
        std::string(libraw_strerror(ret))
    );
}

// ---- LibRaw upstream bug workaround ----
// When wavelet denoising is enabled (threshold > 0), LibRaw sets shrink=1 and
// halves iheight/iwidth. After wavelet_denoise(), pre_interpolate() upscales
// the buffer to full size and sets shrink=0, but NEVER updates iheight/iwidth.
static void fixDimensions(LibRaw* raw) {
    raw->imgdata.sizes.iheight = raw->imgdata.sizes.height;
    raw->imgdata.sizes.iwidth = raw->imgdata.sizes.width;
}

// ---- Custom demosaic callbacks (LibRaw native API) ----
// LibRaw calls these via interpolate_bayer_cb / interpolate_xtrans_cb,
// replacing its built-in demosaic entirely. LibRaw handles ALL preprocessing
// (FBDD, wavelet, WB, highlight) before the callback, and ALL postprocessing
// (mix_green, median filter, color conversion, gamma) after.

/// Extract single-channel CFA mosaic from LibRaw's 4-channel image[] buffer.
/// Uses fcol() semantics to determine which channel holds the CFA value.
static AlignedVector<float> extractCfa(const unsigned short (*image)[4],
                                        int w, int h, const LibRaw& raw,
                                        float whiteLevel = 65535.0f) {
    const size_t n = static_cast<size_t>(w) * h;
    AlignedVector<float> cfa(n);
    const bool xt = (raw.imgdata.idata.filters == 9);

    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            unsigned c;
            if (xt) {
                c = xtransColor(row, col, raw.imgdata.idata.xtrans);
            } else {
                c = bayerColor(row, col, raw.imgdata.idata.filters);
            }
            // Caller (decodeRawNn) uses raw2image_ex(1), which runs adjust_bl()
            // and subtracts the real per-channel black during the copy — matching
            // the classical dcraw_process path. So image[][c] is already
            // black-subtracted here; do NOT subtract again.
            cfa[static_cast<size_t>(row) * w + col] =
                static_cast<float>(image[static_cast<size_t>(row) * w + col][c]) / whiteLevel;
        }
    }
    return cfa;
}

/// Write demosaiced planar RGB back into LibRaw's 4-channel image[] buffer.
/// All 4 channels are filled so LibRaw's mix_green and convert_to_rgb work.
static void writeBackRgb(unsigned short (*image)[4],
                          const float* planarRgb, int w, int h) {
    const size_t n = static_cast<size_t>(w) * h;
    for (size_t i = 0; i < n; ++i) {
        auto clamp = [](float v) -> unsigned short {
            return static_cast<unsigned short>(
                std::max(0.0f, std::min(65535.0f, v * 65535.0f)));
        };
        image[i][0] = clamp(planarRgb[i]);
        image[i][1] = clamp(planarRgb[i + n]);
        image[i][2] = clamp(planarRgb[i + 2 * n]);
        image[i][3] = image[i][1];  // Gb = G for mix_green
    }
}

/// Write a denoised single-channel CFA mosaic back into LibRaw's 4-channel
/// image[] buffer, at each pixel's CFA channel only. Inverse of extractCfa();
/// used by the pre-WB denoise callback.
static void writeBackCfa(unsigned short (*image)[4], const float* cfa,
                          int w, int h, const LibRaw& raw, float whiteLevel) {
    const bool xt = (raw.imgdata.idata.filters == 9);
    auto clamp = [whiteLevel](float v) -> unsigned short {
        return static_cast<unsigned short>(
            std::max(0.0f, std::min(65535.0f, v * whiteLevel)));
    };
    for (int row = 0; row < h; ++row)
        for (int col = 0; col < w; ++col) {
            const unsigned c = xt ? xtransColor(row, col, raw.imgdata.idata.xtrans)
                                  : bayerColor(row, col, raw.imgdata.idata.filters);
            const size_t i = static_cast<size_t>(row) * w + col;
            image[i][c] = clamp(cfa[i]);
        }
}

/// Bayer custom demosaic callback (LibRaw interpolate_bayer_cb).
/// Reads CFA from image[], runs RCD, writes full RGB back.
static void customBayerDemosaic(void* ctx) {
    LibRaw* raw = static_cast<LibRaw*>(ctx);
    fixDimensions(raw);

    auto& img = raw->imgdata;
    int w = static_cast<int>(img.sizes.width);
    int h = static_cast<int>(img.sizes.height);

    auto (*image)[4] = reinterpret_cast<unsigned short (*)[4]>(img.image);
    if (!image) return;

    auto cfa = extractCfa(image, w, h, *raw);
    AlignedVector<float> rgb(static_cast<size_t>(3) * w * h);

    rcd_demosaic(cfa.data(), rgb.data(), w, h, img.idata.filters);

    writeBackRgb(image, rgb.data(), w, h);
}

/// X-Trans custom demosaic callback (LibRaw interpolate_xtrans_cb).
/// Reads CFA from image[], runs 3-pass Markesteijn, writes full RGB back.
static void customXtransDemosaic(void* ctx) {
    LibRaw* raw = static_cast<LibRaw*>(ctx);
    fixDimensions(raw);

    auto& img = raw->imgdata;
    int w = static_cast<int>(img.sizes.width);
    int h = static_cast<int>(img.sizes.height);

    auto (*image)[4] = reinterpret_cast<unsigned short (*)[4]>(img.image);
    if (!image) return;

    auto cfa = extractCfa(image, w, h, *raw);
    AlignedVector<float> rgb(static_cast<size_t>(3) * w * h);

    markesteijn_demosaic(cfa.data(), rgb.data(), w, h, img.idata.xtrans);

    writeBackRgb(image, rgb.data(), w, h);
}

/// Pre-WB X-Trans denoise callback (LibRaw pre_scalecolors_cb). Fires after
/// black-subtract / adjust_maximum / green_matching but BEFORE scale_colors
/// (white balance). imgdata.image[] here is black-subtracted raw CFA at native
/// sensor bit-depth (14-bit Fuji ~16383 max), so extractCfa is called with
/// img.color.maximum (not its 65535 default) to land the CFA directly in
/// darktable's [0,1 = raw white] domain where NOISE_ALL/threshold are calibrated;
/// writeBackCfa uses the matching maximum to round-trip back to native scale.
/// No-op for Bayer and for X-Trans when the threshold is <= 0.
static void denoiseXtransPreWB(void* ctx) {
    LibRaw* raw = static_cast<LibRaw*>(ctx);
    auto& img = raw->imgdata;
    if (!isXtrans(img.idata.filters)) return;
    if (tl_xtransDenoise.threshold <= 0.0f) return;

    // half_size (preview) decodes allocate image[] with stride iwidth≈width/2
    // (LibRaw raw2image: iwidth = (width+shrink)>>shrink). Indexing it with
    // sizes.width would read out of bounds and crash. Previews don't need
    // pre-WB denoise (and the half-size CFA is subsampled, so the 6x6 X-Trans
    // pattern wouldn't align anyway), so skip shrink/half_size decodes entirely.
    if (img.sizes.iwidth != img.sizes.width ||
        img.sizes.iheight != img.sizes.height)
        return;

    int w = static_cast<int>(img.sizes.width);
    int h = static_cast<int>(img.sizes.height);
    auto (*image)[4] = reinterpret_cast<unsigned short (*)[4]>(img.image);
    if (!image) return;

    // extractCfa divides by whiteLevel; at pre_scalecolors_cb the data is at native
    // sensor bit-depth, so pass img.color.maximum to land directly in darktable's
    // [0,1 = raw white] domain where NOISE_ALL/threshold are calibrated. writeBackCfa
    // uses the matching maximum to round-trip back to native scale.
    const float rawMax = static_cast<float>(img.color.maximum);
    if (!(rawMax > 0.0f)) return;  // corrupt/missing maximum → skip denoise safely

    // in/out must not alias across channels (see denoise_xtrans contract), so
    // denoise into a fresh buffer then write back to the CFA channel positions.
    auto cfa = extractCfa(image, w, h, *raw, rawMax);
    AlignedVector<float> denoised(static_cast<size_t>(w) * h);
    denoise_xtrans(cfa.data(), denoised.data(), w, h,
                   img.idata.xtrans, tl_xtransDenoise.threshold);
    writeBackCfa(image, denoised.data(), w, h, *raw, rawMax);
}

/// Pre-interpolate callback: just fixes iheight/iwidth (wavelet denoise bug).
static void preInterpolateFix(void* ctx) {
    fixDimensions(static_cast<LibRaw*>(ctx));
}

// LibRaw::callbacks is protected — expose via subclass.
class LibRawAccessor : public LibRaw {
public:
    using LibRaw::callbacks;
};

// ---- NN demosaic path (Plan B Task 3) ----
// When params.enableNnDemosaic && !halfSize, decodeRaw branches here BEFORE
// dcraw_process. We bypass LibRaw's demosaic + color conversion entirely:
// raw2image() gives black-subtracted CFA, the NN produces linear camRGB, and
// camRgbToProPhotoLinear merges us back into the classical pipeline's contract
// (linear ProPhoto [0,1]) so lens/vLog/LUT are unchanged.

#ifdef RA_ENABLE_NN_DEMOSAIC
/// Populate NnDemosaicInput.wbRgb + .xyzToCam from LibRaw color data.
static void fillNnMetadata(NnDemosaicInput& in, const LibRaw& raw) {
    const auto& color = raw.imgdata.color;
    // cam_mul[0..3] are R,G1,B,G2; G-normalize so the green multiplier is 1.
    const float g = color.cam_mul[1] > 0 ? color.cam_mul[1] : 1.0f;
    in.wbRgb[0] = color.cam_mul[0] / g;
    in.wbRgb[1] = 1.0f;
    in.wbRgb[2] = color.cam_mul[2] / g;
    // cam_xyz: LibRaw stores 4x3; take the leading 3x3 row-major.
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            in.xyzToCam[i * 3 + j] = static_cast<float>(color.cam_xyz[i][j]);
}

    // Fused finalize of the NN accumulated output: crop (pad offset) + normalize
    // by blend weight + camRGB->ProPhoto matrix + orientation flip, all in one
    // pass. Reads the NN's padded accumulators directly so no full-image camRGB
    // intermediate is materialized. flip==0 (landscape) takes a branch-free
    // sequential path; flip!=0 composes the crop offset with LibRaw's flip_index
    // (file_write.cpp:22-31) for the source read. The color matrix is position-
    // invariant, so finalize-and-rearrange is bit-identical to the old
    // crop -> normalize -> convert -> rotate chain.
    static void finalizeNnToProPhoto(float* dst, int dstW, int dstH,
                                     const float* outAccum,
                                     const float* weightAccum,
                                     int paddedW, int phaseDx, int phaseDy,
                                     int srcW, int srcH,
                                     int flip, const float outCam[9]) {
        if (flip == 0) {
            for (int y = 0; y < srcH; ++y) {
                const size_t padRow =
                    static_cast<size_t>(y + phaseDy) * paddedW + phaseDx;
                float* drow = dst + static_cast<size_t>(y) * srcW * 3;
                for (int x = 0; x < srcW; ++x) {
                    const size_t pIdx = padRow + x;
                    const float w = weightAccum[pIdx];
                    const float invW = (w > 0.0f) ? (1.0f / w) : 0.0f;
                    const float cr = outAccum[pIdx * 3 + 0] * invW;
                    const float cg = outAccum[pIdx * 3 + 1] * invW;
                    const float cb = outAccum[pIdx * 3 + 2] * invW;
                    const size_t d = static_cast<size_t>(x) * 3;
                    drow[d + 0] = outCam[0] * cr + outCam[1] * cg + outCam[2] * cb;
                    drow[d + 1] = outCam[3] * cr + outCam[4] * cg + outCam[5] * cb;
                    drow[d + 2] = outCam[6] * cr + outCam[7] * cg + outCam[8] * cb;
                }
            }
        } else {
            for (int orow = 0; orow < dstH; ++orow) {
                for (int ocol = 0; ocol < dstW; ++ocol) {
                    int r = orow, c = ocol;
                    if (flip & 4) std::swap(r, c);
                    if (flip & 2) r = srcH - 1 - r;
                    if (flip & 1) c = srcW - 1 - c;
                    const size_t pIdx =
                        static_cast<size_t>(r + phaseDy) * paddedW + (c + phaseDx);
                    const float w = weightAccum[pIdx];
                    const float invW = (w > 0.0f) ? (1.0f / w) : 0.0f;
                    const float cr = outAccum[pIdx * 3 + 0] * invW;
                    const float cg = outAccum[pIdx * 3 + 1] * invW;
                    const float cb = outAccum[pIdx * 3 + 2] * invW;
                    const size_t d = (static_cast<size_t>(orow) * dstW + ocol) * 3;
                    dst[d + 0] = outCam[0] * cr + outCam[1] * cg + outCam[2] * cb;
                    dst[d + 1] = outCam[3] * cr + outCam[4] * cg + outCam[5] * cb;
                    dst[d + 2] = outCam[6] * cr + outCam[7] * cg + outCam[8] * cb;
                }
            }
        }
    }

    /// Decode via the x-veon NN demosaic. Returns linear ProPhoto RGB [0,1].
    /// Throws std::runtime_error on any failure (caller surfaces via RaResult).
    static ImageBuffer decodeRawNn(LibRawAccessor& rawProcessor,
                                    const DecodeParams& params) {
    // raw2image_ex(1): copy CFA into imgdata.image[] AND subtract the real
    // per-channel black (runs adjust_bl() first, then subtracts cblack during
    // the copy — see LibRaw raw2image.cpp:427-432). This MUST match the
    // classical dcraw_process path, which uses the same raw2image_ex(1). The
    // plain raw2image() does NOT subtract black, leaving a per-channel offset
    // that — after WB amplifies R and B far more than G — renders as a strong
    // magenta cast (the model's residual CFA skip passes the imbalance through).
    int ret = rawProcessor.raw2image_ex(1);
    if (ret != LIBRAW_SUCCESS) {
        throwLibRawError(ret, "raw2image_ex (NN path)");
    }

    auto& img = rawProcessor.imgdata;
    const int w = static_cast<int>(img.sizes.width);
    const int h = static_cast<int>(img.sizes.height);

    auto (*image)[4] = reinterpret_cast<unsigned short (*)[4]>(img.image);
    if (!image) {
        throw std::runtime_error("[NN] imgdata.image is null after raw2image");
    }

    // One-time NN session init (idempotent singleton — Plan A's NnDemosaicSession).
    auto& sess = NnDemosaicSession::instance();
    if (!sess.isReady()) {
        NnSessionConfig cfg;
        cfg.bayerModelPath = params.nnBayerModelPath;
        cfg.xtransModelPath = params.nnXtransModelPath;
#ifdef _WIN32
        cfg.directmlDllPath = params.nnDirectmlDllPath;
        #elif defined(__ANDROID__)
            cfg.socModel = params.nnSocModel.empty() ? std::string{"0"} : params.nnSocModel;
            cfg.htpArch = params.nnHtpArch;
            cfg.ctxDir = params.nnCtxDir;
            cfg.appVersion = params.nnAppVersion;
        #endif
        if (!sess.init(cfg)) {
            throw std::runtime_error("[NN] session init failed");
        }
    }

    // extractCfa divides by whiteLevel, landing the CFA in [0,1]. nnDemosaic's
    // fused prep loop applies (raw-black)/(white-black) * WB; since the CFA is
    // already normalized and raw2image subtracted black, we pass blackLevel=0
    // and whiteLevel=1 so the normalize factor is identity (no double scale).
    const float rawMax = static_cast<float>(img.color.maximum);
    if (!(rawMax > 0.0f)) {
        throw std::runtime_error("[NN] imgdata.color.maximum is zero/invalid");
    }
    AlignedVector<float> cfa = extractCfa(image, w, h, rawProcessor, rawMax);

    NnDemosaicInput in{};
    in.width = w;
    in.height = h;
    in.filters = img.idata.filters;
    in.cfaMosaic = cfa.data();
    in.blackLevel = 0.0f;   // raw2image already subtracted black
    in.whiteLevel = 1.0f;   // cfa already normalized by extractCfa above
    fillNnMetadata(in, rawProcessor);

    // Copy the camera's ACTUAL X-Trans pattern — cameras ship different
    // rotations of the 6×6 arrangement; the hardcoded canonical does NOT match.
    if (img.idata.filters == 9) {
        for (int yy = 0; yy < 6; ++yy)
            for (int xx = 0; xx < 6; ++xx)
                in.xtransPattern[yy][xx] = img.idata.xtrans[yy][xx];
    }

    NnDemosaicOutput out;
    NnDemosaicStatus st = demosaicDispatch(in, out, DemosaicPath::Neural);
    if (st != NnDemosaicStatus::Ok) {
        const char* reason = "unknown";
        switch (st) {
            case NnDemosaicStatus::SessionNotReady: reason = "session not ready"; break;
            case NnDemosaicStatus::NaNOutput:       reason = "NaN output"; break;
            case NnDemosaicStatus::InferenceFailed: reason = "inference failed"; break;
            case NnDemosaicStatus::InvalidParam:    reason = "invalid param"; break;
            default: break;
        }
        throw std::runtime_error(std::string("[NN] demosaic failed: ") + reason);
    }

    // nnDemosaic now hands off its padded accumulators + geometry (no camRGB
    // intermediate). Finalize straight from them below: crop + weight-normalize
    // + camRGB->ProPhoto + orientation flip in one pass. nnDemosaic always
    // accumulates camRGB (outputCamRgb was removed in Plan A); we convert to
    // linear ProPhoto via LibRaw's rgb_cam (out_cam = prophoto_rgb · rgb_cam,
    // the same matrix the classical output_color=4 path uses) so we stay
    // bit-identical into vLog/LUT.

    // Allocate at display dimensions: the NN path bypasses LibRaw's flip (applied
    // in dcraw_process), so the orientation is folded into the finalize pass
    // below. flip==0 (landscape) takes a branch-free sequential read path.
    const int flip = img.sizes.flip;
    const bool transpose = (flip & 4) != 0;
    const int outW = transpose ? h : w;
    const int outH = transpose ? w : h;
    ImageBuffer result(outW, outH, 3);

    // camRGB→ProPhoto matching LibRaw convert_to_rgb() output_color=4 exactly:
    //   out_cam = prophoto_rgb · rgb_cam   (postprocessing_utils_dcrdefs.cpp:98-101)
    // rgb_cam is derived from cam_xyz by cam_xyz_coeff() (utils_dcraw.cpp:282-313),
    // so it already carries the correct D65 white-point normalization. The NN
    // output is cam_mul-WB'd (equivalent to scale_colors under use_camera_wb=1),
    // which is the input rgb_cam expects — apply directly. Do NOT add daylight
    // pre_mul on top (it would double-white-balance and re-introduce a cast).
    float outCam[3][4];
    computeProPhotoMatrix(img.color.rgb_cam, outCam);
    const float camToProPhoto[9] = {
        outCam[0][0], outCam[0][1], outCam[0][2],
        outCam[1][0], outCam[1][1], outCam[1][2],
        outCam[2][0], outCam[2][1], outCam[2][2]
    };
    finalizeNnToProPhoto(result.ptr(), outW, outH,
                         out.outAccum.data(), out.weightAccum.data(),
                         out.paddedW, out.phaseDx, out.phaseDy,
                         w, h, flip, camToProPhoto);

    return result;
}
#endif // RA_ENABLE_NN_DEMOSAIC

// ---- decodeRaw ----
ImageBuffer decodeRaw(const std::string& rawPath, const DecodeParams& params,
                       ExifCollector* exifCollector) {
    LibRawAccessor rawProcessor;

    // --- Configure decoding parameters ---
    auto& p = rawProcessor.imgdata.params;

    p.output_color = params.outputColor;
    p.gamm[0] = 1.0 / static_cast<double>(params.gammaPower);
    p.gamm[1] = static_cast<double>(params.gammaSlope);
    p.output_bps = params.outputBps;
    p.use_camera_wb = params.useCameraWb ? 1 : 0;
    p.no_auto_bright = params.noAutoBright ? 1 : 0;
    p.bright = static_cast<double>(params.bright);
    p.highlight = params.highlightMode;
    p.med_passes = params.medPasses;

    if (params.halfSize) {
        p.half_size = 1;
    }

    // --- Set EXIF callback before open_file ---
    if (exifCollector) {
        rawProcessor.set_exifparser_handler(
            rawalchemy::getExifCallback(), exifCollector);
    }

    // --- Open the RAW file ---
#ifdef _WIN32
    auto widePath = rawalchemy::utf8_to_wide(rawPath);
    int ret = rawProcessor.open_file(widePath.c_str());
#else
    int ret = rawProcessor.open_file(rawPath.c_str());
#endif
    if (ret != LIBRAW_SUCCESS) {
        throwLibRawError(ret, "open_file");
    }

    // --- Unpack RAW data ---
    ret = rawProcessor.unpack();
    if (ret != LIBRAW_SUCCESS) {
        throwLibRawError(ret, "unpack");
    }

    // --- Detect CFA type and register demosaic callbacks ---
    // We need to check filters AFTER open_file (which populates idata).
    const bool isXtransFile = (rawProcessor.imgdata.idata.filters == 9);
    const bool useCustomDemosaic = !params.halfSize && (
        params.demosaicAlgorithm == DemosaicAlgorithm::RCD ||
        params.demosaicAlgorithm == DemosaicAlgorithm::MARKESTEIJN_3PASS ||
        params.demosaicAlgorithm == DemosaicAlgorithm::AUTO
    );

    // Always fix iheight/iwidth (wavelet denoise bug workaround)
    rawProcessor.callbacks.pre_interpolate_cb = preInterpolateFix;
    // X-Trans pre-WB wavelet denoise (darktable port). Self-gates inside the
    // callback: no-op for Bayer and when threshold <= 0. Registered for ALL
    // files so X-Trans denoise works even in LIBRAW_FALLBACK demosaic mode.
    rawProcessor.callbacks.pre_scalecolors_cb = denoiseXtransPreWB;

    if (useCustomDemosaic) {
        // Register our custom demosaic callbacks. LibRaw will call the
        // appropriate one based on filters, replacing its built-in demosaic.
        // ALL preprocessing (FBDD/wavelet/WB) and postprocessing (mix_green,
        // median filter, color conversion, gamma) are handled by LibRaw.
        rawProcessor.callbacks.interpolate_bayer_cb = customBayerDemosaic;
        rawProcessor.callbacks.interpolate_xtrans_cb = customXtransDemosaic;
    } else {
        p.user_qual = params.demosaicQuality;
    }

    // Green matching: disable for X-Trans (LibRaw's implementation is
    // Bayer-only; running it on X-Trans corrupts the green channel).
    p.green_matching = params.greenMatching ? 1 : 0;

    // --- Noise reduction (Bayer: ISO-adaptive; X-Trans: flat 0.01, ISO-independent) ---
    float iso = rawProcessor.imgdata.other.iso_speed;

    // X-Trans: LibRaw's wavelet_denoise() inside scale_colors() is designed for
    // Bayer's 2x2 grid; applied to X-Trans's 6x6 CFA it produces row-dependent
    // wavelet artifacts (horizontal banding), so it stays disabled. Our own
    // darktable-ported channel-separated wavelet runs earlier, in the
    // pre_scalecolors_cb hook (before white balance — see Step 6).
    if (isXtransFile) {
        p.threshold = 0;
        tl_xtransDenoise.threshold =
            computeXtransDenoiseThreshold(iso, params.xtransDenoiseThreshold);
    } else {
        tl_xtransDenoise.threshold = 0.0f;
    }
    if (!isXtransFile && params.denoiseThreshold < 0.0f) {
        if (iso <= 100.0f) {
            p.threshold = 0;
        } else if (iso <= 400.0f) {
            p.threshold = 100;
        } else {
            float logLow = 8.644f;
            float logHigh = 13.644f;
            float t = (log2f(iso) - logLow) / (logHigh - logLow);
            p.threshold = 100.0f + std::min(std::max(t, 0.0f), 1.0f) * 900.0f;
        }
    } else if (!isXtransFile) {
        p.threshold = params.denoiseThreshold;
    }

    if (params.fbddNoiserd < 0) {
        // Note: LibRaw internally guards FBDD with "filters > 1000" (dcraw_process.cpp:169),
        // so FBDD never runs for X-Trans (filters == 9) regardless of this setting.
        if (iso <= 100.0f) {
            p.fbdd_noiserd = 0;
        } else if (iso < 3200.0f) {
            p.fbdd_noiserd = 1;
        } else {
            p.fbdd_noiserd = 2;
        }
    } else {
        p.fbdd_noiserd = params.fbddNoiserd;
    }

    // --- NN demosaic branch (final full-res only; preview always classical) ---
    // Branch BEFORE dcraw_process: the NN path calls raw2image() itself and
    // bypasses LibRaw's demosaic + color conversion, returning linear ProPhoto
    // directly. Unreachable unless params.enableNnDemosaic is set AND models are
    // loaded (Task 8 exercises it). Throws on failure; the CAPI layer wraps the
    // exception into an RaResult.
#ifdef RA_ENABLE_NN_DEMOSAIC
    if (params.enableNnDemosaic && !params.halfSize) {
        return decodeRawNn(rawProcessor, params);
    }
#endif

    // --- Process (demosaic + color conversion + gamma) ---
    // With our callbacks registered, LibRaw calls our RCD/Markesteijn
    // during the demosaic step, then continues with its own postprocessing.
    ret = rawProcessor.dcraw_process();
    if (ret != LIBRAW_SUCCESS) {
        throwLibRawError(ret, "dcraw_process");
    }

    // --- Extract processed image (standard LibRaw output path) ---
    int errCode = 0;
    libraw_processed_image_t* processed = rawProcessor.dcraw_make_mem_image(&errCode);
    if (!processed || errCode != LIBRAW_SUCCESS) {
        throw std::runtime_error(
            "[RawDecoder] dcraw_make_mem_image() failed (error code: " +
            std::to_string(errCode) + ")"
        );
    }

    if (processed->type != LIBRAW_IMAGE_BITMAP) {
        LibRaw::dcraw_clear_mem(processed);
        throw std::runtime_error("[RawDecoder] Expected bitmap output, got JPEG/other");
    }

    int imgWidth  = static_cast<int>(processed->width);
    int imgHeight = static_cast<int>(processed->height);
    int imgColors = static_cast<int>(processed->colors);
    if (imgColors < 3) imgColors = 3;

    // --- Convert to float32 ImageBuffer ---
    ImageBuffer result;

    if (processed->bits == 16) {
        const uint16_t* srcData = reinterpret_cast<const uint16_t*>(processed->data);
        if (imgColors == 3) {
            result = ImageBuffer::fromUint16(srcData, imgWidth, imgHeight, 3);
        } else {
            result = ImageBuffer(imgWidth, imgHeight, 3);
            const size_t pixelCount = result.pixelCount();
            for (size_t i = 0; i < pixelCount; ++i) {
                result.data[i * 3 + 0] = static_cast<float>(srcData[i * imgColors + 0]) / 65535.0f;
                result.data[i * 3 + 1] = static_cast<float>(srcData[i * imgColors + 1]) / 65535.0f;
                result.data[i * 3 + 2] = static_cast<float>(srcData[i * imgColors + 2]) / 65535.0f;
            }
        }
    } else if (processed->bits == 8) {
        const uint8_t* srcData = processed->data;
        result = ImageBuffer(imgWidth, imgHeight, 3);
        const size_t pixelCount = result.pixelCount();
        for (size_t i = 0; i < pixelCount; ++i) {
            result.data[i * 3 + 0] = static_cast<float>(srcData[i * imgColors + 0]) / 255.0f;
            result.data[i * 3 + 1] = static_cast<float>(srcData[i * imgColors + 1]) / 255.0f;
            result.data[i * 3 + 2] = static_cast<float>(srcData[i * imgColors + 2]) / 255.0f;
        }
    } else {
        LibRaw::dcraw_clear_mem(processed);
        throw std::runtime_error(
            "[RawDecoder] Unexpected bit depth: " + std::to_string(processed->bits)
        );
    }

    LibRaw::dcraw_clear_mem(processed);

    return result;
}

// ---- extractMetadata ----
CameraMetadata extractMetadata(const std::string& rawPath) {
    CameraMetadata meta;
    LibRaw rawProcessor;

#ifdef _WIN32
    auto widePath = rawalchemy::utf8_to_wide(rawPath);
    int ret = rawProcessor.open_file(widePath.c_str());
#else
    int ret = rawProcessor.open_file(rawPath.c_str());
#endif
    if (ret != LIBRAW_SUCCESS) {
        throwLibRawError(ret, "open_file (metadata)");
    }

    auto& idata = rawProcessor.imgdata.idata;
    auto& lens  = rawProcessor.imgdata.lens;
    auto& other = rawProcessor.imgdata.other;

    if (idata.make[0])  meta.cameraMaker  = idata.make;
    if (idata.model[0]) meta.cameraModel  = idata.model;
    if (lens.Lens[0])   meta.lensModel    = lens.Lens;
    if (lens.LensMake[0]) meta.lensMaker  = lens.LensMake;

    meta.focalLength = static_cast<float>(other.focal_len);
    meta.aperture    = static_cast<float>(other.aperture);
    meta.isoSpeed    = static_cast<int>(other.iso_speed);

    return meta;
}

} // namespace rawalchemy
