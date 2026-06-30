// SPDX-License-Identifier: AGPL-3.0-or-later
// Implementation of the x-veon NN demosaic tile loop + dispatch entry point.
//
// Pipeline (design docs/nn-demosaic-design.md §2.3-§2.4):
//   1. param validation + session readiness
//   2. normalize CFA in place (working copy): (raw - black) / (white - black)
//   3. per-site white balance on the working copy
//   4. highlight reconstruction (post-WB, darktable-faithful position: opposed +
//      segmentation-based, per-channel clips scaled by WB)
//   5. mirror-pad top/left by (dy,dx) to phase-align to the canonical pattern,
//      and right/bottom so an integer tile grid covers the whole image
//   6. build canonical masks + trapezoid blend window (tile-invariant)
//   7. OpenMP-parallel tile loop: pack [1,4,288,288] input -> ORT Run ->
//      NaN guard -> trapezoid-weighted accumulate into RGB/weight buffers
//   8. hand off the accumulated buffers + geometry (caller finalizes: crop +
//      weight-normalize + color matrix + orientation flip)
//
// Highlight reconstruction (step 3) implements design §2.3 step 4 (inpaint-opposed,
// ported from darktable). Earlier revisions shipped without it and clipped highlights
// showed a magenta cast — the model was trained on non-clipped [0,1] data and
// hallucinated chromaticity into saturated regions after per-channel WB amplification.
// See nn_highlight_recon.cpp for the algorithm and the deliberate deviations from
// the darktable reference.

#include "demosaic_nn_xveon.h"

#include "nn_logging.h"
#include "nn_nan_guard.h"
#include "nn_postprocess.h"
#include "nn_preprocess.h"
#include "nn_highlight_recon.h"
#include "nn_highlight_segbased.h"
#include "nn_session.h"

#include <onnxruntime_cxx_api.h>  // Ort::Session, Ort::Value, Ort::Run
#include <chrono>
#include <cstring>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <vector>

namespace rawalchemy {

namespace {

// Reflect a coordinate into [0, extent) using "reflect-101" (no edge repeat).
// Used by mirror-pad: a coordinate that lands at -1 maps to 0, -2 to 1, etc.,
// and extent maps to extent-1, extent+1 to extent-2. Handles arbitrary
// out-of-range by folding through the 2*(extent-1) reflection period.
inline int mirrorCoord(int c, int extent) {
    if (extent <= 1) return 0;
    const int period = 2 * (extent - 1);
    int m = c % period;
    if (m < 0) m += period;
    if (m >= extent) m = period - m;
    return m;
}

// Number of tiles needed so a grid of stride-spaced PATCH_SIZE tiles fully
// covers `extent` pixels. Returns at least 1 (a single tile, possibly padded).
// Matches x-veon's overlap=stride tiling: tiles start at 0, stride, 2*stride...
// and the last tile must reach >= extent.
int tileCount(int extent) {
    if (extent <= NN_PATCH_SIZE) return 1;
    // First tile covers [0, PATCH); each additional tile extends coverage by
    // STRIDE. Need (extent - PATCH) more pixels after the first tile.
    const int extra = extent - NN_PATCH_SIZE;
    return 1 + (extra + NN_STRIDE - 1) / NN_STRIDE;
}

}  // namespace

NnDemosaicStatus nnDemosaic(const NnDemosaicInput& in, NnDemosaicOutput& out) {
    // --- Step 1a: param validation (cheap, fail-fast on programmer error) ---
    if (in.width <= 0 || in.height <= 0 || in.cfaMosaic == nullptr) {
        return NnDemosaicStatus::InvalidParam;
    }

    // --- Step 1b: session readiness (design §6.1: NN unavailable -> caller falls back) ---
    NnDemosaicSession& session = NnDemosaicSession::instance();
    if (!session.isReady()) {
        return NnDemosaicStatus::SessionNotReady;
    }

    CfaPhase phase = detectCfaPhase(in.filters);
    // Override the X-Trans pattern with the camera's actual one (not the hardcoded canonical)
    if (phase.isXtrans) {
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                phase.cameraPattern[i][j] = in.xtransPattern[i][j];
    }
    Ort::Session* ort = session.sessionForCfaPeriod(phase.period);
    if (ort == nullptr) {
        // Session ready but no model loaded for this CFA period (e.g. X-Trans
        // model path was empty at init). Treat as not-ready.
        return NnDemosaicStatus::SessionNotReady;
    }

    const int W = in.width;
    const int H = in.height;

    // --- Step 2: normalize CFA into the working copy: (raw - black) / (white - black)
    // range<=0 zeroes all sites (degenerate guard).
    const float black = in.blackLevel;
    const float range = in.whiteLevel - in.blackLevel;
    const float invRange = (range > 0.0f) ? (1.0f / range) : 0.0f;
    std::vector<float> workingCfa(static_cast<size_t>(W) * static_cast<size_t>(H));
    const size_t cfaPix = static_cast<size_t>(W) * static_cast<size_t>(H);
    for (size_t i = 0; i < cfaPix; ++i) {
        workingCfa[i] = (in.cfaMosaic[i] - black) * invRange;
    }

    // --- Step 3: per-site white balance (before recon — darktable pipeline order:
    // rawprepare → temperature/WB → highlights → demosaic).
    // After phase-align mirror-pad by (dy,dx), original pixel (y,x) lands at canonical
    // position (y+dy, x+dx), so its WB channel is the canonical color at that position.
    for (int y = 0; y < H; ++y) {
        const size_t rowBase = static_cast<size_t>(y) * W;
        for (int x = 0; x < W; ++x) {
            const int ch = canonicalCfaColor(y + phase.dy, x + phase.dx, phase);
            workingCfa[rowBase + x] *= in.wbRgb[ch];
        }
    }

    // --- Step 4: highlight reconstruction (post-WB, darktable-faithful position).
    // Per-channel clips scale with WB (clips[c] = clipFactor × wbRgb[c]). No-op for
    // well-exposed images (early-exit inside). Reconstructs before the NN model sees
    // the data, so the model never receives out-of-distribution clipped input.
    reconstructHighlightsOpposed(workingCfa.data(), W, H, phase, kNnHighlightClipFactor, in.wbRgb);

    // --- Step 4b: segmentation-based full-clip recovery (post-WB).
    // Opposed is a no-op where all 3 channels are clipped; segbased recovers texture
    // via gradient propagation. No-op when there aren't enough fully-clipped sensels.
    reconstructHighlightsSegmentBased(workingCfa.data(), W, H, phase, kNnHighlightClipFactor, in.wbRgb);

    // --- Step 5: mirror-pad to phase-aligned origin + integer tile grid ---
    const int alignedH = H + phase.dy;  // after top mirror-pad
    const int alignedW = W + phase.dx;  // after left mirror-pad
    const int ny = tileCount(alignedH);
    const int nx = tileCount(alignedW);
    const int paddedH = (ny - 1) * NN_STRIDE + NN_PATCH_SIZE;
    const int paddedW = (nx - 1) * NN_STRIDE + NN_PATCH_SIZE;

    std::vector<float> paddedCfa(static_cast<size_t>(paddedH) * static_cast<size_t>(paddedW));
    for (int py = 0; py < paddedH; ++py) {
        const int sy = mirrorCoord(py - phase.dy, H);  // map padded -> original
        for (int px = 0; px < paddedW; ++px) {
            const int sx = mirrorCoord(px - phase.dx, W);
            paddedCfa[static_cast<size_t>(py) * paddedW + px] =
                workingCfa[static_cast<size_t>(sy) * W + sx];
        }
    }

    // --- Step 6: canonical masks + blend window (both tile-invariant) ---
    constexpr int NPS = NN_PATCH_SIZE;
    constexpr int TILE_PIX = NPS * NPS;
    std::vector<float> maskR(TILE_PIX), maskG(TILE_PIX), maskB(TILE_PIX);
    makeCanonicalMasks(maskR.data(), maskG.data(), maskB.data(), phase);

    std::vector<float> blendW(TILE_PIX);
    makeTrapezoidWeights(blendW.data());

    // --- Accumulation buffers (over the full padded extent; cropped in step 7) ---
    std::vector<float> outAccum(static_cast<size_t>(paddedH) * paddedW * 3, 0.0f);
    std::vector<float> weightAccum(static_cast<size_t>(paddedH) * paddedW, 0.0f);

    // --- ORT plumbing: names + memory info are tile-invariant, build once ---
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr inName = ort->GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr outName = ort->GetOutputNameAllocated(0, allocator);
    const char* inNames[] = {inName.get()};
    const char* outNames[] = {outName.get()};
    const Ort::MemoryInfo memInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 4> inShape = {1, 4, NPS, NPS};

    // Early-exit flags shared across tile workers (design §6.2: NaN -> error).
    std::atomic<bool> nanDetected{false};
    std::atomic<bool> inferenceFailed{false};

    // --- Step 7: OpenMP-parallel tile loop ---
    // Thread-local tile buffers are hoisted out of the loop body so each worker
    // allocates them once, not per tile. ORT's Session::Run is thread-safe.
    // Per-tile inference time is accumulated atomically across workers for
    // diagnostic logging; correctness does not depend on the exact totals.
    std::atomic<long long> totalInferenceMs{0};
    std::atomic<size_t> tileCount{0};
#ifdef RA_USE_OPENMP
    #pragma omp parallel
#endif
    {
        std::vector<float> tileInput(static_cast<size_t>(4) * TILE_PIX);
        std::vector<float> cfaTile(static_cast<size_t>(TILE_PIX));

#ifdef RA_USE_OPENMP
        #pragma omp for collapse(2) schedule(dynamic)
#endif
        for (int ty = 0; ty < ny; ++ty) {
            for (int tx = 0; tx < nx; ++tx) {
                // Cheap early-exit: a prior tile already failed; skip the rest.
                if (nanDetected.load(std::memory_order_relaxed) ||
                    inferenceFailed.load(std::memory_order_relaxed)) {
                    continue;
                }

                const int originY = ty * NN_STRIDE;
                const int originX = tx * NN_STRIDE;

                // Extract the CFA tile from the padded buffer (contiguous rows).
                for (int ly = 0; ly < NPS; ++ly) {
                    const float* src =
                        &paddedCfa[static_cast<size_t>(originY + ly) * paddedW + originX];
                    float* dst = &cfaTile[static_cast<size_t>(ly) * NPS];
                    std::copy(src, src + NPS, dst);
                }

                packTileInput(tileInput.data(), cfaTile.data(),
                              maskR.data(), maskG.data(), maskB.data());

                // Bind the tile buffer as an ORT tensor (no copy: zero-copy view
                // of tileInput, valid as long as tileInput outlives Run()).
                Ort::Value inTensor = Ort::Value::CreateTensor<float>(
                    memInfo, tileInput.data(), tileInput.size(),
                    inShape.data(), inShape.size());

                std::vector<Ort::Value> outputs;
                auto _t0 = std::chrono::high_resolution_clock::now();
                try {
                    outputs = ort->Run(Ort::RunOptions{nullptr},
                                       inNames, &inTensor, 1, outNames, 1);
                } catch (const Ort::Exception&) {
                    inferenceFailed.store(true, std::memory_order_relaxed);
                    continue;
                } catch (const std::exception&) {
                    inferenceFailed.store(true, std::memory_order_relaxed);
                    continue;
                }
                auto _t1 = std::chrono::high_resolution_clock::now();
                totalInferenceMs.fetch_add(
                    std::chrono::duration_cast<std::chrono::milliseconds>(_t1 - _t0).count(),
                    std::memory_order_relaxed);
                tileCount.fetch_add(1, std::memory_order_relaxed);

                const float* outData = outputs[0].GetTensorData<float>();
                // NaN guard MUST run with -ffast-math disabled on the guard TU.
                if (nnOutputHasNaNInf(outData, static_cast<size_t>(3) * TILE_PIX)) {
                    nanDetected.store(true, std::memory_order_relaxed);
                    continue;
                }

                // Trapezoid-weighted accumulate into the padded RGB/weight
                // buffers. Output is planar [1,3,NPS,NPS]: channel c stride is
                // c*TILE_PIX. Tiles overlap, so the scattered writes are guarded
                // by `#pragma omp atomic` (correct under concurrent tiles).
                for (int ly = 0; ly < NPS; ++ly) {
                    const int py = originY + ly;
                    for (int lx = 0; lx < NPS; ++lx) {
                        const int px = originX + lx;
                        const size_t tileIdx = static_cast<size_t>(ly) * NPS + lx;
                        const float w = blendW[tileIdx];
                        const size_t outIdx = static_cast<size_t>(py) * paddedW + px;
                        const float r = outData[tileIdx] * w;
                        const float g = outData[TILE_PIX + tileIdx] * w;
                        const float b = outData[2 * TILE_PIX + tileIdx] * w;
#ifdef RA_USE_OPENMP
                        #pragma omp atomic
#endif
                        outAccum[outIdx * 3 + 0] += r;
#ifdef RA_USE_OPENMP
                        #pragma omp atomic
#endif
                        outAccum[outIdx * 3 + 1] += g;
#ifdef RA_USE_OPENMP
                        #pragma omp atomic
#endif
                        outAccum[outIdx * 3 + 2] += b;
#ifdef RA_USE_OPENMP
                        #pragma omp atomic
#endif
                        weightAccum[outIdx] += w;
                    }
                }
            }
        }
    }

    if (nanDetected.load(std::memory_order_acquire)) {
        return NnDemosaicStatus::NaNOutput;
    }
    if (inferenceFailed.load(std::memory_order_acquire)) {
        return NnDemosaicStatus::InferenceFailed;
    }

    // --- Step 8: hand off the accumulated buffers + geometry ---
    // Finalize (crop + weight-normalize + color matrix + orientation) is done by
    // the caller, fused into a single pass so no full-image camRGB intermediate
    // is materialized. See NnDemosaicOutput for the recovery formula.
    out.outAccum = std::move(outAccum);
    out.weightAccum = std::move(weightAccum);
    out.paddedW = paddedW;
    out.width = W;
    out.height = H;
    out.phaseDx = phase.dx;
    out.phaseDy = phase.dy;

    nnlog::info("[NN] demosaic: %zu tiles, total inference %lld ms",
                tileCount.load(), totalInferenceMs.load());
    return NnDemosaicStatus::Ok;
}

}  // namespace rawalchemy
