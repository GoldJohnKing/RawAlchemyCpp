// SPDX-License-Identifier: AGPL-3.0-or-later
// Implementation of the ORT session singleton (see nn_session.h).
//
// EP selection is platform-conditional (design docs/nn-demosaic-design.md sec 3):
//   _WIN32     -> DirectML (device 0, app-local DLL via SetDllDirectoryA)
//   __ANDROID__-> QNN HTP FP16 (generic SessionOptionsAppendExecutionProvider,
//                "QNN" + key/value map; CPU fallback disabled via session config)
//   Linux      -> CPU EP (ORT default; no Append call needed)
//
// API provenance (ORT 1.24.1, verified against vendored headers +
// onnxruntime.ai/docs/execution-providers/):
//   - DirectML: OrtApi::GetExecutionProviderApi("DML", ORT_API_VERSION, &ptr)
//               → cast to const OrtDmlApi* → SessionOptionsAppendExecutionProvider_DML.
//               (SessionOptionsAppendExecutionProvider_DML is NOT an OrtApi struct
//               member in 1.24 — it migrated to the provider-factory pattern. The
//               OrtDmlApi struct ships only in the DirectML ORT package's
//               dml_provider_factory.h, which the generic GitHub release zip we
//               vendor omits, so the ABI-matching layout is defined locally below.)
//   - QNN:      Ort::SessionOptions::AppendExecutionProvider("QNN", unordered_map)
//               wrapping OrtApi::SessionOptionsAppendExecutionProvider(...)
//   - CPU-fallback disable: session config key kOrtSessionOptionsDisableCPUEPFallback
//               ("session.disable_cpu_ep_fallback") — a SESSION config entry, NOT
//               a QNN provider option (commonly misdocumented as the latter).
//
// PIMPL: all Ort:: members live in NnDemosaicSession::Impl below so that
// nn_session.h does not transitively include <onnxruntime_cxx_api.h>.

#include "nn_session.h"

#include "nn_logging.h"

#include <onnxruntime_cxx_api.h>  // Ort::Env, Ort::Session, Ort::SessionOptions
#include <onnxruntime_session_options_config_keys.h>  // kOrtSessionOptionsDisableCPUEPFallback

#include <chrono>
#include <cstdio>       // std::remove
#include <sys/stat.h>   // ::stat
#include <stdexcept>
#include <string>
#include <unordered_map>

#ifdef _WIN32
#include "win_unicode.h"  // utf8_to_wide

#include <windows.h>  // SetDllDirectoryA, GetFullPathNameA
#endif

#if defined(__ANDROID__)
#include <cstdlib>   // setenv, getenv
#include <dlfcn.h>   // dladdr
#endif

namespace rawalchemy {

namespace {

#ifdef _WIN32
// Returns the parent directory of a UTF-8 path (without trailing separator).
// Used only on Windows to derive the DirectML.dll search dir.
std::string parentDir(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    return (slash == std::string::npos) ? std::string{} : path.substr(0, slash);
}

// Hand-defined because the generic ORT GitHub-release package omits
// dml_provider_factory.h. Pinned to ORT_API_VERSION; replace with
// #include <dml_provider_factory.h> once the DirectML-capable onnxruntime
// package is adopted (Plan B). The real OrtDmlApi has 6 members in 1.24
// (_DML, _DML1, CreateGPUAllocationFromD3DResource, FreeGPUAllocation,
// GetD3D12ResourceFromAllocation, _DML2); only slot 0 (_DML, the device_id
// form) is used here.
struct OrtDmlApi {
    OrtStatus* (*SessionOptionsAppendExecutionProvider_DML)(
        OrtSessionOptions* options, int device_id);
};
#endif

#if defined(__ANDROID__)
// Point the cDSP FastRPC loader at the app's nativeLibraryDir so it can load
// the HTP skel (libQnnHtpVxxSkel.so). MainActivity's System.loadLibrary() only
// maps the skel into the CPU process; the DSP has its own filesystem view and
// without ADSP_LIBRARY_PATH reports "Failed to load skel, error: 4000" →
// QNN_DEVICE_ERROR_INVALID_CONFIG (ORT issue #21203). The skels ship in the same
// jniLibs dir as libraw_alchemy_core.so, so we derive that dir from our own .so
// path via dladdr (robust to the randomized /data/app/~~hash/ install dir).
static void ensureQnnLibraryPaths() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&ensureQnnLibraryPaths), &info) == 0 ||
        info.dli_fname == nullptr) {
        return;
    }
    std::string soPath(info.dli_fname);
    const auto slash = soPath.find_last_of('/');
    if (slash == std::string::npos) return;
    soPath.resize(slash);  // dirname → arm64 jniLibs dir holding the skels
    setenv("ADSP_LIBRARY_PATH", soPath.c_str(), /*overwrite=*/1);
    // LD_LIBRARY_PATH: CPU-side libs are already loaded via System.loadLibrary,
    // but prepend our dir so any QNN-internal dlopen of stub libs resolves too.
    std::string ldp = soPath;
    if (const char* existing = getenv("LD_LIBRARY_PATH"); existing && *existing) {
        ldp += ":";
        ldp += existing;
    }
    setenv("LD_LIBRARY_PATH", ldp.c_str(), /*overwrite=*/1);
}
#endif

// --- Runtime context-cache config (Android) ---
// Bump kNnOptsFingerprint whenever a compile-affecting QNN EP option changes
// (performance_mode / finalization_mode / vtcm / fp16 / shared_mem), so the
// on-disk context cache invalidates and rebuilds instead of going stale.
constexpr const char* kQnnRuntimeVersion = "2.42.0";
constexpr const char* kOrtVersion = "1.24.1";
constexpr const char* kNnOptsFingerprint = "burst-m3-vtcm8-fp16";

bool fileExists(const std::string& path) {
    if (path.empty()) return false;
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0;
}

// Builds SessionOptions with intra/inter thread pools pinned to 1 and the
// platform EP registered. Throws Ort::Exception (caught by init()) on EP
// registration failure.
//
// Thread pinning (design sec 3.4): the pipeline runs OpenMP-parallel tiles
// ABOVE the ORT session, so each session must itself be single-threaded
// (intra=1, inter=1) to avoid thread oversubscription across tile workers.
// intra is config-driven (defaults to 1); inter is hard-pinned to 1 since no
// x-veon model benefits from running independent graph ops concurrently.
Ort::SessionOptions makeSessionOptions(const NnSessionConfig& cfg) {
    Ort::SessionOptions sopts;
    sopts.SetIntraOpNumThreads(cfg.intraOpNumThreads);
    sopts.SetInterOpNumThreads(1);

#ifdef _WIN32
    // --- DirectML EP (design sec 3.3) ---
    // Bypass the stale System32 DirectML.dll (ORT issue #18831): prepend the
    // app-local DirectML.dll directory to the loader search path so the DML
    // device that ORT creates internally binds to OUR DLL. This is the
    // "simplest correct v1 form" — ORT owns DMLCreateDevice; we only steer the
    // loader. (The design's pre-create-device path via DML1 is a future
    // hardening, not needed for v1.)
    if (!cfg.directmlDllPath.empty()) {
        const std::string dir = parentDir(cfg.directmlDllPath);
        if (!dir.empty()) {
            SetDllDirectoryA(dir.c_str());
        }
    }
    // DirectML mandates sequential execution + disabled memory-pattern opt
    // (ORT DirectML docs). Applying unconditionally on Windows is harmless when
    // no DML adapter is found (init throws before these matter).
    sopts.SetExecutionMode(ORT_SEQUENTIAL);
    sopts.DisableMemPattern();

    // ORT 1.24 exposes DML via GetExecutionProviderApi("DML") → OrtDmlApi*, NOT
    // as an OrtApi struct member (the old
    // OrtApi::SessionOptionsAppendExecutionProvider_DML does not exist in 1.24).
    // GetExecutionProviderApi is exported by every onnxruntime.dll (generic or
    // DirectML build), so this links against the generic lib we vendor. With a
    // generic runtime DLL the call returns an error → Ort::ThrowOnError throws →
    // init() catches it → graceful fallback to traditional demosaic (design
    // sec 6.1). A DirectML-capable DLL is a deployment/packaging concern.
    // The 2-arg _DML form takes only device_id; ORT internally enumerates +
    // creates the D3D12 device (device_id 0 = primary DX12 adapter via
    // IDXGIFactory). This is the "simplest v1 form" from design sec 3.3 (ORT
    // owns DMLCreateDevice). GetExecutionProviderApi writes into a `const void*`
    // out-param (the ORT C API signature), which we then cast to OrtDmlApi*.
    const void* dmlApiVoid = nullptr;
    Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi(
        "DML", ORT_API_VERSION, &dmlApiVoid));
    // Defense-in-depth: GetExecutionProviderApi may return success but leave the
    // out-param null when the runtime DLL has no DML EP (generic ORT build). The
    // subsequent deref would crash with 0xc0000005 (access violation) before ORT
    // can surface an error. Convert it into a thrown exception so init() catches
    // it and the caller falls back to classical demosaic.
    if (dmlApiVoid == nullptr) {
        throw std::runtime_error(
            "[NN] DirectML EP not available in this ORT build (dmlApiVoid is null)");
    }
    const OrtDmlApi* dmlApi = static_cast<const OrtDmlApi*>(dmlApiVoid);
    Ort::ThrowOnError(dmlApi->SessionOptionsAppendExecutionProvider_DML(
        static_cast<OrtSessionOptions*>(sopts), /*device_id=*/0));

    #elif defined(__ANDROID__)
        // QNN HTP EP (Qualcomm NPU). socModel/htpArch forwarded from Build.SOC_MODEL.
        ensureQnnLibraryPaths();
        sopts.SetExecutionMode(ORT_SEQUENTIAL);
        sopts.DisableMemPattern();
        sopts.AddConfigEntry("session.record_ep_graph_assignment_info", "1");
        std::unordered_map<std::string, std::string> qnnOpts;
        qnnOpts.emplace("backend_path", "libQnnHtp.so");
        // burst (not sustained_high_performance): although batch processing runs
        // many photos, each RAW transfer over camera WiFi is the bottleneck
        // (seconds per file), giving the NPU ample cooling gaps between demosaics
        // — so burst's thermal-throttle/SSR risk doesn't materialize, and its
        // ~10% per-tile speed edge over sustained wins.
        qnnOpts.emplace("htp_performance_mode", "burst");
        qnnOpts.emplace("enable_htp_fp16_precision", "1");
        qnnOpts.emplace("soc_model", cfg.socModel.empty() ? std::string{"0"} : cfg.socModel);
        if (!cfg.htpArch.empty()) qnnOpts.emplace("htp_arch", cfg.htpArch);
        // mode 3 = most aggressive graph finalization (compile-time-heavy, but
        // with the runtime context cache the compile is paid once → near-free).
        qnnOpts.emplace("htp_graph_finalization_optimization_mode", "3");
        qnnOpts.emplace("vtcm_mb", "8");  // Hexagon v73 physical VTCM ceiling
        // NOTE: enable_htp_shared_memory_allocator was removed — it triggers a
        // FastRPC init→teardown→HANG on SM8550 / ORT 1.24.1 (the DMA-buf shared
        // memory allocator path is incompatible here; the compile thread blocks
        // indefinitely after fastrpc_apps_user_deinit). Re-enable only after
        // isolated verification on this device.
        qnnOpts.emplace("device_id", "0");
        sopts.AppendExecutionProvider("QNN", qnnOpts);

#else
    // --- Linux dev-loop: CPU EP ---
    // ORT defaults to the CPU EP; no explicit Append is needed. This is the
    // only branch that can be compiled AND run on the WSL2 host.
#endif

    return sopts;
}

// Verifies QNN EP actually engaged for this session. Throws if QNN took 0 nodes
// (i.e. silent full-CPU fallback). Deliberately preserves mixed inference (some
// ops still on CPU): only a complete QNN miss is rejected. Requires
// session.record_ep_graph_assignment_info="1" set in makeSessionOptions.
void verifyQnnEngaged(const Ort::Session& session, const char* modelName) {
    auto info = session.GetEpGraphAssignmentInfo();
    if (info.empty()) {
        // Recording produced nothing — cannot verify. Warn but don't fail
        // (guards against unexpected ORT behavior breaking all builds).
        nnlog::info("[NN] %s: EP graph info empty (cannot verify QNN engagement)", modelName);
        return;
    }
    size_t qnnNodes = 0, cpuNodes = 0;
    for (const auto& sg : info) {
        const std::string ep = sg.GetEpName();
        const auto n = sg.GetNodes().size();
        if (ep.find("QNN") != std::string::npos) {
            qnnNodes += n;
        } else if (ep.find("CPU") != std::string::npos) {
            cpuNodes += n;
        }
    }
    nnlog::info("[NN] %s placement: QNN nodes=%zu, CPU nodes=%zu (%s)",
                modelName, qnnNodes, cpuNodes,
                qnnNodes > 0 ? "NPU engaged" : "NPU NOT engaged");
    if (qnnNodes == 0) {
        throw std::runtime_error(
            std::string("[NN] ") + modelName +
            ": NPU not engaged — QNN EP took 0 nodes (full-CPU fallback disabled "
            "for verification). Check libQnnHtp*.so / libcdsprpc.so presence and model ops.");
    }
}

#if defined(__ANDROID__)
// Deterministic context-cache path. Empty when caching is disabled (no ctx dir
// or app version). Filename embeds every version dimension so any change — app
// update, QNN/ORT lib bump, SoC arch, compile opts — forces a rebuild.
std::string buildCtxPath(const NnSessionConfig& cfg, const std::string& modelTag) {
    if (cfg.ctxDir.empty() || cfg.appVersion.empty()) return {};
    const std::string& arch = cfg.htpArch.empty() ? std::string{"0"} : cfg.htpArch;
    return cfg.ctxDir + "/" + modelTag +
           "_v" + cfg.appVersion +
           "_qnn" + kQnnRuntimeVersion +
           "_ort" + kOrtVersion +
           "_arch" + arch +
           "_" + kNnOptsFingerprint +
           ".ctx.onnx";
}
#endif

// Loads a session either from a cached QNN context binary (fast: ~200ms) or by
// compiling the original model with cache generation enabled (slow: once per
// cache-key lifetime). A failed cached-load deletes the stale file and falls
// back to recompilation (self-healing). When caching is disabled (non-Android
// or missing ctx dir), behaves as a plain session constructor + (on Android)
// NPU-engagement verify, exactly like the previous behavior.
std::unique_ptr<Ort::Session> createSessionWithCache(
    Ort::Env& env, const std::string& modelPath, const std::string& modelTag,
    const NnSessionConfig& cfg) {

#if defined(__ANDROID__)
    const std::string ctxPath = buildCtxPath(cfg, modelTag);

    // Fast path: existing cache → load it. Same base SessionOptions (QNN EP
    // registered) but NO ep.context_enable — we consume, not generate.
    if (!ctxPath.empty() && fileExists(ctxPath)) {
        try {
            Ort::SessionOptions opts = makeSessionOptions(cfg);
            auto session = std::make_unique<Ort::Session>(env, ctxPath.c_str(), opts);
            nnlog::info("[NN] %s context loaded from cache: %s", modelTag.c_str(), ctxPath.c_str());
            return session;  // cached context is QNN by construction
        } catch (const std::exception& e) {
            nnlog::info("[NN] %s cache load failed (%s); regenerating",
                        modelTag.c_str(), e.what());
            std::remove(ctxPath.c_str());  // corrupt/stale → self-heal
            // fall through to compile path
        }
    }
#endif

    // Compile path.
    Ort::SessionOptions opts = makeSessionOptions(cfg);
#if defined(__ANDROID__)
    if (!ctxPath.empty()) {
        // Generate the cache while compiling. embed_mode=1 → single .ctx.onnx
        // file (no separate .bin to manage); fine for these small models.
        opts.AddConfigEntry("ep.context_embed_mode", "1");
        opts.AddConfigEntry("ep.context_enable", "1");
        opts.AddConfigEntry("ep.context_file_path", ctxPath.c_str());
        nnlog::info("[NN] %s compiling + caching context → %s",
                    modelTag.c_str(), ctxPath.c_str());
    }
#endif
    std::unique_ptr<Ort::Session> session;
#ifdef _WIN32
    // ORTCHAR_T is wchar_t on Windows: widen the UTF-8 model path.
    std::wstring wide = utf8_to_wide(modelPath);
    session = std::make_unique<Ort::Session>(env, wide.c_str(), opts);
#else
    session = std::make_unique<Ort::Session>(env, modelPath.c_str(), opts);
#endif
#if defined(__ANDROID__)
    verifyQnnEngaged(*session, modelTag.c_str());
#endif
    return session;
}

}  // namespace

// --- PIMPL: the ORT-typed members live here, invisible to nn_session.h users. ---
struct NnDemosaicSession::Impl {
    Ort::Env env;  // default-constructed (p_ = nullptr) until init()
    std::unique_ptr<Ort::Session> bayerSession;
    std::unique_ptr<Ort::Session> xtransSession;
};

// Impl is complete in this TU, so unique_ptr<Impl>'s destructor instantiates.
NnDemosaicSession::NnDemosaicSession() : impl_(std::make_unique<Impl>()) {}
NnDemosaicSession::~NnDemosaicSession() = default;

NnDemosaicSession& NnDemosaicSession::instance() {
    // Meyers singleton: initialization is thread-safe and deferred to first use.
    static NnDemosaicSession inst;
    return inst;
}

bool NnDemosaicSession::init(const NnSessionConfig& cfg) {
    // Fast path: already ready (lock-free, acquire so the compile-side release
    // is visible).
    if (ready_.load(std::memory_order_acquire)) {
        return true;
    }

    // Serialize concurrent callers (background warmup + first edit). The first
    // through the lock compiles; latecomers block, then observe ready on re-check.
    std::lock_guard<std::mutex> lk(initMutex_);
    if (ready_.load(std::memory_order_relaxed)) {
        return true;  // double-check under lock
    }

    try {
        auto t0 = std::chrono::high_resolution_clock::now();
        // One shared Env for all sessions.
        impl_->env = Ort::Env{ORT_LOGGING_LEVEL_WARNING, "rawalchemy-nn-demosaic"};

        // Each session is built via createSessionWithCache: on Android it serves
        // the cached QNN context (fast) or compiles+caches (slow once); the cache
        // path skips EP verification (cached context is QNN by construction), the
        // compile path verifies inside. On Windows/Linux this is a plain ctor.
        if (!cfg.bayerModelPath.empty()) {
            impl_->bayerSession = createSessionWithCache(impl_->env, cfg.bayerModelPath, "bayer", cfg);
        }
        if (!cfg.xtransModelPath.empty()) {
            impl_->xtransSession = createSessionWithCache(impl_->env, cfg.xtransModelPath, "xtrans", cfg);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        nnlog::info("[NN] session init took %lld ms", static_cast<long long>(ms));
    } catch (const std::exception& e) {
        // Honor the header contract: init() never throws — log, clean up,
        // return false so callers route to traditional demosaic. (Previously
        // this threw std::runtime_error, contradicting the "never throws" doc.)
        nnlog::info("[NN] session init failed: %s", e.what());
        impl_->env = Ort::Env{};
        impl_->bayerSession.reset();
        impl_->xtransSession.reset();
        // ready_ stays false; a later init() call may re-attempt.
        return false;
    }

    ready_.store(true, std::memory_order_release);
    return true;
}

bool NnDemosaicSession::isReady() const {
    return ready_.load(std::memory_order_acquire);
}

Ort::Session* NnDemosaicSession::sessionForCfaPeriod(int period) {
    if (!ready_.load(std::memory_order_acquire)) {
        return nullptr;
    }
    if (period == NN_CFA_PERIOD_BAYER) {
        return impl_->bayerSession.get();
    }
    if (period == NN_CFA_PERIOD_XTRANS) {
        return impl_->xtransSession.get();
    }
    return nullptr;
}

}  // namespace rawalchemy
