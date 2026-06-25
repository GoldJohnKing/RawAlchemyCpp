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

#include <onnxruntime_cxx_api.h>  // Ort::Env, Ort::Session, Ort::SessionOptions
#include <onnxruntime_session_options_config_keys.h>  // kOrtSessionOptionsDisableCPUEPFallback

#include <stdexcept>

#ifdef _WIN32
#include "win_unicode.h"  // utf8_to_wide

#include <windows.h>  // SetDllDirectoryA, GetFullPathNameA
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

// Forward-declared so the OrtDmlApi function-pointer signature below can name it
// without pulling in <d3d12.h> (we always pass nullptr — ORT creates the device).
struct ID3D12Device;

// Minimal ABI-matching definition of OrtDmlApi for ORT 1.24. The full struct
// ships in dml_provider_factory.h, which is only included in the DirectML ORT
// package (onnxruntime-win-x64-directml-*), NOT the generic GitHub release zip
// that this project vendors. GetExecutionProviderApi("DML",...) returns a
// pointer whose first (and, in 1.24, only) slot is this function pointer, so a
// single-member struct matches the ABI. Layout verified against the
// onnxruntime_c_api.h docstring for GetExecutionProviderApi (line ~3733):
//   "the provider_api pointer can be cast to the OrtDmlApi* when the
//    provider_name is 'DML'."
struct OrtDmlApi {
    OrtStatus* (*SessionOptionsAppendExecutionProvider_DML)(
        OrtSessionOptions* options, ID3D12Device* device,
        int device_id, bool bypass_runtime_pool_resource_lifetime_rules);
};
#endif

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
    // as an OrtApi struct member (Task 9 / C1 fix: the old
    // OrtApi::SessionOptionsAppendExecutionProvider_DML does not exist in 1.24).
    // GetExecutionProviderApi is exported by every onnxruntime.dll (generic or
    // DirectML build), so this links against the generic lib we vendor. With a
    // generic runtime DLL the call returns an error → Ort::ThrowOnError throws →
    // init() catches it → graceful fallback to traditional demosaic (design
    // sec 6.1). A DirectML-capable DLL is a deployment/packaging concern.
    const void* dmlApiVoid = nullptr;
    Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi(
        "DML", ORT_API_VERSION, &dmlApiVoid));
    const OrtDmlApi* dmlApi = static_cast<const OrtDmlApi*>(dmlApiVoid);
    // device=nullptr → ORT enumerates + creates the D3D12 device itself
    // (device_id 0 = primary DX12 adapter, via IDXGIFactory). bypass_...=false
    // uses ORT's standard DML resource-lifetime management. This is the
    // "simplest v1 form" from design sec 3.3 (ORT owns DMLCreateDevice).
    Ort::ThrowOnError(dmlApi->SessionOptionsAppendExecutionProvider_DML(
        static_cast<OrtSessionOptions*>(sopts),
        /*device=*/nullptr, /*device_id=*/0,
        /*bypass_runtime_pool_resource_lifetime_rules=*/false));

#elif defined(__ANDROID__)
    // --- QNN HTP FP16 EP (design sec 3.2) ---
    // No CPU EP fallback (design sec 3.4): an op the QNN HTP backend cannot
    // offload raises instead of silently running on slow CPU. Implemented as a
    // SESSION config entry (NOT a QNN provider option) per ORT docs.
    sopts.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
    sopts.SetExecutionMode(ORT_SEQUENTIAL);
    sopts.DisableMemPattern();

    // QNN provider options as key/value strings (ORT generic EP append).
    std::unordered_map<std::string, std::string> qnnOpts;
    qnnOpts.emplace("backend_path", "libQnnHtp.so");
    qnnOpts.emplace("htp_performance_mode", "burst");   // max-throughput for batch tile inference
    qnnOpts.emplace("enable_htp_fp16_precision", "1");  // feed fp32, HTP math in fp16 (design sec 3.2)
    qnnOpts.emplace("soc_model", cfg.socModel.empty() ? std::string{"0"} : cfg.socModel);
    qnnOpts.emplace("device_id", "0");
    sopts.AppendExecutionProvider("QNN", qnnOpts);

#else
    // --- Linux dev-loop: CPU EP ---
    // ORT defaults to the CPU EP; no explicit Append is needed. This is the
    // only branch that can be compiled AND run on the WSL2 host.
#endif

    return sopts;
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
    // A prior successful init is a no-op.
    if (ready_) {
        return true;
    }

    try {
        // One shared Env for all sessions. Threading mode is ORT default (multi).
        impl_->env = Ort::Env{ORT_LOGGING_LEVEL_WARNING, "rawalchemy-nn-demosaic"};

        Ort::SessionOptions sopts = makeSessionOptions(cfg);

        // ORTCHAR_T is wchar_t on Windows, char elsewhere. Model paths are stored
        // as UTF-8 and widened here for the Windows Session constructor.
#ifdef _WIN32
        if (!cfg.bayerModelPath.empty()) {
            std::wstring wide = utf8_to_wide(cfg.bayerModelPath);
            impl_->bayerSession = std::make_unique<Ort::Session>(impl_->env, wide.c_str(), sopts);
        }
        if (!cfg.xtransModelPath.empty()) {
            std::wstring wide = utf8_to_wide(cfg.xtransModelPath);
            impl_->xtransSession = std::make_unique<Ort::Session>(impl_->env, wide.c_str(), sopts);
        }
#else
        if (!cfg.bayerModelPath.empty()) {
            impl_->bayerSession = std::make_unique<Ort::Session>(impl_->env, cfg.bayerModelPath.c_str(), sopts);
        }
        if (!cfg.xtransModelPath.empty()) {
            impl_->xtransSession = std::make_unique<Ort::Session>(impl_->env, cfg.xtransModelPath.c_str(), sopts);
        }
#endif
    } catch (const Ort::Exception& e) {
        // EP unavailable, model missing, or graph build error. Per design sec 6.1
        // this is permanent: leave ready_ false and let the caller fall back.
        impl_->env = Ort::Env{};  // release the half-built Env
        impl_->bayerSession.reset();
        impl_->xtransSession.reset();
        ready_ = false;
        return false;
    } catch (const std::exception&) {
        impl_->env = Ort::Env{};
        impl_->bayerSession.reset();
        impl_->xtransSession.reset();
        ready_ = false;
        return false;
    }

    ready_ = true;
    return true;
}

bool NnDemosaicSession::isReady() const {
    return ready_;
}

Ort::Session* NnDemosaicSession::sessionForCfaPeriod(int period) {
    if (!ready_) {
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
