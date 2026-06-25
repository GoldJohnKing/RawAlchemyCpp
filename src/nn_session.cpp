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
//   - DirectML: OrtApi::SessionOptionsAppendExecutionProvider_DML(options, device_id)
//   - QNN:      Ort::SessionOptions::AppendExecutionProvider("QNN", unordered_map)
//               wrapping OrtApi::SessionOptionsAppendExecutionProvider(...)
//   - CPU-fallback disable: session config key kOrtSessionOptionsDisableCPUEPFallback
//               ("session.disable_cpu_ep_fallback") — a SESSION config entry, NOT
//               a QNN provider option (commonly misdocumented as the latter).

#include "nn_session.h"

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
#endif

}  // namespace

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
        env_ = Ort::Env{ORT_LOGGING_LEVEL_WARNING, "rawalchemy-nn-demosaic"};

        Ort::SessionOptions sopts = makeSessionOptions(cfg);

        // ORTCHAR_T is wchar_t on Windows, char elsewhere. Model paths are stored
        // as UTF-8 and widened here for the Windows Session constructor.
#ifdef _WIN32
        if (!cfg.bayerModelPath.empty()) {
            std::wstring wide = utf8_to_wide(cfg.bayerModelPath);
            bayerSession_ = std::make_unique<Ort::Session>(env_, wide.c_str(), sopts);
        }
        if (!cfg.xtransModelPath.empty()) {
            std::wstring wide = utf8_to_wide(cfg.xtransModelPath);
            xtransSession_ = std::make_unique<Ort::Session>(env_, wide.c_str(), sopts);
        }
#else
        if (!cfg.bayerModelPath.empty()) {
            bayerSession_ = std::make_unique<Ort::Session>(env_, cfg.bayerModelPath.c_str(), sopts);
        }
        if (!cfg.xtransModelPath.empty()) {
            xtransSession_ = std::make_unique<Ort::Session>(env_, cfg.xtransModelPath.c_str(), sopts);
        }
#endif
    } catch (const Ort::Exception& e) {
        // EP unavailable, model missing, or graph build error. Per design sec 6.1
        // this is permanent: leave ready_ false and let the caller fall back.
        env_ = Ort::Env{};  // release the half-built Env
        bayerSession_.reset();
        xtransSession_.reset();
        ready_ = false;
        return false;
    } catch (const std::exception&) {
        env_ = Ort::Env{};
        bayerSession_.reset();
        xtransSession_.reset();
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
        return bayerSession_.get();
    }
    if (period == NN_CFA_PERIOD_XTRANS) {
        return xtransSession_.get();
    }
    return nullptr;
}

Ort::SessionOptions NnDemosaicSession::makeSessionOptions(const NnSessionConfig& cfg) {
    Ort::SessionOptions sopts;
    sopts.SetIntraOpNumThreads(cfg.intraOpNumThreads);

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
    // device_index 0 = primary DX12 adapter (ORT enumerates via IDXGIFactory).
    Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_DML(
        static_cast<OrtSessionOptions*>(sopts), /*device_index=*/0));

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
    (void)cfg;
#endif

    return sopts;
}

}  // namespace rawalchemy
