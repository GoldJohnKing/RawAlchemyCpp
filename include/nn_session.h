// SPDX-License-Identifier: AGPL-3.0-or-later
// ORT session singleton for the x-veon NN demosaic.
// Owns the Ort::Env plus two Ort::Sessions (bayer.onnx + xtrans.onnx) and
// registers the platform execution provider:
//   Android  -> QNN HTP FP16  (design docs/nn-demosaic-design.md sec 3.2; SoC-gated upstream)
//   Windows  -> DirectML      (sec 3.3; app-local DLL, primary DX12 adapter)
//   Linux    -> CPU EP        (dev-loop only; no GPU EP is wired on Linux)
// EP/session init failure is permanent: isReady() returns false and the pipeline
// falls back to the traditional demosaic (sec 3.4 / 6.1). init() never throws.
//
// PIMPL: the heavy Ort:: types (Env, Session) live only in nn_session.cpp, so
// including this header does NOT pull the ORT C++ wrapper header into callers.
// The only Ort:: type that appears here is Session, as an opaque pointer return.
#pragma once
#include <memory>
#include <string>

// Forward-declared only — the complete definitions live in nn_session.cpp, which
// is the sole TU that pulls in the ORT C++ wrapper. This keeps the ORT headers
// out of every caller's translation unit.
namespace Ort {
class Env;
class Session;
}  // namespace Ort

namespace rawalchemy {

/** CFA period constants: 2 for Bayer RGGB, 6 for X-Trans 6x6. */
static constexpr int NN_CFA_PERIOD_BAYER = 2;
static constexpr int NN_CFA_PERIOD_XTRANS = 6;

/** Configuration for the NN demosaic session singleton. Model paths are UTF-8
 *  (converted to wide on Windows where ORTCHAR_T == wchar_t). */
struct NnSessionConfig {
    std::string bayerModelPath;   // bayer.onnx (period-2 CFA model)
    std::string xtransModelPath;  // xtrans.onnx (period-6 X-Trans model)
    int intraOpNumThreads = 1;    // ORT intra-op thread pool size (inter-op pinned to 1; see .cpp)

#ifdef _WIN32
    // App-local DirectML.dll (design sec 3.3). Its directory is prepended to the
    // DLL search path so ORT loads OUR DirectML.dll, not a stale System32 copy.
    std::string directmlDllPath;
#elif defined(__ANDROID__)
    // Build.SOC_MODEL forwarded as QNN "soc_model" (helps graph-finalization).
    // "0" (unknown) is the ORT default and is safe.
    std::string socModel = "0";
    std::string htpArch;     // "73"/"75"/... Empty = let QNN infer.
    std::string ctxDir;      // app data dir for context cache. Empty = no caching.
    std::string appVersion;  // cache key component.
#endif
};

/** Singleton wrapping the ONNX Runtime lifecycle for the two x-veon demosaic
 *  models. Thread-safe after init(): construction is lazy (Meyers singleton),
 *  and init() must be called once from a single thread before any inference.
 *  Inference itself (Run) is thread-safe per ORT's contract. */
class NnDemosaicSession {
public:
    static NnDemosaicSession& instance();

    /** Initialize both sessions. Returns true only if BOTH models load AND the
     *  platform EP registers without error. Any failure (EP unavailable, model
     *  file missing, ORT error) returns false; the caller treats false as
     *  permanent and routes to traditional demosaic. Idempotent: a no-op after
     *  success; re-attempts after failure. */
    bool init(const NnSessionConfig& cfg);

    /** True after a successful init(). */
    bool isReady() const;

    /** Returns the session matching the CFA period (2 -> Bayer, 6 -> X-Trans).
     *  Returns nullptr if not ready or period is unsupported. Ownership stays
     *  with the singleton; do not delete. The returned Ort::Session* is opaque
     *  here — callers that drive inference must include the ORT C++ header
     *  themselves to use the pointer. */
    Ort::Session* sessionForCfaPeriod(int period);

private:
    NnDemosaicSession();
    ~NnDemosaicSession();
    NnDemosaicSession(const NnDemosaicSession&) = delete;
    NnDemosaicSession& operator=(const NnDemosaicSession&) = delete;

    // PIMPL: holds the Ort::Env and Ort::Session objects. Defined in the .cpp so
    // the ORT C++ wrapper header never reaches callers of this header.
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool ready_ = false;
};

}  // namespace rawalchemy
