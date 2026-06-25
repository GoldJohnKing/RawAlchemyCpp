// SPDX-License-Identifier: AGPL-3.0-or-later
// Central demosaic dispatcher implementation.
//
// Plan A status:
//   - Neural path: live (delegates to nnDemosaic, returns status verbatim).
//   - Classical path: STUB returning InvalidParam. The real RCD (Bayer) and
//     Markesteijn (X-Trans) ports are wired in Plan B; their headers remain
//     included so the classical compilation units stay in the build for the
//     preview path (demosaic_rcd.cpp / demosaic_markesteijn.cpp).
//
// Design §6.2: the neural path NEVER silently falls back to classical on
// NaNOutput or InferenceFailed — the caller receives the raw status and
// decides how to react (surface error, retry, re-run classically, etc.).

#include "demosaic_dispatch.h"

#include "demosaic_markesteijn.h"
#include "demosaic_rcd.h"

namespace rawalchemy {

NnDemosaicStatus demosaicDispatch(const NnDemosaicInput& in,
                                  NnDemosaicOutput& out,
                                  DemosaicPath path) {
    switch (path) {
        case DemosaicPath::Neural:
            // Verbatim status — no auto-fallback (design §6.2).
            return nnDemosaic(in, out);
        case DemosaicPath::Classical:
            // Plan A stub. Plan B will route Bayer (filters != 9) ->
            // rcd_demosaic and X-Trans (filters == 9) ->
            // markesteijn_demosaic, packaging the result into `out`.
            // `in`/`out` are deliberately unused until then; both are still
            // referenced by the Neural branch above so no -Wunused-parameter.
            return NnDemosaicStatus::InvalidParam;
    }
    // Unreachable for a 2-value enum, but keeps -Wreturn-type quiet under
    // every supported compiler (no exhaustiveness warning reliance).
    return NnDemosaicStatus::InvalidParam;
}

} // namespace rawalchemy
