// SPDX-License-Identifier: AGPL-3.0-or-later
// Central demosaic dispatcher. Routes a demosaic request to either the
// classical (RCD / Markesteijn) or neural (x-veon) path based on a
// caller-chosen DemosaicPath value.
//
// The neural path returns its NnDemosaicStatus verbatim — NO silent fallback
// to the classical path on NaNOutput / InferenceFailed (design §6.2: the
// caller decides how to react, e.g. surface an error or retry classically).
//
// The classical path is a Plan A stub: it returns InvalidParam. Plan B wires
// the real RCD (Bayer) and Markesteijn (X-Trans) ports, which are already
// compiled into the library for the preview path (demosaic_rcd.cpp /
// demosaic_markesteijn.cpp).
#pragma once

#include "demosaic_nn_xveon.h"

namespace rawalchemy {

/** Selects which demosaic algorithm demosaicDispatch() routes to. */
enum class DemosaicPath {
    Classical,  // RCD (Bayer) / Markesteijn (X-Trans) — Plan A stub
    Neural      // x-veon NN demosaic via ONNX Runtime
};

/** Route a demosaic request to the selected algorithm.
 *
 *  @param in    CFA + metadata bundle (see NnDemosaicInput).
 *  @param out   filled with linear sRGB on Ok; contents undefined otherwise.
 *  @param path  Classical or Neural.
 *  @return NnDemosaicStatus::Ok on success; otherwise the underlying path's
 *          status. For DemosaicPath::Neural the status is returned verbatim
 *          (no auto-fallback on NaNOutput / InferenceFailed — design §6.2).
 *          For DemosaicPath::Classical the Plan A stub returns InvalidParam. */
NnDemosaicStatus demosaicDispatch(const NnDemosaicInput& in,
                                  NnDemosaicOutput& out,
                                  DemosaicPath path);

} // namespace rawalchemy
