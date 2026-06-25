// SPDX-License-Identifier: AGPL-3.0-or-later
// NaN/Inf output guard for NN demosaic inference.
// IMPORTANT: the .cpp implementation MUST be compiled WITHOUT -ffast-math,
// otherwise the optimizer will delete isnan()/isinf() checks.
#pragma once
#include <cstddef>

namespace rawalchemy {

/** Returns true if any element of `data` (length `count`) is NaN or +/- Inf.
 *  Safe to call with count == 0 (returns false). `data` may be nullptr if count == 0. */
bool nnOutputHasNaNInf(const float* data, size_t count);

} // namespace rawalchemy
