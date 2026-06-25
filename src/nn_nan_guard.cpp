// SPDX-License-Identifier: AGPL-3.0-or-later
// DO NOT compile this file with -ffast-math (CMakeLists enforces this).
#include "nn_nan_guard.h"
#include <cmath>

namespace rawalchemy {

bool nnOutputHasNaNInf(const float* data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i])) {
            return true;
        }
    }
    return false;
}

} // namespace rawalchemy
