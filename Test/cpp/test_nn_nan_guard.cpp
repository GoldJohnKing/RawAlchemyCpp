// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../../include/nn_nan_guard.h"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    using namespace rawalchemy;

    // Clean buffer -> false
    {
        float data[] = {0.0f, 1.0f, -1.0f, 0.5f, 1000.0f};
        assert(nnOutputHasNaNInf(data, 5) == false);
    }
    // Contains NaN -> true
    {
        float data[] = {0.0f, NAN, 1.0f};
        assert(nnOutputHasNaNInf(data, 3) == true);
    }
    // Contains +Inf -> true
    {
        float data[] = {1.0f, INFINITY, 2.0f};
        assert(nnOutputHasNaNInf(data, 3) == true);
    }
    // Contains -Inf -> true
    {
        float data[] = {-INFINITY, 0.0f};
        assert(nnOutputHasNaNInf(data, 2) == true);
    }
    // Zero count -> false
    {
        assert(nnOutputHasNaNInf(nullptr, 0) == false);
    }
    std::cout << "test_nn_nan_guard: OK\n";
    return 0;
}
