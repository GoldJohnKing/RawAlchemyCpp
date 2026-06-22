// SPDX-License-Identifier: AGPL-3.0-or-later
#include "../../include/aligned_allocator.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    using rawalchemy::AlignedVector;

    // Test 1: alignment guarantee
    {
        AlignedVector<float, 64> v(1024);
        assert(reinterpret_cast<std::uintptr_t>(v.data()) % 64 == 0);
    }

    // Test 2: small allocation also aligned
    {
        AlignedVector<float, 64> v(1);
        assert(reinterpret_cast<std::uintptr_t>(v.data()) % 64 == 0);
    }

    // Test 3: write/read correctness
    {
        AlignedVector<float, 64> v(100);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i);
        for (size_t i = 0; i < v.size(); ++i) assert(v[i] == static_cast<float>(i));
    }

    // Test 4: resize preserves alignment
    {
        AlignedVector<float, 64> v(10);
        v.resize(5000);
        assert(reinterpret_cast<std::uintptr_t>(v.data()) % 64 == 0);
    }

    std::cout << "test_aligned_allocator: PASS\n";
    return 0;
}
