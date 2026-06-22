// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <memory>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace rawalchemy {

/// STL-compatible allocator that guarantees N-byte alignment for SIMD buffers.
/// Used by demosaic scratch buffers (RCD needs 64-byte alignment for AVX-512).
template <typename T, std::size_t N = 64>
struct AlignedAllocator {
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    static constexpr std::size_t alignment = N;

    AlignedAllocator() noexcept = default;
    AlignedAllocator(const AlignedAllocator&) noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, N>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n > (std::numeric_limits<std::size_t>::max)() / sizeof(T)) {
            throw std::bad_alloc();
        }
        std::size_t bytes = n * sizeof(T);
#if defined(_WIN32)
        void* ptr = _aligned_malloc(bytes, N);
        if (!ptr) throw std::bad_alloc();
#else
        // C17 aligned_alloc requires size to be a multiple of alignment; round up.
        std::size_t aligned_bytes = (bytes + N - 1) & ~(N - 1);
        void* ptr = std::aligned_alloc(N, aligned_bytes);
        if (!ptr) throw std::bad_alloc();
#endif
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t) noexcept {
#if defined(_WIN32)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, N>;
    };
};

template <typename T, typename U, std::size_t N>
bool operator==(const AlignedAllocator<T, N>&, const AlignedAllocator<U, N>&) noexcept {
    return true;
}

template <typename T, typename U, std::size_t N>
bool operator!=(const AlignedAllocator<T, N>& a, const AlignedAllocator<U, N>& b) noexcept {
    return !(a == b);
}

/// Convenience alias: std::vector with 64-byte aligned storage
template <typename T, std::size_t N = 64>
using AlignedVector = std::vector<T, AlignedAllocator<T, N>>;

} // namespace rawalchemy
