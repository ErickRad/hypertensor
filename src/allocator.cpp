#include "allocator.h"
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
    #include <mm_malloc.h>
    #define HT_WINDOWS 1
#else
    #define HT_WINDOWS 0
#endif

namespace hypertensor {

void* Allocator::allocate(size_t bytes, bool zeroInit) {
    void* p = nullptr;

#if HT_WINDOWS
    p = _mm_malloc(bytes, 64);
    if (!p) return nullptr;
#else
    if (posix_memalign(&p, 64, bytes) != 0) return nullptr;
#endif

    if (zeroInit)
        std::memset(p, 0, bytes);

    return p;
}

void Allocator::deallocate(void* ptr) {
    if (!ptr) return;

#if HT_WINDOWS
    _mm_free(ptr);
#else
    free(ptr);
#endif
}

}
