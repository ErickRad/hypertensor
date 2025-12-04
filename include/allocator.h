#pragma once
#include <cstddef>

namespace hypertensor {

class Allocator {
public:
    virtual void* allocate(size_t bytes, bool zeroInit = false);
    virtual void deallocate(void* ptr);
};

}
