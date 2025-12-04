#pragma once
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include "allocator.h"

namespace hypertensor {

class MemoryPool {
public:
    explicit MemoryPool(std::shared_ptr<hypertensor::Allocator> allocator);

    void* allocate(size_t bytes);
    void  deallocate(void* ptr, size_t bytes);
    void  flushAll();

private:
    void flushLocalToGlobal(size_t key, std::vector<void*>& localVec);

    std::shared_ptr<hypertensor::Allocator> allocator_;
    std::unordered_map<size_t, std::vector<void*>> pool_;
    std::mutex globalMutex_;
    const size_t maxPoolBlock_ = 1 << 20;
};

} // namespace hypertensor_internal

void hypertensor_init_global_pool(std::shared_ptr<hypertensor::Allocator> allocator);
