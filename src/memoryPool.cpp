#include "memoryPool.h"
#include "allocator.h"
#include <mutex>
#include <cstdlib>
#include <cstring>

namespace hypertensor {

MemoryPool::MemoryPool(std::shared_ptr<hypertensor::Allocator> allocator)
    : allocator_(allocator) {}

void* MemoryPool::allocate(size_t bytes) {
    size_t key = bytes;

    if (key > maxPoolBlock_)
        return allocator_->allocate(bytes);

    {
        std::lock_guard<std::mutex> lk(globalMutex_);
        auto& vec = pool_[key];
        if (!vec.empty()) {
            void* p = vec.back();
            vec.pop_back();
            return p;
        }
    }

    return allocator_->allocate(key);
}

void MemoryPool::deallocate(void* ptr, size_t bytes) {
    size_t key = bytes;

    if (key > maxPoolBlock_) {
        allocator_->deallocate(ptr);
        return;
    }

    {
        std::lock_guard<std::mutex> lk(globalMutex_);
        pool_[key].push_back(ptr);
    }
}

void MemoryPool::flushAll() {
    std::lock_guard<std::mutex> lk(globalMutex_);

    for (auto& p : pool_) {
        for (void* ptr : p.second)
            allocator_->deallocate(ptr);
    }

    pool_.clear();
}

void MemoryPool::flushLocalToGlobal(size_t key, std::vector<void*>& localVec) {
    std::lock_guard<std::mutex> lk(globalMutex_);

    auto& global = pool_[key];

    global.insert(global.end(), localVec.begin(), localVec.end());
    localVec.clear();
}

}

static std::shared_ptr<hypertensor::MemoryPool> GLOBAL_POOL;

void hypertensor_init_global_pool(std::shared_ptr<hypertensor::Allocator> allocator) {
    GLOBAL_POOL = std::make_shared<hypertensor::MemoryPool>(allocator);
}
