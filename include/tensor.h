#ifndef HYPERTENSOR_TENSOR_H
#define HYPERTENSOR_TENSOR_H

#include <vector>
#include <memory>

namespace hypertensor {

enum class DeviceType { CPU, CUDA };
enum class DType { Float32, Float16, Int32, UInt8 };

struct Device {
    DeviceType type = DeviceType::CPU;
    int index = 0;

    bool operator==(const Device& other) const {
        return type == other.type && index == other.index;
    }
};

class Buffer;
class Allocator;

class Tensor {
public:
    Tensor();
    static Tensor empty(const std::vector<size_t>& shape, DType dtype, const Device& device = Device());

    size_t size() const;
    const std::vector<size_t>& shape() const;
    DType dtype() const;
    Device device() const;

    void* data();
    const void* data() const;

    std::vector<float> dataList() const;

    Tensor to(const Device& device, bool nonBlocking=false) const;
    Tensor contiguous() const;

    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor div(const Tensor& other) const;

    Tensor mm(const Tensor& other) const;
    Tensor transpose() const;
    float det() const;

    Tensor reshape(const std::vector<size_t>& newShape) const;
    Tensor squeeze(int dim = -1) const;
    Tensor expand(size_t dim, size_t size) const;

private:
    std::shared_ptr<Buffer> buffer_;
    std::vector<size_t> shape_;
    DType dtype_;
    Device device_;

    void checkBroadcastShape(const Tensor& other, std::vector<size_t>& outShape) const;
    size_t computeIndex(const std::vector<size_t>& idx, const std::vector<size_t>& shape) const;
};

class Buffer {
public:
    Buffer(void* ptr, size_t bytes, std::shared_ptr<Allocator> allocator, const Device& device);
    ~Buffer();

    void* data() const;
    size_t sizeBytes() const;
    const Device& device() const;

private:
    void* ptr_;
    size_t bytes_;
    std::shared_ptr<Allocator> allocator_;
    Device device_;
};

} // namespace hypertensor

#endif
