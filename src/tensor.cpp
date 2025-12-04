#include "tensor.h"
#include "allocator.h"
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#ifdef HT_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace hypertensor {

// ===== Buffer =====
Buffer::Buffer(void* ptr, size_t bytes, std::shared_ptr<Allocator> allocator, const Device& device)
: ptr_(ptr), bytes_(bytes), allocator_(allocator), device_(device) {}

Buffer::~Buffer() {
    if(ptr_) allocator_->deallocate(ptr_);
}

void* Buffer::data() const { return ptr_; }
size_t Buffer::sizeBytes() const { return bytes_; }
const Device& Buffer::device() const { return device_; }

// ===== Tensor =====
Tensor::Tensor(): buffer_(nullptr), shape_(), dtype_(DType::Float32), device_() {}

static size_t dtypeSize(DType dt) {
    switch(dt) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::Int32: return 4;
        case DType::UInt8: return 1;
    }
    return 4;
}

Tensor Tensor::empty(const std::vector<size_t>& shape, DType dtype, const Device& device) {
    size_t n = 1;
    for(size_t s : shape) n *= s;

    size_t bytes = n * dtypeSize(dtype);
    auto alloc = std::make_shared<Allocator>();
    void* ptr = alloc->allocate(bytes, true);

    auto buf = std::make_shared<Buffer>(ptr, bytes, alloc, device);
    Tensor t;
    t.buffer_ = buf;
    t.shape_ = shape;
    t.dtype_ = dtype;
    t.device_ = device;
    return t;
}

size_t Tensor::size() const {
    size_t n = 1;
    for(size_t s: shape_) n *= s;
    return n;
}

const std::vector<size_t>& Tensor::shape() const { return shape_; }
DType Tensor::dtype() const { return dtype_; }
Device Tensor::device() const { return device_; }

void* Tensor::data() { 
    if(!buffer_) throw std::runtime_error("Tensor has no buffer"); 
    return buffer_->data(); 
}

const void* Tensor::data() const { 
    if(!buffer_) throw std::runtime_error("Tensor has no buffer"); 
    return buffer_->data(); 
}

std::vector<float> Tensor::dataList() const {
    const float* ptr = static_cast<const float*>(data());
    return std::vector<float>(ptr, ptr + size());
}

// ===== Device transfer =====
Tensor Tensor::to(const Device& device, bool nonBlocking) const {
    if(device == device_) return *this;

    Tensor dst = Tensor::empty(shape_, dtype_, device);

    if(device_.type == DeviceType::CPU && device.type == DeviceType::CPU){
        std::memcpy(dst.data(), data(), buffer_->sizeBytes());
        return dst;
    }

#ifdef HT_ENABLE_CUDA
    cudaMemcpyKind kind = cudaMemcpyHostToHost;
    if(device_.type == DeviceType::CPU && device.type == DeviceType::CUDA)
        kind = cudaMemcpyHostToDevice;
    else if(device_.type == DeviceType::CUDA && device.type == DeviceType::CPU)
        kind = cudaMemcpyDeviceToHost;
    else if(device_.type == DeviceType::CUDA && device.type == DeviceType::CUDA)
        kind = cudaMemcpyDeviceToDevice;
    else
        throw std::runtime_error("Unsupported device transfer");

    cudaMemcpy(dst.data(), data(), buffer_->sizeBytes(), kind);
    return dst;
#else
    throw std::runtime_error("Device transfer cannot be performed: unavailable device type");
#endif
}

Tensor Tensor::contiguous() const { return *this; }

// ===== Broadcasting helpers =====
void Tensor::checkBroadcastShape(const Tensor& other, std::vector<size_t>& outShape) const {
    const auto& aShape = shape_;
    const auto& bShape = other.shape();
    size_t ndim = std::max(aShape.size(), bShape.size());
    outShape.resize(ndim);

    for(size_t i = 0; i < ndim; ++i){
        size_t aDim = i < ndim - aShape.size() ? 1 : aShape[i - (ndim - aShape.size())];
        size_t bDim = i < ndim - bShape.size() ? 1 : other.shape()[i - (ndim - bShape.size())];
        if(aDim != bDim && aDim != 1 && bDim != 1)
            throw std::runtime_error("Shape not broadcastable");
        outShape[i] = std::max(aDim, bDim);
    }
}

size_t Tensor::computeIndex(const std::vector<size_t>& idx, const std::vector<size_t>& shape) const {
    size_t flatIndex = 0, multiplier = 1;
    for(int i=(int)shape.size()-1;i>=0;--i){
        flatIndex += idx[i]*multiplier;
        multiplier *= shape[i];
    }
    return flatIndex;
}

// ===== Element-wise operations =====
Tensor Tensor::add(const Tensor& other) const {
    std::vector<size_t> outShape;
    checkBroadcastShape(other, outShape);
    Tensor out = Tensor::empty(outShape, dtype_, device_);

    std::vector<size_t> idx(outShape.size(),0);
    size_t total = out.size();

    const float* aPtr = static_cast<const float*>(data());
    const float* bPtr = static_cast<const float*>(other.data());
    float* cPtr = static_cast<float*>(out.data());

    for(size_t i=0;i<total;++i){
        size_t aIndex=0, bIndex=0;
        for(size_t d=0;d<outShape.size();++d){
            size_t aDim = d < outShape.size()-shape_.size() ? 1 : shape_[d-(outShape.size()-shape_.size())];
            size_t bDim = d < outShape.size()-other.shape().size() ? 1 : other.shape()[d-(outShape.size()-other.shape().size())];
            size_t idxDim = idx[d];
            aIndex = aIndex * aDim + (aDim==1?0:idxDim);
            bIndex = bIndex * bDim + (bDim==1?0:idxDim);
        }
        cPtr[i] = aPtr[aIndex] + bPtr[bIndex];

        for(int d=(int)idx.size()-1;d>=0;--d){
            idx[d]++;
            if(idx[d]<outShape[d]) break;
            idx[d]=0;
        }
    }
    return out;
}

Tensor Tensor::sub(const Tensor& other) const {
    Tensor out = add(other);
    float* cPtr = static_cast<float*>(out.data());
    const float* aPtr = static_cast<const float*>(data());
    const float* bPtr = static_cast<const float*>(other.data());
    for(size_t i=0;i<out.size();++i) cPtr[i] = aPtr[i]-bPtr[i];
    return out;
}

Tensor Tensor::mul(const Tensor& other) const {
    Tensor out = add(other);
    float* cPtr = static_cast<float*>(out.data());
    const float* aPtr = static_cast<const float*>(data());
    const float* bPtr = static_cast<const float*>(other.data());
    for(size_t i=0;i<out.size();++i) cPtr[i] = aPtr[i]*bPtr[i];
    return out;
}

Tensor Tensor::div(const Tensor& other) const {
    Tensor out = add(other);
    float* cPtr = static_cast<float*>(out.data());
    const float* aPtr = static_cast<const float*>(data());
    const float* bPtr = static_cast<const float*>(other.data());
    for(size_t i=0;i<out.size();++i) cPtr[i] = aPtr[i]/bPtr[i];
    return out;
}

// ===== Matrix multiplication =====
Tensor Tensor::mm(const Tensor& other) const {
    if(shape_.size()!=2 || other.shape().size()!=2)
        throw std::runtime_error("mm expects 2D tensors");
    size_t m=shape_[0], k=shape_[1], n=other.shape()[1];
    Tensor out = Tensor::empty({m,n}, dtype_, device_);

    const float* aPtr = static_cast<const float*>(data());
    const float* bPtr = static_cast<const float*>(other.data());
    float* cPtr = static_cast<float*>(out.data());
    std::fill_n(cPtr, m*n, 0.0f);

#if defined(_OPENMP)
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for(size_t i=0;i<m;++i)
        for(size_t kk=0;kk<k;++kk)
            for(size_t j=0;j<n;++j)
                cPtr[i*n+j] += aPtr[i*k+kk]*bPtr[kk*n+j];

    return out;
}

// ===== Linear algebra =====
Tensor Tensor::transpose() const {
    if(shape_.size()!=2) throw std::runtime_error("transpose: only 2D supported");
    size_t m=shape_[0], n=shape_[1];
    Tensor out = Tensor::empty({n,m}, dtype_, device_);
    const float* a = static_cast<const float*>(data());
    float* b = static_cast<float*>(out.data());
    for(size_t i=0;i<m;i++)
        for(size_t j=0;j<n;j++)
            b[j*m+i] = a[i*n+j];
    return out;
}

float Tensor::det() const {
    if(shape_.size()!=2 || shape_[0]!=shape_[1]) throw std::runtime_error("det requires square matrix");
    size_t n=shape_[0];
    std::vector<float> mat(n*n);
    const float* src = static_cast<const float*>(data());
    std::copy(src, src+n*n, mat.begin());

    float detVal=1.0f;
    for(size_t i=0;i<n;i++){
        size_t pivotRow=i;
        for(size_t j=i+1;j<n;j++)
            if(std::abs(mat[j*n+i])>std::abs(mat[pivotRow*n+i])) pivotRow=j;
        if(std::abs(mat[pivotRow*n+i])<1e-8f) return 0.0f;
        if(pivotRow!=i) { for(size_t k=0;k<n;k++) std::swap(mat[i*n+k], mat[pivotRow*n+k]); detVal*=-1.0f;}
        detVal*=mat[i*n+i];
        for(size_t j=i+1;j<n;j++){
            float factor = mat[j*n+i]/mat[i*n+i];
            for(size_t k=i;k<n;k++) mat[j*n+k]-=factor*mat[i*n+k];
        }
    }
    return detVal;
}

// ===== Reshape / squeeze / expand =====
Tensor Tensor::reshape(const std::vector<size_t>& newShape) const {
    size_t n=1; for(size_t s:newShape) n*=s;
    if(n!=size()) throw std::runtime_error("reshape: element count mismatch");
    Tensor out=*this; out.shape_=newShape; return out;
}

Tensor Tensor::squeeze(int dim) const {
    std::vector<size_t> newShape;

    if(dim < 0) {
        for(size_t s : shape_) {
            if(s > 1) newShape.push_back(s);
        }
    } else {
        if(dim >= (int)shape_.size())
            throw std::runtime_error("squeeze: dimension out of range");

        for(size_t i = 0; i < shape_.size(); ++i) {
            if((int)i == dim) {
                if(shape_[i] != 1)
                    throw std::runtime_error("squeeze: cannot squeeze dimension with size != 1");

            } else {
                newShape.push_back(shape_[i]);
            }
        }
    }

    Tensor out = *this;
    out.shape_ = newShape;
    return out;
}

Tensor Tensor::expand(size_t dim, size_t size) const {
    if(dim > shape_.size()) throw std::runtime_error("expand: dim out of range");
    std::vector<size_t> newShape = shape_;
    newShape.insert(newShape.begin() + dim, size);
    Tensor out = *this;
    out.shape_ = newShape;
    return out;
}

} // namespace hypertensor
