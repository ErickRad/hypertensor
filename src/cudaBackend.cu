#ifdef HT_ENABLE_CUDA

#include "tensor.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <mutex>

using namespace hypertensor;

namespace {

std::once_flag cudaInitFlag;
cublasHandle_t cublasHandle = nullptr;
int activeDevice = 0;

void createCublas(int deviceIndex) {
    activeDevice = deviceIndex;
    cudaSetDevice(deviceIndex);
    cublasCreate(&cublasHandle);
    cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST);
}

} // anonymous

namespace hypertensor { namespace cudaBackend {

void initializeCuda(int deviceIndex) {
    std::call_once(cudaInitFlag, [deviceIndex](){ createCublas(deviceIndex); });
}

Tensor matmul(const Tensor& A, const Tensor& B) {
    if (A.device().type != DeviceType::CUDA || B.device().type != DeviceType::CUDA)
        throw std::runtime_error("cudaBackend::matmul expects CUDA tensors");

    if (A.dtype() != DType::Float32 || B.dtype() != DType::Float32)
        throw std::runtime_error("cudaBackend::matmul currently supports Float32");

    if (A.shape().size() != 2 || B.shape().size() != 2) 
        throw std::runtime_error("matmul expects 2D tensors");

    size_t m = A.shape()[0];
    size_t k = A.shape()[1];
    size_t kb = B.shape()[0];
    size_t n = B.shape()[1];

    if (k != kb) 
        throw std::runtime_error("matmul dimension mismatch");

    Device dev = A.device();
    Tensor C = Tensor::empty({m,n}, DType::Float32, dev);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const float* aPtr = static_cast<const float*>(A.data());
    const float* bPtr = static_cast<const float*>(B.data());

    float* cPtr = static_cast<float*>(C.data());

    initializeCuda(dev.index);
    cublasSetStream(cublasHandle, 0);

    cublasStatus_t st = cublasSgemm(cublasHandle,
                                   CUBLAS_OP_N, CUBLAS_OP_N,
                                   (int)m, (int)n, (int)k,
                                   &alpha,
                                   aPtr, (int)m,
                                   bPtr, (int)k,
                                   &beta,
                                   cPtr, (int)m);

    if (st != CUBLAS_STATUS_SUCCESS) 
        throw std::runtime_error("cublasSgemm failed");

    cudaError_t cerr = cudaGetLastError();

    if (cerr != cudaSuccess) 
        throw std::runtime_error("CUDA error after cublasSgemm");
        
    return C;
}

}} // namespace hypertensor::cudaBackend

#endif // HT_ENABLE_CUDA
