# HyperTensor 🚀

> **Next-generation high-performance tensor library**  
> Seamless Python integration. GPU-ready. Faster than Numpy.

---

## Why HyperTensor?

HyperTensor is **blazing fast**, **lightweight**, and designed for modern machine learning and scientific computing. It gives you:

- **NumPy-like API** for tensors, broadcasting, and element-wise operations.
- **PyTorch-inspired tensor operations**, including `squeeze` and `expand` with indices.
- **Cross-device support**: CPU ✅, CUDA ✅ (GPU acceleration ready).
- **Linear algebra tools**: matrix multiplication, transpose, determinant.
- **Memory-efficient design** with smart buffers.
- **Python bindings** for a smooth, intuitive workflow.

In short: **High performance meets Pythonic simplicity**.

---

## Installation

### From source

```bash
git clone https://github.com/yourusername/HyperTensor.git
cd HyperTensor
mkdir build && cd build
cmake ..
make -j$(nproc)

USAGE: 

import hypertensor as ht

# ===== Devices =====
cpu = ht.Device()               # CPU device
cuda = ht.Device()              # CUDA device (GPU)
cuda.type = ht.DeviceType.CUDA
cuda.index = 0                  # GPU index

# ===== Tensor creation =====
# Tensor.empty(shape, dtype, device) -> create empty tensor with uninitialized values
a = ht.Tensor.empty([2, 3], ht.DType.Float32, cpu)
b = ht.Tensor.empty([2, 3], ht.DType.Float32, cpu)

# Fill tensor values manually
for i in range(a.size()):
    a.data()[i] = float(i + 10)  # dataList returns Python list
for i in range(b.size()):
    b.data()[i] = float(i + 2)

print("Tensor A:", a.dataList())
print("Tensor B:", b.dataList())

# ===== Element-wise operations =====
# Each operation returns a new tensor
c_add = a.add(b)    # add tensor a + b
c_sub = a.sub(b)    # subtract tensor a - b
c_mul = a.mul(b)    # multiply element-wise a * b
c_div = a.div(b)    # divide element-wise a / b

print("A + B:", c_add.dataList())
print("A - B:", c_sub.dataList())
print("A * B:", c_mul.dataList())
print("A / B:", c_div.dataList())

# ===== Matrix multiplication =====
# mm(tensor) -> matrix multiplication, expects 2D tensors

a2 = ht.Tensor.empty([2, 3], ht.DType.Float32, cpu)
b2 = ht.Tensor.empty([3, 2], ht.DType.Float32, cpu)

# fill with sample data
for i in range(a2.size()):
    a2.data()[i] = float(i + 1)

for i in range(b2.size()):
    b2.data()[i] = float(i + 1) * 2

print("\nTensor A2: ", a2.dataList())
print("Tensor B2: ", b2.dataList())

mm_result = a2.mm(b2)
print("A2 @ B2:", mm_result.dataList())

# ===== Transpose =====
# transpose() -> returns the transposed tensor

transposed = a2.transpose()
print("A2^T:", transposed.dataList())

# ===== Determinant =====
# det() -> computes determinant (only square 2D tensors)
square = ht.Tensor.empty([3,3], ht.DType.Float32, cpu)

for i in range(square.size()):
    square.dataList()[i] = float(i + 1)
det_val = square.det()
print("Determinant of 3x3:", det_val)

# ===== Reshape / Squeeze / Expand =====
# reshape(newShape) -> reshapes tensor, number of elements must match
reshaped = a.reshape([3,2])
print("Reshape A 2x3 -> 3x2:", reshaped.shape())

# squeeze(index) -> remove dimensions at index
tensor_squeeze = ht.Tensor.empty([1,3,1,2], ht.DType.Float32, cpu)
squeezed = tensor_squeeze.squeeze(0)
print("Squeeze dim at index 0 from [1,3,1,2] ->", squeezed.shape())

# expand(index) -> reshape tensor, preserving data
expanded_dim1 = a.expand(1, 4)  # add a new dimension of size 4 at index 1
print("Expand dim at index 1 with 4 of size from [2, 3] ->", expanded_dim1.shape())

# ===== Device transfer =====
# to(device) -> returns new tensor on specified device
a_cuda = a.to(cuda)
print("A on CUDA device:", a_cuda.device().type == ht.DeviceType.CUDA)

# ===== Memory access =====
# data() -> memoryview of tensor data
# dataList() -> Python list of tensor data
print("A memoryview:", a.data())
print("A as Python list:", a.dataList())

``
