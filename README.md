HyperTensor 🚀

Next-generation high-performance tensor library
Seamless Python integration. GPU-ready. NumPy & PyTorch-inspired.

Why HyperTensor?

HyperTensor is blazing fast, lightweight, and designed for modern machine learning and scientific computing. It gives you:

NumPy-like API for tensors, broadcasting, and element-wise operations.

PyTorch-inspired tensor operations, including squeeze and expand with indices.

Cross-device support: CPU ✅, CUDA ✅ (GPU acceleration ready).

Linear algebra tools: matrix multiplication, transpose, determinant.

Memory-efficient design with smart buffers.

Python bindings for a smooth, intuitive workflow.

In short: High performance meets Pythonic simplicity.

Installation
From source
git clone https://github.com/yourusername/HyperTensor.git
cd HyperTensor
mkdir build && cd build
cmake ..
make -j$(nproc)

Python bindings
cd build
python3 -m pip install .

Quickstart Example
import hypertensor as ht

# ===== Devices =====
cpu = ht.Device()
cuda = ht.Device()
cuda.type = ht.DeviceType.CUDA
cuda.index = 0

# ===== Tensor creation =====
a = ht.Tensor.empty([2, 3], ht.DType.Float32, cpu)
b = ht.Tensor.empty([2, 3], ht.DType.Float32, cpu)

# Fill tensors with data
for i in range(a.numel()):
    a.dataList()[i] = float(i+1)
    b.dataList()[i] = float((i+1)*2)

# ===== Element-wise operations =====
print("A + B:", a.add(b).dataList())
print("A * B:", a.mul(b).dataList())

# ===== Matrix operations =====
A2 = ht.Tensor.empty([2, 3])
B2 = ht.Tensor.empty([3, 2])
print("A2 @ B2:", A2.mm(B2).dataList())
print("Transpose:", A2.transpose().dataList())

# ===== Shape manipulation =====
reshaped = a.reshape([3,2])
squeezed = ht.Tensor.empty([1,3,1,2]).squeeze(0)
expanded = a.expand(0, 6)

print("Reshape 2x3 -> 3x2:", reshaped.shape())
print("Squeeze dim 0:", squeezed.shape())
print("Expand dim 0 to 6:", expanded.shape())

API Highlights
Function	Description
Tensor.empty(shape, dtype, device)	Create uninitialized tensor
add/sub/mul/div(other)	Element-wise operations with broadcasting
mm(other)	2D matrix multiplication
transpose()	Transpose 2D tensors
det()	Determinant of square 2D tensors
reshape(new_shape)	Reshape tensor
squeeze(dim)	Remove dimension at given index
expand(dim, size)	Add dimension of size size at index
dataList()	Get tensor data as Python list
Key Features

Blazing Performance – Optimized for CPU and CUDA.

Pythonic – Minimal friction when transitioning from NumPy/PyTorch.

Flexible Broadcasting – Works like NumPy, no manual reshaping.

Memory Efficient – Smart buffer management avoids unnecessary copies.

Future-ready – Extendable for deep learning, scientific computing, and GPU workloads.

License

MIT License © 2025 Érick Radmann
