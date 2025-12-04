# HyperTensor 🚀

> **Next-generation high-performance tensor library**  
> Seamless Python integration. GPU-ready. NumPy & PyTorch-inspired.

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
