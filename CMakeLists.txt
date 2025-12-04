cmake_minimum_required(VERSION 3.18)
project(hypertensor LANGUAGES CXX CUDA)

# ================== C++ ==================
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)

# ================== Includes ==================
include_directories(include)

# ================== Pybind11 ==================
add_subdirectory(thirdParty/pybind11)

# ================== Sources ==================
file(GLOB CPU_SOURCES "src/*.cpp")
file(GLOB CUDA_SOURCES "src/*.cu")

# ================== Static library ==================
add_library(hypertensor_lib STATIC ${CPU_SOURCES} ${CUDA_SOURCES})
set_target_properties(hypertensor_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ================== Python ==================
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

pybind11_add_module(hypertensor python/bindings/pybind.cpp)
target_link_libraries(hypertensor PRIVATE hypertensor_lib pybind11::module)

# ================== CUDA ==================
if(CMAKE_CUDA_COMPILER)
    set_target_properties(hypertensor_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()
