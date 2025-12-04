cmake_minimum_required(VERSION 3.12)

pybind11_add_module(hypertensor MODULE
    bindings/pybind.cpp
)

target_link_libraries(hypertensor PRIVATE hypertensor_core)

target_include_directories(hypertensor PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/thirdParty
)
