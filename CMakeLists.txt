cmake_minimum_required(VERSION 3.18)
project(esa_interface LANGUAGES CXX CUDA)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)   # provides Torch::torch
if(DEFINED ENV{MY_INCLUDE_PATH})
    message(STATUS "Adding include path from MY_INCLUDE_PATH: $ENV{MY_INCLUDE_PATH}")
    # split colon‐separated paths into a CMake list
    string(REPLACE ":" ";" MY_INCLUDE_PATH_LIST "$ENV{MY_INCLUDE_PATH}")
    include_directories(${MY_INCLUDE_PATH_LIST})
endif()
set(CUDA_ROOT "/usr/local/cuda/" CACHE PATH "Path to CUDA root directory")
set(CMAKE_CUDA_COMPILER ${CUDA_ROOT}/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
enable_language(CUDA)
add_library(esa_kernels OBJECT esa_kernels.cu)
add_library(esa_copy_kernels OBJECT cuda_sm_copy.cu)
target_compile_options(esa_kernels PRIVATE
    --diag-suppress=128 --diag-suppress=2417 --diag-suppress=2597
    -Wall -fPIC
)
target_compile_options(esa_copy_kernels PRIVATE
    --diag-suppress=128 --diag-suppress=2417 --diag-suppress=2597
    -Wall -fPIC
)


# ---------- 2.  create the shared module ------------------
pybind11_add_module(esa_interface
    esa_interface.cc
)
target_include_directories(esa_interface PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${TORCH_INCLUDE_DIRS})
target_link_libraries(esa_interface PRIVATE esa_kernels esa_copy_kernels  ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})

# ---------- 3.  compile flags -----------------------------
target_compile_features(esa_interface PRIVATE cxx_std_17)
set_target_properties(esa_interface PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON   # needed if kernels call each other
    POSITION_INDEPENDENT_CODE ON    # redundant but explicit
)
