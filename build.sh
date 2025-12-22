#!/usr/bin/bash
# nvcc -o retrieval_kernel -I/usr/local/lib/python3.12/dist-packages/torch/include -I/usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include -I/usr/include/python3.12 ./retrieval_kernel.cu

# export MY_INCLUDE_PATH="/usr/local/lib/python3.12/dist-packages/torch/include:/usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include:/usr/include/python3.12:/usr/local/cuda/include"
# [[ -d ./build/ ]] && rm -rf ./build/
# cmake -Dpybind11_DIR="/usr/local/lib/python3.12/dist-packages/pybind11/share/cmake/pybind11" -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/root/hk/ucm/myenv/bin/python -DCMAKE_PREFIX_PATH="/usr/local/lib/python3.12/dist-packages/pybind11/share/cmake/pybind11" -DCMAKE_PREFIX_PATH=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path)") -DRUNTIME_ENVIRONMENT=cuda -S . -B ./build
# cd build
# make
# cd ..
