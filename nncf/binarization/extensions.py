"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os.path

import torch
from torch.utils.cpp_extension import load

from nncf.extensions import CudaNotAvailableStub
from nncf.definitions import NNCF_PACKAGE_ROOT_DIR


BASE_EXT_DIR = os.path.join(NNCF_PACKAGE_ROOT_DIR, "extensions/src/binarization")

EXT_INCLUDE_DIRS = [
    os.path.join(NNCF_PACKAGE_ROOT_DIR, "extensions/include"),
]

CPU_EXT_SRC_LIST = [
    os.path.join(BASE_EXT_DIR, "cpu/functions_cpu.cpp"),
    os.path.join(NNCF_PACKAGE_ROOT_DIR, "extensions/src/common/cpu/tensor_funcs.cpp")
]

CUDA_EXT_SRC_LIST = [
    os.path.join(BASE_EXT_DIR, "cuda/functions_cuda.cpp"),
    os.path.join(BASE_EXT_DIR, "cuda/functions_cuda_impl.cu")
]

BinarizedFunctionsCPU = load(
    'binarized_functions_cpu', CPU_EXT_SRC_LIST, extra_include_paths=EXT_INCLUDE_DIRS,
    verbose=False
)

if torch.cuda.is_available():
    BinarizedFunctionsCUDA = load(
        'binarized_functions_cuda', CUDA_EXT_SRC_LIST, extra_include_paths=EXT_INCLUDE_DIRS,
        verbose=False
    )
else:
    BinarizedFunctionsCUDA = CudaNotAvailableStub
