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

import pathlib
import os
import os.path
import torch
from torch.utils.cpp_extension import load

from nncf.extensions import CudaNotAvailableStub, ExtensionsType, ExtensionLoader, EXTENSIONS
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


@EXTENSIONS.register()
class BinarizedFunctionsCPULoader(ExtensionLoader):
    @staticmethod
    def extension_type():
        return ExtensionsType.CPU

    @staticmethod
    def load():
        return load('binarized_functions_cpu', CPU_EXT_SRC_LIST, extra_include_paths=EXT_INCLUDE_DIRS,
                    verbose=False)


@EXTENSIONS.register()
class BinarizedFunctionsCUDALoader(ExtensionLoader):
    @staticmethod
    def extension_type():
        return ExtensionsType.CUDA

    @staticmethod
    def load():
        name = 'binarized_functions_cuda'
        # pylint:disable=protected-access
        cuda_arch_build_dir = torch.utils.cpp_extension._get_build_directory(name, verbose=False)
        cuda_arch_list_file = os.path.join(cuda_arch_build_dir, 'cuda_arch_list.txt')
        p = pathlib.Path(cuda_arch_list_file)
        if not p.exists():
            arch_list = os.getenv('TORCH_CUDA_ARCH_LIST')
            if arch_list:
                with open(p, 'w') as f:
                    print('The "TORCH_CUDA_ARCH_LIST" environment variable has been saving in a file. '
                          'This environment variable will be set every time before loading cpp extensions. '
                          f'The filepath is {cuda_arch_list_file}'
                          'If you want to build extensions locally and according to your CUDA version, '
                          'please remove this file and set enviroment variable "TORCH_CUDA_ARCH_LIST"'
                          ' to an empty string')
                    f.write(arch_list)
        else:
            print('The file containing "TORCH_CUDA_ARCH_LIST" environment variable was detected. '
                  'The "TORCH_CUDA_ARCH_LIST" environment variable will be set according to the file. '
                  'A process of loading/building CUDA extensions will be with according to "TORCH_CUDA_ARCH_LIST"'
                  f'The filepath is {cuda_arch_list_file}')
            with open(p, 'r') as f:
                arch_list = f.readline()
            os.environ['TORCH_CUDA_ARCH_LIST'] = arch_list

        return load(name, CUDA_EXT_SRC_LIST, extra_include_paths=EXT_INCLUDE_DIRS,
                    verbose=True)


BinarizedFunctionsCPU = BinarizedFunctionsCPULoader.load()

if torch.cuda.is_available():
    BinarizedFunctionsCUDA = BinarizedFunctionsCUDALoader.load()
else:
    BinarizedFunctionsCUDA = CudaNotAvailableStub
