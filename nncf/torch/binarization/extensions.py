# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path
import subprocess

import torch

import nncf
from nncf import nncf_logger
from nncf.definitions import NNCF_PACKAGE_ROOT_DIR
from nncf.torch.binarization.reference import ReferenceBinarizedFunctions
from nncf.torch.extensions import EXTENSIONS
from nncf.torch.extensions import CudaNotAvailableStub
from nncf.torch.extensions import ExtensionLoader
from nncf.torch.extensions import ExtensionLoaderTimeoutException
from nncf.torch.extensions import ExtensionNamespace
from nncf.torch.extensions import ExtensionsType

BASE_EXT_DIR = os.path.join(NNCF_PACKAGE_ROOT_DIR, "torch/extensions/src/binarization")

EXT_INCLUDE_DIRS = [
    os.path.join(NNCF_PACKAGE_ROOT_DIR, "torch/extensions/include"),
]

CPU_EXT_SRC_LIST = [
    os.path.join(BASE_EXT_DIR, "cpu/functions_cpu.cpp"),
    os.path.join(NNCF_PACKAGE_ROOT_DIR, "torch/extensions/src/common/cpu/tensor_funcs.cpp"),
]

CUDA_EXT_SRC_LIST = [
    os.path.join(BASE_EXT_DIR, "cuda/functions_cuda.cpp"),
    os.path.join(BASE_EXT_DIR, "cuda/functions_cuda_impl.cu"),
]


@EXTENSIONS.register()
class BinarizedFunctionsCPULoader(ExtensionLoader):
    @classmethod
    def name(cls) -> str:
        return "binarized_functions_cpu"

    @classmethod
    def extension_type(cls):
        return ExtensionsType.CPU

    @classmethod
    def load(cls):
        try:
            retval = torch.utils.cpp_extension.load(
                cls.name(),
                CPU_EXT_SRC_LIST,
                extra_include_paths=EXT_INCLUDE_DIRS,
                build_directory=cls.get_build_dir(),
                verbose=False,
            )
        except ExtensionLoaderTimeoutException as e:
            raise e
        except Exception as e:
            nncf_logger.warning(
                f"Could not compile CPU binarization extensions. "
                f"Falling back on torch native operations - "
                f"CPU binarization fine-tuning may be slower than expected.\n"
                f"Reason: {str(e)}"
            )
            retval = ReferenceBinarizedFunctions
        return retval


@EXTENSIONS.register()
class BinarizedFunctionsCUDALoader(ExtensionLoader):
    @classmethod
    def name(cls) -> str:
        return "binarized_functions_cuda"

    @classmethod
    def extension_type(cls):
        return ExtensionsType.CUDA

    @classmethod
    def load(cls):
        try:
            return torch.utils.cpp_extension.load(
                cls.name(),
                CUDA_EXT_SRC_LIST,
                extra_include_paths=EXT_INCLUDE_DIRS,
                build_directory=cls.get_build_dir(),
                verbose=False,
            )
        except ExtensionLoaderTimeoutException as e:
            raise e
        except (subprocess.CalledProcessError, OSError, RuntimeError) as e:
            assert torch.cuda.is_available()
            raise nncf.InstallationError(
                "CUDA is available for PyTorch, but NNCF could not compile "
                "GPU binarization extensions. Make sure that you have installed CUDA development "
                "tools (see https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html for "
                "guidance) and that 'nvcc' is available on your system's PATH variable.\n"
            ) from e


BinarizedFunctionsCPU = ExtensionNamespace(BinarizedFunctionsCPULoader())

if torch.cuda.is_available():
    BinarizedFunctionsCUDA = ExtensionNamespace(BinarizedFunctionsCUDALoader())
else:
    BinarizedFunctionsCUDA = CudaNotAvailableStub()