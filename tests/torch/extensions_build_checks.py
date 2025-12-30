# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

import nncf

if __name__ == "__main__":
    if len(sys.argv) != 2:
        msg = "Must be run with target extensions build mode"
        raise nncf.ValidationError(msg)
    mode = sys.argv[1]

    if mode == "cpu":
        # Do not remove - the import here is for testing purposes.

        from nncf.torch.quantization.extensions import QuantizedFunctionsCPULoader

        QuantizedFunctionsCPULoader.load()
    elif mode == "cuda":
        from nncf.torch.quantization.extensions import QuantizedFunctionsCUDALoader

        # Set CUDA Architecture
        # See cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake
        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5+PTX"
        QuantizedFunctionsCUDALoader.load()
    else:
        msg = "Invalid mode type!"
        raise nncf.ValidationError(msg)
