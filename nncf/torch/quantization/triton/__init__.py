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

from typing import Callable

import torch

from nncf.torch.utils import CudaNotAvailableStub


class TritonFunctionsWrapper:
    def __init__(self):
        """
        Wrapper that handles Triton kernel imports since it would trigger compilation.
        To prevent issues with non-CUDA environment.
        """
        from nncf.torch.quantization.triton.reference import backward
        from nncf.torch.quantization.triton.reference import forward

        self.Quantize_forward = forward
        self.Quantize_backward = backward

    def get(self, fn_name: str) -> Callable:
        return getattr(self, fn_name)


if torch.cuda.is_available():
    QuantizedFunctionsCUDA = TritonFunctionsWrapper()
else:
    QuantizedFunctionsCUDA = CudaNotAvailableStub()
