"""
 Copyright (c) 2022 Intel Corporation
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
from typing import List

import torch
from torch import nn

from nncf.torch.layer_utils import COMPRESSION_MODULES
from nncf.torch.sparsity.functions import apply_binary_mask as apply_binary_mask_impl
from nncf.torch.utils import is_tracing_state, no_jit_trace


@COMPRESSION_MODULES.register()
class BinaryMask(nn.Module):
    def __init__(self, shape: List[int]):
        super().__init__()
        self.register_buffer("_binary_mask", torch.ones(shape))
        self.frozen = False

    @property
    def binary_mask(self):
        return self._binary_mask

    @binary_mask.setter
    def binary_mask(self, tensor):
        with torch.no_grad():
            self._binary_mask.set_(tensor)

    def forward(self, weight):
        if is_tracing_state():
            with no_jit_trace():
                return weight.mul_(self.binary_mask)
        tmp_tensor = self._calc_training_binary_mask(weight)
        return apply_binary_mask_impl(tmp_tensor, weight)

    def _calc_training_binary_mask(self, weight):
        return self.binary_mask

    def apply_binary_mask(self, weight):
        return apply_binary_mask_impl(self.binary_mask, weight)
