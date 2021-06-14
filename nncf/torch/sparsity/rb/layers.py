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
from typing import List

import torch

from nncf.torch.sparsity.layers import BinaryMask
from nncf.torch.sparsity.rb.functions import calc_rb_binary_mask, binary_mask
from nncf.torch.functions import logit
from nncf.torch.layer_utils import COMPRESSION_MODULES, CompressionParameter



@COMPRESSION_MODULES.register()
class RBSparsifyingWeight(BinaryMask):
    def __init__(self, weight_shape: List[int], frozen=True, compression_lr_multiplier=None, eps=1e-6):
        super().__init__(weight_shape)
        self.frozen = frozen
        self.eps = eps
        self._mask = CompressionParameter(logit(torch.ones(weight_shape) * 0.99), requires_grad=not self.frozen,
                                          compression_lr_multiplier=compression_lr_multiplier)
        self.binary_mask = binary_mask(self._mask)
        self.register_buffer("uniform", torch.zeros(weight_shape))
        self.mask_calculation_hook = MaskCalculationHook(self)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, tensor):
        self._mask.data = tensor
        self.binary_mask = binary_mask(self._mask)

    def _calc_training_binary_mask(self, weight):
        u = self.uniform if self.training and not self.frozen else None
        return calc_rb_binary_mask(self._mask, u, self.eps)

    def loss(self):
        return binary_mask(self._mask)


class MaskCalculationHook():
    def __init__(self, module):
        # pylint: disable=protected-access
        self.hook = module._register_state_dict_hook(self.hook_fn)

    def hook_fn(self, module, destination, prefix, local_metadata):
        module.binary_mask = binary_mask(module.mask)
        destination[prefix + '_binary_mask'] = module.binary_mask
        return destination

    def close(self):
        self.hook.remove()
