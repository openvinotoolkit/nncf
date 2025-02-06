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
from typing import Any, Dict, List

import torch

from nncf.torch.functions import logit
from nncf.torch.layer_utils import COMPRESSION_MODULES
from nncf.torch.layer_utils import CompressionParameter
from nncf.torch.layer_utils import StatefullModuleInterface
from nncf.torch.sparsity.layers import BinaryMask
from nncf.torch.sparsity.rb.functions import binary_mask
from nncf.torch.sparsity.rb.functions import calc_rb_binary_mask


@COMPRESSION_MODULES.register()
class RBSparsifyingWeight(BinaryMask, StatefullModuleInterface):
    WEIGHTS_SHAPE_KEY = "weight_shape"
    FROZEN_KEY = "frozen"
    COMPRESSION_LR_MULTIPLIER_KEY = "compression_lr_multiplier"
    EPS_KEY = "eps"

    def __init__(self, weight_shape: List[int], frozen=True, compression_lr_multiplier=None, eps=1e-6):
        super().__init__(weight_shape)
        self.frozen = frozen
        self.eps = eps
        self._mask = CompressionParameter(
            logit(torch.ones(weight_shape) * 0.99),
            requires_grad=not self.frozen,
            compression_lr_multiplier=compression_lr_multiplier,
        )
        self._compression_lr_multiplier = compression_lr_multiplier
        self.binary_mask = binary_mask(self._mask)
        self.register_buffer("uniform", torch.zeros(weight_shape))
        self.mask_calculation_hook = MaskCalculationHook(self)

    @property
    def mask(self) -> torch.nn.Parameter:
        return self._mask

    @mask.setter
    def mask(self, tensor: torch.Tensor):
        self._mask.data = tensor
        self.binary_mask = binary_mask(self._mask)

    def _calc_training_binary_mask(self, weight):
        u = self.uniform if self.training and not self.frozen else None
        return calc_rb_binary_mask(self._mask, u, self.eps)

    def loss(self):
        return binary_mask(self._mask)

    def get_config(self) -> Dict[str, Any]:
        return {
            self.WEIGHTS_SHAPE_KEY: list(self.mask.shape),
            self.FROZEN_KEY: self.frozen,
            self.COMPRESSION_LR_MULTIPLIER_KEY: self._compression_lr_multiplier,
            self.EPS_KEY: self.eps,
        }

    @classmethod
    def from_config(cls, state: Dict[str, Any]) -> "RBSparsifyingWeight":
        return RBSparsifyingWeight(
            weight_shape=state[cls.WEIGHTS_SHAPE_KEY],
            frozen=state[cls.FROZEN_KEY],
            compression_lr_multiplier=state[cls.COMPRESSION_LR_MULTIPLIER_KEY],
            eps=state[cls.EPS_KEY],
        )


class MaskCalculationHook:
    def __init__(self, module):
        self.hook = module._register_state_dict_hook(self.hook_fn)

    def hook_fn(self, module, destination, prefix, local_metadata):
        module.binary_mask = binary_mask(module.mask)
        destination[prefix + "_binary_mask"] = module.binary_mask
        return destination

    def close(self):
        self.hook.remove()
