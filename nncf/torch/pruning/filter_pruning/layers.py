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

from typing import Any, Dict

import numpy as np
import torch
from torch import nn

import nncf
from nncf.common.graph import NNCFNodeName
from nncf.torch.layer_utils import COMPRESSION_MODULES
from nncf.torch.layer_utils import StatefullModuleInterface


@COMPRESSION_MODULES.register()
class FilterPruningMask(nn.Module, StatefullModuleInterface):
    """
    A module contains the mask for pruning.
    On forward pass applying the mask to weight and bias of the module.
    """

    MASK_APPLYING_DIM_KEY = "dim"
    NODE_NAME_KEY = "node_name"
    SIZE_KEY = "size_key"

    def __init__(self, size, node_name, dim=0):
        super().__init__()
        self.register_buffer("_binary_filter_pruning_mask", torch.ones(size))
        self.mask_applying_dim = dim
        self.node_name = node_name

    @property
    def binary_filter_pruning_mask(self) -> torch.Tensor:
        return self._binary_filter_pruning_mask

    @binary_filter_pruning_mask.setter
    def binary_filter_pruning_mask(self, mask: torch.Tensor):
        with torch.no_grad():
            self._binary_filter_pruning_mask.set_(mask)

    def forward(self, **params):
        new_params = []
        for param_name, param_value in params.items():
            # In case of None weight (or bias) mask shouldn't be applied
            if param_value is None:
                new_params.append(param_value)
                continue

            # For weights self.mask_applying_dim should be used, for bias dim=0
            dim = 0 if param_name == "bias" else self.mask_applying_dim
            new_params.append(
                apply_filter_binary_mask(
                    self.binary_filter_pruning_mask, param_value, node_name_for_logging=self.node_name, dim=dim
                )
            )
        return new_params

    def get_config(self) -> Dict[str, Any]:
        return {
            self.MASK_APPLYING_DIM_KEY: self.mask_applying_dim,
            self.NODE_NAME_KEY: self.node_name,
            self.SIZE_KEY: list(self.binary_filter_pruning_mask.size()),
        }

    @classmethod
    def from_config(cls, state: Dict[str, Any]) -> "FilterPruningMask":
        return FilterPruningMask(
            size=state[cls.SIZE_KEY], node_name=state[cls.NODE_NAME_KEY], dim=state[cls.MASK_APPLYING_DIM_KEY]
        )


def broadcast_filter_mask(filter_mask, shape, dim=0):
    broadcasted_shape = np.ones(len(shape), dtype=np.int64)
    broadcasted_shape[dim] = filter_mask.size(0)
    broadcasted_filter_mask = torch.reshape(filter_mask, tuple(broadcasted_shape))
    return broadcasted_filter_mask


def apply_filter_binary_mask(
    filter_mask: torch.Tensor,
    module_parameter: torch.nn.Parameter,
    node_name_for_logging: NNCFNodeName = "",
    dim: int = 0,
):
    """
    Applying binary filter mask to parameter of the module - usually to weight/bias of convolution or linear layer.
    Mask is applied to a given dimension without overriding parameter's values.
    :param filter_mask: binary mask (should have the same shape as conv weight on the given dimension)
    :param module_parameter: a tensor representing a module parameter (e.g. weight or bias of convolution)
    :param node_name_for_logging: name of the module to which the mask is applied
    :param dim: a dimension to apply the mask (0 by default)
    :return: result with applied mask
    """
    if filter_mask.size(0) != module_parameter.size(dim):
        raise nncf.InternalError(
            "Shape of mask = {} for module {} isn't broadcastable to weight shape={}."
            " ".format(filter_mask.shape, node_name_for_logging, module_parameter.shape)
        )
    broadcasted_filter_mask = broadcast_filter_mask(filter_mask, module_parameter.shape, dim)
    return module_parameter.mul(broadcasted_filter_mask)
