"""
 Copyright (c) 2021 Intel Corporation
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
from typing import Tuple


class BaseModuleAttributes:
    """
    This class stores base useful for some algorithms attributes
    of modules/layers.
    """

    def __init__(self, weight_requires_grad: bool):
        self.weight_requires_grad = weight_requires_grad

    def __eq__(self, other):
        return isinstance(other, BaseModuleAttributes) \
               and self.weight_requires_grad == other.weight_requires_grad


class ConvolutionModuleAttributes(BaseModuleAttributes):
    """
    This class stores attributes of convolution modules/layers
    that are useful for some algorithms.
    """

    def __init__(self,
                 weight_requires_grad: bool,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 groups: int):
        super().__init__(weight_requires_grad)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups

    def __eq__(self, other):
        return isinstance(other, ConvolutionModuleAttributes) \
               and super().__eq__(other) \
               and self.in_channels == other.in_channels \
               and self.out_channels == other.out_channels \
               and self.kernel_size == other.kernel_size \
               and self.stride == other.stride \
               and self.groups == other.groups


class GroupNormModuleAttributes(BaseModuleAttributes):
    """
    This class stores attributes of group normalization modules/layers
    that are useful for some algorithms.
    """

    def __init__(self,
                 weight_requires_grad: bool,
                 num_channels: int,
                 num_groups: int):
        super().__init__(weight_requires_grad)
        self.num_channels = num_channels
        self.num_groups = num_groups

    def __eq__(self, other):
        return isinstance(other, BaseModuleAttributes) \
               and super().__eq__(other) \
               and self.num_channels == other.num_channels \
               and self.num_groups == other.num_groups
