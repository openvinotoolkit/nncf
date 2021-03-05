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
    def __init__(self, weight_requires_grad: bool):
        self.weight_requires_grad = weight_requires_grad

    def __eq__(self, other):
        return isinstance(other, BaseModuleAttributes) \
               and self.weight_requires_grad == other.weight_requires_grad


class ConvolutionModuleAttributes(BaseModuleAttributes):
    def __init__(self,
                 weight_requires_grad: bool,
                 in_channels: int,
                 out_channels: int,
                 stride: Tuple[int, int],
                 groups: int):
        super().__init__(weight_requires_grad)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups

    def __eq__(self, other):
        return isinstance(other, ConvolutionModuleAttributes) \
               and super().__eq__(other) \
               and self.in_channels == other.in_channels \
               and self.out_channels == other.out_channels \
               and self.stride == other.stride \
               and self.groups == other.groups
