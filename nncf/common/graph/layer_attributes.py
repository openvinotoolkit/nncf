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
from abc import ABC
from abc import abstractmethod
from enum import Enum
from typing import List
from typing import Tuple


class Dtype(Enum):
    FLOAT = 'float'
    INTEGER = 'int'


class BaseLayerAttributes(ABC):
    """
    This class stores base useful for some algorithms attributes
    of modules/layers.
    """


class WeightedLayerAttributes(BaseLayerAttributes):
    """
    Represents a layer with weights.
    """
    def __init__(self, weight_requires_grad: bool, dtype: Dtype = Dtype.FLOAT):
        self.weight_requires_grad = weight_requires_grad
        self.dtype = dtype

    def __eq__(self, other):
        return isinstance(other, WeightedLayerAttributes) \
               and self.weight_requires_grad == other.weight_requires_grad

    @abstractmethod
    def get_weight_shape(self) -> List[int]:
        pass

    @abstractmethod
    def get_num_filters(self) -> int:
        pass


class GenericWeightedLayerAttributes(WeightedLayerAttributes):
    """
    Represents a weighted layer for which there is no information ahead of time
    of the exact meaning of the weight indices.
    """
    def __init__(self, weight_requires_grad: bool, weight_shape: List[int],
                 filter_dimension_idx: int = 0):
        super().__init__(weight_requires_grad)
        self.weight_shape = weight_shape
        self.filter_dimension_idx = filter_dimension_idx

    def get_weight_shape(self) -> List[int]:
        return self.weight_shape

    def get_num_filters(self) -> int:
        return self.weight_shape[0]


class LinearLayerAttributes(WeightedLayerAttributes):
    def __init__(self,
                 weight_requires_grad: bool,
                 in_features: int,
                 out_features: int):
        super().__init__(weight_requires_grad)
        self.in_features = in_features
        self.out_features = out_features

    def get_weight_shape(self) -> List[int]:
        return [self.out_features, self.in_features]

    def get_num_filters(self) -> int:
        return self.out_features


class ConvolutionLayerAttributes(WeightedLayerAttributes):
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
                 groups: int,
                 transpose: bool,
                 padding_values: List[int]):
        super().__init__(weight_requires_grad)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.transpose = transpose
        self.padding_values = padding_values

    def __eq__(self, other):
        return isinstance(other, ConvolutionLayerAttributes) \
               and super().__eq__(other) \
               and self.in_channels == other.in_channels \
               and self.out_channels == other.out_channels \
               and self.kernel_size == other.kernel_size \
               and self.stride == other.stride \
               and self.groups == other.groups \
               and self.transpose == other.transpose

    def get_weight_shape(self) -> List[int]:
        if not self.transpose:
            return [self.out_channels, self.in_channels // self.groups, *self.kernel_size]
        return [self.in_channels, self.out_channels // self.groups, *self.kernel_size]

    def get_num_filters(self) -> int:
        return self.out_channels


class GroupNormLayerAttributes(WeightedLayerAttributes):
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
        return isinstance(other, GroupNormLayerAttributes) \
               and super().__eq__(other) \
               and self.num_channels == other.num_channels \
               and self.num_groups == other.num_groups

    def get_weight_shape(self) -> List[int]:
        return [self.num_channels]

    def get_num_filters(self) -> int:
        return 1
