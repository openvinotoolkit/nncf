"""
 Copyright (c) 2023 Intel Corporation
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
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Tuple, Union


class Dtype(Enum):
    FLOAT = "float"
    INTEGER = "int"


class BaseLayerAttributes(ABC):
    """
    This class stores base useful for some algorithms attributes
    of modules/layers.
    """


class MultipleInputLayerAttributes(BaseLayerAttributes):
    def __init__(self, axis: int):
        """

        :param axis: the dimension over which the inputs are combined (e.g. concatenated).
        """
        self.axis = axis

    def __eq__(self, other: Any):
        return isinstance(other, MultipleInputLayerAttributes) and self.axis == other.axis


class MultipleOutputLayerAttributes(BaseLayerAttributes):
    def __init__(self, chunks: Union[int, List], axis: int):
        """

        :param chunks:  number of chunks (outputs)
        :param axis: the dimension along which to make multiple outputs (e.g. split the tensor).
        """
        self.chunks = chunks
        self.axis = axis

    def __eq__(self, other: Any):
        return (
            isinstance(other, MultipleOutputLayerAttributes) and self.chunks == other.chunks and self.axis == other.axis
        )


class WeightedLayerAttributes(BaseLayerAttributes):
    def __init__(self, weight_requires_grad: bool, dtype: Dtype = Dtype.FLOAT):
        """

        :param weight_requires_grad: Is True if gradients need to be computed for the corresponding Tensor,
        False otherwise.
        :param dtype: is an object that represents the type of data.
        """
        self.weight_requires_grad = weight_requires_grad
        self.dtype = dtype

    def __eq__(self, other: Any):
        return isinstance(other, WeightedLayerAttributes) and self.weight_requires_grad == other.weight_requires_grad

    @abstractmethod
    def get_weight_shape(self) -> List[int]:
        pass

    def get_num_filters(self) -> int:
        weight_shape = self.get_weight_shape()
        return weight_shape[self.get_target_dim_for_compression()]

    @abstractmethod
    def get_target_dim_for_compression(self) -> int:
        pass


class GenericWeightedLayerAttributes(WeightedLayerAttributes):
    """
    Represents a weighted layer for which there is no information ahead of time
    of the exact meaning of the weight indices.
    """

    def __init__(self, weight_requires_grad: bool, weight_shape: List[int], filter_dimension_idx: int = 0):
        """

        :param weight_requires_grad: Is True if gradients need to be computed for the corresponding Tensor,
        False otherwise.
        :param weight_shape: shape of weight tensor.
        :param filter_dimension_idx: the axis along which the filters are stored.
        """
        super().__init__(weight_requires_grad)
        self.weight_shape = weight_shape
        self.filter_dimension_idx = filter_dimension_idx

    def get_weight_shape(self) -> List[int]:
        return self.weight_shape

    def get_target_dim_for_compression(self) -> int:
        return 0


class LinearLayerAttributes(WeightedLayerAttributes):
    def __init__(self, weight_requires_grad: bool, in_features: int, out_features: int, bias: bool = True):
        """

        :param weight_requires_grad: Is True if gradients need to be computed for the corresponding Tensor,
        False otherwise.
        :param in_features: number of input channels in the layer's input.
        :param out_features: number of channels produced by the layer.
        :param bias: If set to ``False``, the layer doesn't learn an additive bias.
        """
        super().__init__(weight_requires_grad)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def get_weight_shape(self) -> List[int]:
        return [self.out_features, self.in_features]

    def get_bias_shape(self) -> int:
        return self.out_features if self.bias is True else 0

    def get_target_dim_for_compression(self) -> int:
        return 0


class ConvolutionLayerAttributes(WeightedLayerAttributes):
    def __init__(
        self,
        weight_requires_grad: bool,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        groups: int,
        transpose: bool,
        padding_values: Tuple[int, ...],
    ):
        """

        :param weight_requires_grad: Is True if gradients need to be computed for the corresponding Tensor,
        False otherwise.
        :param in_channels: number of input channels in the layer's input.
        :param out_channels: number of channels produced by the layer.
        :param kernel_size: size of the convolving kernel.
        :param stride: stride of the convolution.
        :param groups: number of blocked connections from input channels to output channels.
        :param transpose: If set to `True`, the layer is an ordinary convolution, otherwise - transpose one.
        :param padding_values: defines the amount of padding applied to the layer's input.
        """
        super().__init__(weight_requires_grad)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.transpose = transpose
        self.padding_values = padding_values

    def __eq__(self, other: Any):
        return (
            isinstance(other, ConvolutionLayerAttributes)
            and super().__eq__(other)
            and self.in_channels == other.in_channels
            and self.out_channels == other.out_channels
            and self.kernel_size == other.kernel_size
            and self.stride == other.stride
            and self.groups == other.groups
            and self.transpose == other.transpose
        )

    def get_weight_shape(self) -> List[int]:
        if not self.transpose:
            return [self.out_channels, self.in_channels // self.groups, *self.kernel_size]
        return [self.in_channels, self.out_channels // self.groups, *self.kernel_size]

    def get_target_dim_for_compression(self) -> int:
        # Always quantize per each "out" channel
        if self.transpose:
            return 1
        return 0


class GroupNormLayerAttributes(WeightedLayerAttributes):
    def __init__(self, weight_requires_grad: bool, num_channels: int, num_groups: int):
        """

        :param weight_requires_grad: Is True if gradients need to be computed for the corresponding Tensor,
        False otherwise.
        :param num_channels: number of channels expected in the layer's input.
        :param num_groups: number of groups to separate the channels into.
        """
        super().__init__(weight_requires_grad)
        self.num_channels = num_channels
        self.num_groups = num_groups

    def __eq__(self, other: Any):
        return (
            isinstance(other, GroupNormLayerAttributes)
            and super().__eq__(other)
            and self.num_channels == other.num_channels
            and self.num_groups == other.num_groups
        )

    def get_weight_shape(self) -> List[int]:
        return [self.num_channels]

    def get_target_dim_for_compression(self) -> int:
        return 0


@dataclass
class ReshapeLayerAttributes(BaseLayerAttributes):
    """
    :param input_shape: number of elements of each of the axes of a input tensor.
    :param output_shape: number of elements of each of the axes of a output tensor.
    """

    input_shape: List[int]
    output_shape: List[int]


@dataclass
class TransposeLayerAttributes(BaseLayerAttributes):
    """
    :param dim0: the first dimension to be transposed.
    :param dim1: the second dimension to be transposed.
    """

    dim0: int
    dim1: int

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, TransposeLayerAttributes)
            and super().__eq__(other)
            and self.dim0 == other.dim0
            and self.dim1 == other.dim1
        )


@dataclass
class PermuteLayerAttributes(BaseLayerAttributes):
    """
    :param permutation: the desired ordering of dimensions.
    """

    permutation: List[int]

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, PermuteLayerAttributes)
            and super().__eq__(other)
            and len(self.permutation) == len(other.permutation)
            and (l == r for l, r in zip(self.permutation, other.permutation))
        )


@dataclass
class GetItemLayerAttributes(BaseLayerAttributes):
    """
    :param key: usually int, tuple of int or slice.
    """

    key: Any


@dataclass
class PadLayerAttributes(BaseLayerAttributes):
    """
    :param mode: mode of the padding operation.
    :param value: fill value of the padding operation.
    """

    mode: str = "constant"
    value: float = 0
