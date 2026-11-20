# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
from dataclasses import dataclass
from enum import Enum


class Dtype(Enum):
    FLOAT = "float"
    INTEGER = "int"


class BaseLayerAttributes(ABC):  # noqa: B024
    """
    This class stores base useful for some algorithms attributes
    of modules/layers.
    """

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, self.__class__) and self.__dict__ == __o.__dict__


class WeightedLayerAttributes(BaseLayerAttributes):
    def __init__(self, weight_requires_grad: bool, dtype: Dtype = Dtype.FLOAT, with_bias: bool = False):
        """

        :param weight_requires_grad: Is True if gradients need to be computed for the corresponding Tensor,
        False otherwise.
        :param dtype: is an object that represents the type of data.
        :param with_bias: Operation include bias.
        """
        self.weight_requires_grad = weight_requires_grad
        self.dtype = dtype
        self.with_bias = with_bias


class GenericWeightedLayerAttributes(WeightedLayerAttributes):
    """
    Represents a weighted layer for which there is no information ahead of time
    of the exact meaning of the weight indices.
    """

    def __init__(
        self,
        weight_requires_grad: bool,
        weight_shape: list[int],
        filter_dimension_idx: int = 0,
        with_bias: bool = False,
    ):
        """

        :param weight_requires_grad: Is True if gradients need to be computed for the corresponding Tensor,
        False otherwise.
        :param weight_shape: shape of weight tensor.
        :param filter_dimension_idx: the axis, along which the filters are stored.
        """
        super().__init__(weight_requires_grad=weight_requires_grad, with_bias=with_bias)
        self.weight_shape = weight_shape
        self.filter_dimension_idx = filter_dimension_idx


class LinearLayerAttributes(WeightedLayerAttributes):
    def __init__(
        self,
        weight_requires_grad: bool,
        in_features: int,
        out_features: int,
        with_bias: bool = True,
    ):
        """

        :param weight_requires_grad: Is True if gradients need to be computed for the corresponding Tensor,
        False otherwise.
        :param in_features: number of input channels in the layer's input.
        :param out_features: number of channels produced by the layer.
        """
        super().__init__(weight_requires_grad, with_bias=with_bias)
        self.in_features = in_features
        self.out_features = out_features


class ConvolutionLayerAttributes(WeightedLayerAttributes):
    def __init__(
        self,
        weight_requires_grad: bool,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        dilations: tuple[int, ...],
        groups: int,
        transpose: bool,
        padding_values: tuple[int, ...] | int,
        with_bias: bool = False,
        output_padding_values: tuple[int, ...] | int | None = None,
    ):
        """

        :param weight_requires_grad: Is True if gradients need to be computed for the corresponding Tensor,
        False otherwise.
        :param in_channels: Number of input channels in the layer's input.
        :param out_channels: Number of channels produced by the layer.
        :param kernel_size: Size of the convolving kernel.
        :param stride: Stride of the convolution.
        :param groups: Number of blocked connections from input channels to output channels.
        :param transpose: If set to `True`, the layer is an ordinary convolution, otherwise - transpose one.
        :param padding_values: Defines the amount of padding applied to the layer's input.
        :param with_bias: Operation include bias.
        :param output_padding_values: Defines the amount of output padding applied to the layer's output, for transpose.
        """
        super().__init__(weight_requires_grad=weight_requires_grad, with_bias=with_bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilations = dilations
        self.groups = groups
        self.transpose = transpose
        self.padding_values = padding_values
        self.output_padding_values = output_padding_values


@dataclass
class ConstantLayerAttributes(BaseLayerAttributes):
    """
    :param name: Constant name.
    :param shape: Constant shape.
    """

    name: str
    shape: list[int]
