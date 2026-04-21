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
import operator
from dataclasses import dataclass
from functools import cached_property
from functools import reduce

import numpy as np

from nncf.common.graph.graph import NNCFNode
from nncf.errors import InternalError
from nncf.errors import ValidationError
from nncf.parameters import CompressWeightsMode
from nncf.tensor import Tensor
from nncf.tensor.definitions import TensorDataType


@dataclass
class WeightCompressionConfig:
    """
    Configuration on how to compress (quantize) a specific weight.

    :param mode: Defines a mode for weight compression. Defaults to INT8_ASYM mode.
    :param group_size: Number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
        The value -1 means no grouping. Defaults to -1.
    :param codebook_values: Optional codebook values for CODEBOOK compression mode.
        Must be nncf.tensor.Tensor which wraps numpy array or ov tensor. Storing ov tensor is useful for having
        destination data type information available.
    """

    mode: CompressWeightsMode = CompressWeightsMode.INT8_ASYM
    group_size: int = -1
    codebook_values: Tensor | None = None

    def __post_init__(self) -> None:
        if self.group_size == 0 or self.group_size < -1:
            msg = f"Invalid group_size={self.group_size}. Group size must be a positive integer or -1."
            raise ValidationError(msg)

    @property
    def num_bits(self) -> int:
        """
        :return: number of bits that is used for storing a single quantized value in the given mode.
        """
        if self.is_codebook:
            if self.codebook_values is None:
                msg = f"Codebook values must be provided for {self.mode}"
                raise InternalError(msg)
            n_quants = self.codebook_values.size
            if n_quants <= 16:
                return 4
            if n_quants <= 256:
                return 8
            return 16

        if self.mode in [
            CompressWeightsMode.INT8_SYM,
            CompressWeightsMode.INT8_ASYM,
            CompressWeightsMode.FP8_E4M3,
            CompressWeightsMode.MXFP8_E4M3,
        ]:
            return 8
        return 4

    @property
    def is_asym_mode(self) -> bool:
        return self.mode in [CompressWeightsMode.INT4_ASYM, CompressWeightsMode.INT8_ASYM]

    @property
    def is_integer(self) -> bool:
        """
        :return: True if compression type in integer, else False.
        """
        return self.mode not in [
            CompressWeightsMode.NF4,
            CompressWeightsMode.MXFP4,
            CompressWeightsMode.MXFP8_E4M3,
            CompressWeightsMode.FP8_E4M3,
            CompressWeightsMode.FP4,
            CompressWeightsMode.NVFP4,
            CompressWeightsMode.CODEBOOK,
            CompressWeightsMode.ADAPTIVE_CODEBOOK,
            CompressWeightsMode.CB4,
        ]

    @property
    def is_codebook(self) -> bool:
        """
        :return: True if compression type is codebook, else False.
        """
        return self.mode in [
            CompressWeightsMode.CODEBOOK,
            CompressWeightsMode.CB4,
            CompressWeightsMode.ADAPTIVE_CODEBOOK,
        ]

    @property
    def compression_dtype(self) -> TensorDataType:
        """
        :return: data type that is used to store compressed weights.
        """
        if self.is_codebook:
            if self.codebook_values is None:
                msg = f"Codebook values must be provided for {self.mode}"
                raise InternalError(msg)
            n_quants = self.codebook_values.size
            if n_quants <= 16:
                return TensorDataType.uint4
            if n_quants <= 256:
                return TensorDataType.uint8
            return TensorDataType.uint16
        dtype_per_mode = {
            CompressWeightsMode.INT4_SYM: TensorDataType.int4,
            CompressWeightsMode.INT4_ASYM: TensorDataType.uint4,
            CompressWeightsMode.INT8_ASYM: TensorDataType.uint8,
            CompressWeightsMode.INT8_SYM: TensorDataType.int8,
            CompressWeightsMode.NF4: TensorDataType.nf4,
            CompressWeightsMode.FP4: TensorDataType.f4e2m1,
            CompressWeightsMode.MXFP4: TensorDataType.f4e2m1,
            CompressWeightsMode.NVFP4: TensorDataType.f4e2m1,
            CompressWeightsMode.FP8_E4M3: TensorDataType.f8e4m3,
            CompressWeightsMode.MXFP8_E4M3: TensorDataType.f8e4m3,
        }
        return dtype_per_mode[self.mode]

    def __hash__(self) -> int:
        return hash((self.mode.value, self.group_size))


@dataclass
class WeightCompressionParameters:
    """
    Weight compression parameters determine how and what weight should be compressed.

    :param weight_name: Unique weight name.
    :param node_with_weight: Node with weight in the NNCF graph.
    :param weight_port_id: Port id of the weight in the node with weight in the NNCF graph.
    :param weight_dtype: Data type of the weight tensor.
    :param weight_shape: Shape of the weight array.
    :param reduction_axes: Axes, along which to reduce (collect) different statistics (e.g. min, max).
    :param compression_config: Configuration of weight compression for the weight node.
    """

    weight_name: str
    node_with_weight: NNCFNode
    weight_port_id: int
    weight_dtype: TensorDataType
    weight_shape: tuple[int, ...]
    reduction_axes: tuple[int, ...]
    compression_config: WeightCompressionConfig | None

    @cached_property
    def num_weights(self) -> np.uint64:
        """
        :return: Total number of weights in the weight tensor.
        """
        # Explicitly use unsigned 64-bit integer for number of weight in weight compression.
        # To avoid overflow when calculating the total number of weights for large models.
        return np.uint64(reduce(operator.mul, self.weight_shape, 1))
