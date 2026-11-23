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

INT8_MODES = [CompressWeightsMode.INT8_ASYM, CompressWeightsMode.INT8_SYM]
NON_INT8_MODES = [mode for mode in CompressWeightsMode if mode not in INT8_MODES]
FIXED_GROUP_SIZE_MODES = {
    CompressWeightsMode.MXFP4: 32,
    CompressWeightsMode.MXFP8_E4M3: 32,
    CompressWeightsMode.NVFP4: 16,
}
PER_CHANNEL_BY_DEFAULT_MODES = [
    *INT8_MODES,
    CompressWeightsMode.CODEBOOK,
    CompressWeightsMode.CB4,
    CompressWeightsMode.ADAPTIVE_CODEBOOK,
]
DEFAULT_GROUP_SIZE_FOR_NON_INT8_MODES = 128


def get_default_group_size(mode: CompressWeightsMode) -> int:
    """
    Returns the group size to use for the given mode when no group size is requested explicitly.

    :param mode: Compression mode to get the default group size for.
    :return: Default group size for the given mode.
    """
    if mode in FIXED_GROUP_SIZE_MODES:
        return FIXED_GROUP_SIZE_MODES[mode]
    if mode in PER_CHANNEL_BY_DEFAULT_MODES:
        return -1
    return DEFAULT_GROUP_SIZE_FOR_NON_INT8_MODES


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
    :param narrow_range: Whether a symmetric integer mode uses the narrow range [-(2^(b-1) - 1), 2^(b-1) - 1]
        with the scale max|w| / (2^(b-1) - 1) instead of the full range. Defaults to False.
    """

    mode: CompressWeightsMode = CompressWeightsMode.INT8_ASYM
    group_size: int = -1
    codebook_values: Tensor | None = None
    narrow_range: bool = False

    def __post_init__(self) -> None:
        if self.group_size == 0 or self.group_size < -1:
            msg = f"Invalid group_size={self.group_size}. Group size must be a positive integer or -1."
            raise ValidationError(msg)
        if self.narrow_range and self.mode not in [
            CompressWeightsMode.INT2_SYM,
            CompressWeightsMode.INT3_SYM,
            CompressWeightsMode.INT4_SYM,
            CompressWeightsMode.INT8_SYM,
        ]:
            msg = f"narrow_range applies to the symmetric integer modes only, not to {self.mode.value}."
            raise ValidationError(msg)

    @property
    def num_bits(self) -> int:
        """
        Number of bits used to store a single quantized value in the given mode.
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

        mode_to_bits_map = {
            CompressWeightsMode.INT8_ASYM: 8,
            CompressWeightsMode.INT8_SYM: 8,
            CompressWeightsMode.INT4_SYM: 4,
            CompressWeightsMode.INT4_ASYM: 4,
            CompressWeightsMode.NF4: 4,
            CompressWeightsMode.FP4: 4,
            CompressWeightsMode.MXFP4: 4,
            CompressWeightsMode.NVFP4: 4,
            CompressWeightsMode.FP8_E4M3: 8,
            CompressWeightsMode.MXFP8_E4M3: 8,
            CompressWeightsMode.INT3_SYM: 3,
            CompressWeightsMode.INT2_SYM: 2,
            CompressWeightsMode.INT1_SYM: 1,
        }

        try:
            return mode_to_bits_map[self.mode]
        except KeyError as e:
            msg = f"Unsupported compression mode for num_bits: {self.mode}"
            raise InternalError(msg) from e

    @property
    def is_asym_mode(self) -> bool:
        return self.mode in [CompressWeightsMode.INT4_ASYM, CompressWeightsMode.INT8_ASYM]

    @property
    def is_integer(self) -> bool:
        """
        True if compression type in integer, else False.
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
        True if compression type is codebook, else False.
        """
        return self.mode in [
            CompressWeightsMode.CODEBOOK,
            CompressWeightsMode.CB4,
            CompressWeightsMode.ADAPTIVE_CODEBOOK,
        ]

    @property
    def compression_dtype(self) -> TensorDataType:
        """
        Data type that is used to store compressed weights.
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
            CompressWeightsMode.INT1_SYM: TensorDataType.uint1,
            CompressWeightsMode.INT2_SYM: TensorDataType.int2,
            CompressWeightsMode.INT3_SYM: TensorDataType.int3,
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

    @property
    def is_symmetric_represented_by_unsigned(self) -> bool:
        """
        True if compression type is symmetric and represented by unsigned integer, else False.
        """
        return self.mode == CompressWeightsMode.INT3_SYM

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
        Total number of weights in the weight tensor.
        """
        # Explicitly use unsigned 64-bit integer for number of weight in weight compression.
        # To avoid overflow when calculating the total number of weights for large models.
        return np.uint64(reduce(operator.mul, self.weight_shape, 1))
