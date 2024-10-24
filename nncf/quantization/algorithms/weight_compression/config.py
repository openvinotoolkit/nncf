# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from dataclasses import field
from typing import Optional, Tuple, TypeVar

import numpy as np

from nncf.common.graph.graph import NNCFNode
from nncf.parameters import CompressWeightsMode

TWeightType = TypeVar("TWeightType")


@dataclass
class WeightCompressionConfig:
    """
    Configuration on how to compress (quantize) a specific weight.

    :param mode: Defines a mode for weight compression. Defaults to INT8_ASYM mode.
    :param group_size: Number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
        The value -1 means no grouping. Defaults to -1.
    """

    mode: Optional[CompressWeightsMode] = CompressWeightsMode.INT8_ASYM
    group_size: Optional[int] = -1

    @property
    def num_bits(self):
        """
        :return: number of bits that is used for storing a single quantized value in the given mode.
        """
        return 8 if self.mode in [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM] else 4

    def is_integer(self):
        """
        :return: True if compression type in integer, else False.
        """
        return self.mode not in [CompressWeightsMode.NF4, CompressWeightsMode.E2M1]

    def __hash__(self):
        return hash((self.mode.value, self.group_size))


@dataclass
class WeightCompressionParameters:
    """
    Weight compression parameters determine how and what weight should be compressed.

    :param weight_name: Unique weight name.
    :param node_with_weight: Node with weight in the NNCF graph.
    :param weight_port_id: Number of elements in the weight array.
    :param num_weights: Number of elements in the weight array.
    :param reduction_axes: Axes, along which to reduce (collect) different statistics (e.g. min, max).
    :param compression_config: Configuration of weight compression for the weight node.
    """

    weight_name: str
    node_with_weight: NNCFNode
    weight_port_id: int
    num_weights: np.uint64
    reduction_axes: Tuple[int, ...]
    compression_config: Optional[WeightCompressionConfig] = field(default_factory=WeightCompressionConfig)

    def __post_init__(self):
        # Explicitly cast num_weights to avoid overflow on finding total number of weights.
        # The issue happens on Windows, because np.ndarray.size() returns np.int32 and sum of weights is more than 2^32.
        self.num_weights = np.uint64(self.num_weights)
