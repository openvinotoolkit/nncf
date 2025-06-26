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
import operator
from dataclasses import dataclass
from dataclasses import field
from functools import reduce
from typing import Optional, TypeVar

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

    @property
    def is_asym_mode(self):
        return self.mode in [CompressWeightsMode.INT4_ASYM, CompressWeightsMode.INT8_ASYM]

    @property
    def is_integer(self):
        """
        :return: True if compression type in integer, else False.
        """
        return self.mode not in [CompressWeightsMode.NF4, CompressWeightsMode.E2M1]

    def __hash__(self):
        return hash((self.mode.value, self.group_size))

    def __str__(self):
        return f"{self.mode.value}_{self.group_size}"


@dataclass
class WeightCompressionParameters:
    """
    Weight compression parameters determine how and what weight should be compressed.

    :param weight_name: Unique weight name.
    :param node_with_weight: Node with weight in the NNCF graph.
    :param weight_port_id: Number of elements in the weight array.
    :param weight_shape: Shape of the weight array.
    :param reduction_axes: Axes, along which to reduce (collect) different statistics (e.g. min, max).
    :param compression_config: Configuration of weight compression for the weight node.
    """

    weight_name: str
    node_with_weight: NNCFNode
    weight_port_id: int
    weight_shape: tuple[int, ...]
    reduction_axes: tuple[int, ...]
    compression_config: Optional[WeightCompressionConfig] = field(default_factory=WeightCompressionConfig)

    @property
    def num_weights(self) -> np.uint64:
        if not hasattr(self, "_num_weights"):
            self._num_weights = np.uint64(reduce(operator.mul, self.weight_shape, 1))
        return self._num_weights
