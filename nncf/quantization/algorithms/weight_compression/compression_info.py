# Copyright (c) 2023 Intel Corporation
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
from typing import Any, Optional, TypeVar

from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.parameters import CompressWeightsMode

TWeightType = TypeVar("TWeightType")


@dataclass
class WeightCompressionConfig:
    """
    Information on how to compress (quantize) a specific weight.

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


@dataclass
class WeightNodeParams:
    """
    Information about weight node in the ov.Model that is useful for weight compression.

    :param reduction_axis: Axis, along which to reduce (collect) different statistics (e.g. min, max).
    :param num_weights: Number of elements in the weight array.
    :param fq_name: Name for the inserted weight compression operation.
    :param weight_node: The weight node itself.
    :param original_weight_dtype: Type of elements in the weight array.
    :param compression_config: Configuration of weight compression for the weight node.
    :param metatype: Metatype of the corresponding operation with weight.
    :param node_name: String representation of the node in the NNCFGraph. It is intended for accessing collected
        activations.
    """

    reduction_axis: int
    num_weights: int
    fq_name: str
    # TODO(nlyalyus): Should be NNCFNode
    weight_node: Any  # ov.Node
    original_weight_dtype: TWeightType
    compression_config = WeightCompressionConfig()
    metatype: OperatorMetatype = None
    node_name: str = None
