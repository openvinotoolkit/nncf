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

from abc import ABC
from abc import abstractmethod
from typing import List, Optional, TypeVar

from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.parameters import CompressWeightsMode
from nncf.scopes import IgnoredScope

TModel = TypeVar("TModel")


class WeightCompressionAlgoBackend(ABC):
    @property
    @abstractmethod
    def weighted_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific metatypes.
        """

    @staticmethod
    @abstractmethod
    def is_node_with_weights(node: NNCFNode) -> bool:
        """
        Checks whether the node with weights or not.

        :param node: NNCFNode to check.
        :return: Boolean indicating whether the node has weights or not.
        """

    @staticmethod
    @abstractmethod
    def validate_params(mode: CompressWeightsMode, ignored_scope: Optional[IgnoredScope] = None) -> None:
        """
        Performs validation of the algorithm's parameters and raises an error for unsupported configuration of
        parameters. Should be called on early algorithm steps to prevent execution of time-consuming operations.

        :param mode: Defines a mode for weight compression.
            INT8 stands for 8-bit integer quantization of all weights.
            INT4_SYM stands for a mixed-precision weights quantization with 4-bit integer as a primary precision.
                Weights are quantized to a primary precision symmetrically with a fixed zero point equals to 8.
                The first and the last layers are always compressed to a backup precision, which is 8-bit integer,
                by default. All others are quantized whether to 4-bit integer or to a backup precision depending on
                criteria and the given ratio.
            INT4_ASYM is the same as INT4_SYM mode, but weights are quantized to a primary precision asymmetrically
                with a typical non-fixed zero point.
            NF4 is the same as INT4_SYM mode, but primary precision is NF4 data type without zero point.
        :param ignored_scope: An ignored scope that defined the list of model control
            flow graph nodes to be ignored during quantization.
        """

    @staticmethod
    @abstractmethod
    def do_compression(
        model: TModel,
        nodes_to_compress: List[NNCFNode],
        mode: CompressWeightsMode,
        ratio: float = None,
        group_size: int = None,
    ) -> TModel:
        """
        Compress weights of Linear and Embedding layers to 8-bit integer or to nf4
        depending on mode, ratio and group size.

        :param model: Model for applying weight compression.
        :param nodes_to_compress: List of nodes in the model's graph,
            corresponding to the layers for weight compression.
        :param mode: Defines a mode for weight compression.
            INT8 stands for 8-bit integer quantization of all weights.
            INT4_SYM stands for a mixed-precision weights quantization with 4-bit integer as a primary precision.
                Weights are quantized to a primary precision symmetrically with a fixed zero point equals to 8.
                The first and the last layers are always compressed to a backup precision, which is 8-bit integer,
                by default. All others are quantized whether to 4-bit integer or to a backup precision depending on
                criteria and the given ratio.
            INT4_ASYM is the same as INT4_SYM mode, but weights are quantized to a primary precision asymmetrically
                with a typical non-fixed zero point.
            NF4 is the same as INT4_SYM mode, but primary precision is NF4 data type without zero point.
        :param ratio: The ratio between baseline and backup precisions (e.g. 0.9 means 90% of layers quantized to NF4
            and the rest to INT8).
        :param group_size: Number of weights (e.g. 128) in the channel dimension
            that share quantization parameters (scale). The value -1 means no grouping.
        :return: A resulting model with compressed weights.
        """
