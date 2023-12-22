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
from typing import Any, Dict, List, Optional, TypeVar

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.parameters import CompressWeightsMode
from nncf.parameters import SensitivityMetric
from nncf.scopes import IgnoredScope

TModel = TypeVar("TModel")


class WeightCompressionAlgoBackend(ABC):
    @property
    @abstractmethod
    def matmul_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific metatypes for matmul layers.
        """

    @property
    @abstractmethod
    def embedding_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific metatypes for embedding layers.
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
            INT8_SYM stands for 8-bit integer symmetric quantization of all weights.
                Weights are quantized symmetrically with a fixed zero point equals to 128.
            INT8_ASYM is the same as INT8_SYM mode, but weights are quantized to a primary precision asymmetrically
                with a typical non-fixed zero point.
            INT4_SYM stands for a mixed-precision weights quantization with 4-bit integer as a primary precision.
                Weights are quantized to a primary precision symmetrically with a fixed zero point equals to 8.
                All embeddings and the last layer are always compressed to a backup precision, which is INT8_ASYM,
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
        all_layers: Optional[bool] = False,
        activations: Optional[Dict[str, Any]] = None,
        sensitivity_metric: Optional[SensitivityMetric] = SensitivityMetric.WEIGHT_QUANTIZATION_ERROR,
    ) -> TModel:
        """
        Compress weights of Linear and Embedding layers to 8-bit integer or to nf4
        depending on mode, ratio and group size.

        :param model: Model for applying weight compression.
        :param nodes_to_compress: List of nodes in the model's graph,
            corresponding to the layers for weight compression.
        :param mode: Defines a mode for weight compression.
            INT8_SYM stands for 8-bit integer symmetric quantization of all weights.
                Weights are quantized symmetrically with a fixed zero point equals to 128.
            INT8_ASYM is the same as INT8_SYM mode, but weights are quantized to a primary precision asymmetrically
                with a typical non-fixed zero point.
            INT4_SYM stands for a mixed-precision weights quantization with 4-bit integer as a primary precision.
                Weights are quantized to a primary precision symmetrically with a fixed zero point equals to 8.
                All embeddings and the last layer are always compressed to a backup precision, which is INT8_ASYM,
                by default. All others are quantized whether to 4-bit integer or to a backup precision depending on
                criteria and the given ratio.
            INT4_ASYM is the same as INT4_SYM mode, but weights are quantized to a primary precision asymmetrically
                with a typical non-fixed zero point.
            NF4 is the same as INT4_SYM mode, but primary precision is NF4 data type without zero point.
        :param ratio: The ratio between baseline and backup precisions (e.g. 0.9 means 90% of layers quantized to NF4
            and the rest to INT8_ASYM).
        :param group_size: Number of weights (e.g. 128) in the channel dimension
            that share quantization parameters (scale). The value -1 means no grouping.
        :param all_layers: Indicates whether embeddings and last layers should be compressed to a primary
            precision. By default, the backup precision is assigned for the embeddings and last layers.
        :param activations: The input activations of the layers considered for compression.
        :param sensitivity_metric: The sensitivity metric for assigning quantization precision to layers. In order to
            preserve the accuracy of the model, the more sensitive layers receives a higher precision.
        :return: A resulting model with compressed weights.
        """

    @staticmethod
    @abstractmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> TargetPoint:
        """
        Returns backend-specific target point.

        :param target_type: Type of the location that should be modified.
        :param target_node_name: Name of the located node.
        :param port_id: id of the port for the statistics distribution.
        :return: Backend-specific TargetPoint.
        """

    @staticmethod
    @abstractmethod
    def raw_statistic_collector(inplace: bool, num_samples: int = None) -> TensorStatisticCollectorBase:
        """
        Returns backend-specific raw statistic collector.
        This statistic collector is used for raw data calculation, without aggregating.

        :param inplace: Whether to calculate statistic inplace or not.
        :param num_samples: Maximum number of samples to collect.
        :return: Backend-specific TensorStatisticCollectorBase for the statistics calculation.
        """

    @staticmethod
    @abstractmethod
    def get_activation_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> int:
        """
        Returns input port id corresponding to activation input edge for
        the node.
        Supports only nodes that could have bias value.

        :param node: Node of NNCFGraph with bias value.
        :param nncf_graph: NNCFGraph instance with the node.
        :return: target input port id.
        """
