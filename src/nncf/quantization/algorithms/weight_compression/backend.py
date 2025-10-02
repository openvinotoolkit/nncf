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

from abc import ABC
from abc import abstractmethod
from typing import Callable, Iterable, Optional, TypeVar

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.patterns.patterns import GraphPattern
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.experimental.common.tensor_statistics.collectors import HAWQAggregator
from nncf.experimental.common.tensor_statistics.collectors import MaxVarianceReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanAbsMaxReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanAggregator
from nncf.experimental.common.tensor_statistics.collectors import MeanVarianceReducer
from nncf.experimental.common.tensor_statistics.collectors import RawReducer
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.statistics import HessianTensorStatistic
from nncf.experimental.common.tensor_statistics.statistics import MaxVarianceTensorStatistic
from nncf.experimental.common.tensor_statistics.statistics import MeanMagnitudeTensorStatistic
from nncf.experimental.common.tensor_statistics.statistics import MeanVarianceTensorStatistic
from nncf.parameters import CompressionFormat
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.lora_correction import LoraCorrectionAlgorithm
from nncf.quantization.algorithms.weight_compression.parameters import CompressedWeight
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType

TModel = TypeVar("TModel")


class WeightCompressionAlgoBackend(ABC):
    @property
    @abstractmethod
    def matmul_metatypes(self) -> list[OperatorMetatype]:
        """
        Property for the backend-specific metatypes for matmul layers.
        """

    @property
    @abstractmethod
    def convolution_metatypes(self) -> list[OperatorMetatype]:
        """
        Property for the backend-specific metatypes for convolution layers.
        """

    @property
    @abstractmethod
    def embedding_metatypes(self) -> list[OperatorMetatype]:
        """
        Property for the backend-specific metatypes for embedding layers.
        """

    @staticmethod
    @abstractmethod
    def is_node_with_weights(node: NNCFNode, graph: NNCFGraph) -> bool:
        """
        Checks whether the node with weights or not.

        :param node: The node to check.
        :param graph: The model graph.
        :return: True if the node contains weights, False otherwise.
        """

    @staticmethod
    @abstractmethod
    def get_reduction_axes(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> Optional[tuple[int]]:
        """
        Returns reduction axes without axes that corresponds to weight channels of the node with weight.

        :param node_with_weight: The node with weight.
        :param weight_port_id: The input port ID that corresponds to weight.
        :param graph: The model graph.
        :return: Reduction shape in tuple format or None if not applicable.
        """

    @staticmethod
    @abstractmethod
    def get_weight_names_and_port_ids(node: NNCFNode, graph: NNCFGraph) -> list[tuple[str, int]]:
        """
        Returns a list of weight names and port ids for the given node.

        :param node: The node.
        :param graph: The model graph.
        :return: List of tuples with weight names and port ids.
        """

    @abstractmethod
    def get_weight(self, node_with_weight: NNCFNode, weight_port_id: int, model: TModel, graph: NNCFGraph) -> Tensor:
        """
        Returns a weight associated with the given node on the given port id.

        :param node_with_weight: The node with weight.
        :param weight_port_id: The weight port id for given node with weight.
        :param model: The model.
        :param graph: The model graph associated with the model.
        :return: The weight tensor.
        """

    @abstractmethod
    def get_weight_dtype(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: TModel, graph: NNCFGraph
    ) -> TensorDataType:
        """
        Returns a weight data type associated with the given node on the given port id.

        :param node_with_weight: The node with weight.
        :param weight_port_id: The weight port id for given node with weight.
        :param model: The model.
        :param graph: The model graph associated with the model.
        :return: The weight data type.
        """

    @staticmethod
    @abstractmethod
    def get_weight_shape(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> tuple:
        """
        Returns a weight shape associated with the given node on the given port id.

        :param node_with_weight: The node with weight.
        :param weight_port_id: The weight port id for given node with weight.
        :param graph: The model graph associated with the model.
        :return: The weight shape.
        """

    @abstractmethod
    def set_weight(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: TModel, graph: NNCFGraph, weight: Tensor
    ) -> None:
        """
        Update a weight associated with the given node on the given port id.

        :param node_with_weight: The node with weight.
        :param weight_port_id: The weight port id for given node with weight.
        :param model: The model.
        :param graph: The model graph associated with the model.
        :param weight: The weight tensor.
        """

    @abstractmethod
    def transform_model(
        self,
        model: TModel,
        graph: NNCFGraph,
        weight_compression_parameters: Iterable[WeightCompressionParameters],
        precomputed_compressed_weights: Optional[dict[str, CompressedWeight]] = None,
        lora_correction_algo: Optional[LoraCorrectionAlgorithm] = None,
        compression_format: CompressionFormat = CompressionFormat.DQ,
        advanced_parameters: Optional[AdvancedCompressionParameters] = None,
    ) -> TModel:
        """
        Applies weight compression transformations to the model.

        :param model: Model in which the weights will be compressed according to the weight compression description.
        :param graph: The graph associated with the model.
        :param weight_compression_parameters: An iterable of weight compression parameters.
        :param precomputed_compressed_weights: Precomputed scales, zero points, or codebook for weight compression.
        :param lora_correction_algo: An optional algorithm to reduce quantization noise after weight compression by
            using low-rank adapters. This algorithm not only overrides weights with their quantized counterparts but
            also expands the model's execution graph following the Low-Rank Adaptation (LoRA) concept.
        :param compression_format: The format in which the model is saved after weight compression.
        :param advanced_parameters: Describes advanced parameters of compression formats.
        :return: The transformed model.
        """

    @abstractmethod
    def insert_adapters(
        self, wc_params: WeightCompressionParameters, lora_A: Tensor, lora_B: Tensor, int8_lora: bool
    ) -> None:
        r"""
        Expands a model's execution graph following the Low-Rank Adaptation (LoRA) concept.

        It inserts two additional Linear layers with weight matrices of low rank that are executed in parallel to the
        target Linear layer.

        Before insertion:

            ----INPUT
                   \
                   orig.MM--------------------------------OUTPUT

        After insertion:

            ----INPUT ----lora_A.MM----lora_B.MM----\
                  \                                add----OUTPUT
                   orig.MM--------------------------/

        :param wc_params: Parameters for weight compression.
        :param lora_A: weights for the first LoRA matrix.
        :param lora_B: weights for the second LoRA matrix.
        :param int8_lora: indicates whether the LoRA matrices should be compressed to 8-bit.
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

    @abstractmethod
    def mean_statistic_collector(
        self, reduction_axes: tuple[int], subset_size: Optional[int] = None
    ) -> TensorStatisticCollectorBase:
        """
        Return mean statistic collector

        :param reduction_axes: Axes along which to apply mean reduction
        :param subset_size: Number of samples to collect
        """

    @staticmethod
    @abstractmethod
    def get_activation_port_id(node: NNCFNode, graph: NNCFGraph) -> int:
        """
        Returns input port id corresponding to activation input edge for the node.
        Supports only nodes that could have bias value.

        :param node: Node of NNCFGraph with bias value.
        :param graph: NNCFGraph instance with the node.
        :return: target input port id.
        """

    @staticmethod
    @abstractmethod
    def dump_parameters(
        model: TModel, parameters: dict, algo_name: Optional[str] = "quantization", path: Optional[list] = None
    ) -> None:
        """
        Dumps the given parameters into Model's meta section.

        :param model: ov.Model instance.
        :param algo_name: Name of the algorithm to which the parameters refer.
        :param parameters: Incoming dictionary with parameters to save.
        :param path: Optional list of the paths.
        """

    @staticmethod
    @abstractmethod
    def get_filter_fn_for_statistics(activation_port_id: int, algorithm_key: str) -> Callable[[StatisticPoint], bool]:
        """
        Returns backend-specific callable to filter statistic containers according to its statistic point.

        :param activation_port_id: Activation port id for the statistic collection target node.
        :param algorithm_key: Current algorithm key.
        :return: Backend-specific callable to filter statistic containers according to its statistic point.
        """

    @staticmethod
    @abstractmethod
    def get_ignored_patterns() -> GraphPattern:
        """
        Return backend-specific ignored patterns.

        :return: backend-specific ignored patterns.
        """


class AWQAlgoBackend(WeightCompressionAlgoBackend):
    @staticmethod
    def get_awq_patterns() -> dict:
        """
        Returns patterns of nodes in network graph for applying AWQ algorithm.
        """

    @staticmethod
    def scale_insertion_command(source_node, next_nodes, source_node_output_port, scale):
        """
        Returns scale insertion command/transformation for applying AWQ algorithm.
        """


class MixedPrecisionAlgoBackend:
    @staticmethod
    def hawq_statistic_collector(subset_size: Optional[int] = None) -> TensorCollector:
        reducer = RawReducer()
        aggregator = HAWQAggregator(num_samples=subset_size)
        collector = TensorCollector(HessianTensorStatistic)
        collector.register_statistic_branch(HessianTensorStatistic.HESSIAN_INPUT_ACTIVATION_STATS, reducer, aggregator)
        return collector

    @staticmethod
    def mean_variance_statistic_collector(
        reduction_axes: tuple[int], subset_size: Optional[int] = None
    ) -> TensorCollector:
        reducer = MeanVarianceReducer(reduction_axes)
        aggregator = MeanAggregator(num_samples=subset_size)
        collector = TensorCollector(MeanVarianceTensorStatistic)
        collector.register_statistic_branch(MeanVarianceTensorStatistic.MEAN_VARIANCE_STAT, reducer, aggregator)
        return collector

    @staticmethod
    def max_variance_statistic_collector(
        reduction_axes: tuple[int], subset_size: Optional[int] = None
    ) -> TensorCollector:
        reducer = MaxVarianceReducer(reduction_axes)
        aggregator = MeanAggregator(num_samples=subset_size)
        collector = TensorCollector(MaxVarianceTensorStatistic)
        collector.register_statistic_branch(MaxVarianceTensorStatistic.MAX_VARIANCE_STAT, reducer, aggregator)
        return collector

    @staticmethod
    def mean_abs_max_statistic_collector(
        reduction_axes: tuple[int], subset_size: Optional[int] = None
    ) -> TensorCollector:
        reducer = MeanAbsMaxReducer(reduction_axes)
        aggregator = MeanAggregator(num_samples=subset_size)
        collector = TensorCollector(MeanMagnitudeTensorStatistic)
        collector.register_statistic_branch(MeanMagnitudeTensorStatistic.MEAN_MAGNITUDE_STAT, reducer, aggregator)
        return collector
