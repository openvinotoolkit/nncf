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
from typing import Dict, List, Tuple

from nncf.api.compression import TModel
from nncf.common.factory import NNCFGraphFactory
from nncf.common.factory import StatisticsAggregatorFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from nncf.data import Dataset
from nncf.parameters import SensitivityMetric
from nncf.quantization.algorithms.weight_compression.algorithm import WeightCompression
from nncf.quantization.algorithms.weight_compression.algorithm import get_weight_compression_configuration
from nncf.quantization.algorithms.weight_compression.gptq import GPTQ
from nncf.quantization.algorithms.weight_compression.mixed_precision import MIXED_PRECISION_CRITERIA


def register_statistics_for_algorithm(
    aggregator: StatisticsAggregator,
    model: TModel,
    graph: NNCFGraph,
    subset_size: int,
    compression_algo: WeightCompression,
) -> None:
    """
    Registers the statistics required for the given compression algorithm.

    :param aggregator: Aggregator to register statistics.
    :param model: Model being analyzed.
    :param graph: Model's computational graph.
    :param subset_size: Size of dataset subset for statistics.
    :param compression_algo: WeightCompression algorithm instance.
    """
    compression_algo.set_backend_entity(model)

    nodes_to_compress = [
        node
        for node in compression_algo.get_nodes_to_compress(graph)
        if node.metatype in compression_algo._backend_entity.matmul_metatypes
    ]

    input_output_map = compression_algo.get_matmul_input_to_output_nodes_map(nodes_to_compress, graph)

    statistic_points = compression_algo.get_statistic_points(model, graph, input_output_map.keys(), subset_size)
    aggregator.register_statistic_points(statistic_points)


def _register_gptq(
    aggregator: StatisticsAggregator,
    model: TModel,
    graph: NNCFGraph,
    nodes_to_compress: List[NNCFNode],
    subset_size: int,
) -> None:
    """
    Registers statistics for the GPTQ compression algorithm.

    :param aggregator: Aggregator to register statistics.
    :param model: Model being analyzed.
    :param graph: Model's computational graph.
    :param nodes_to_compress: Nodes selected for GPTQ compression.
    :param subset_size: Size of dataset subset for statistics.
    """
    gptq_algo = GPTQ(subset_size=subset_size)
    statistic_points = gptq_algo.get_statistic_points(model, graph, nodes_to_compress)
    aggregator.register_statistic_points(statistic_points)


def _register_mixed_precision(
    aggregator: StatisticsAggregator,
    model: TModel,
    graph: NNCFGraph,
    input_output_map: Dict[Tuple[NNCFNode, int], List[NNCFNode]],
    subset_size: int,
) -> None:
    """
    Registers statistics for mixed precision compression algorithm.

    :param aggregator: Aggregator to register statistics.
    :param model: Model being analyzed.
    :param graph: Model's computational graph.
    :param input_output_map: Map of input to output nodes for matmul operations.
    :param subset_size: Size of dataset subset for statistics.
    """
    sensitivities = [
        SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
        SensitivityMetric.MEAN_ACTIVATION_VARIANCE,
        SensitivityMetric.MAX_ACTIVATION_VARIANCE,
        SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE,
    ]

    for sensitivity in sensitivities:
        criterion_cls = MIXED_PRECISION_CRITERIA.get(sensitivity)
        mixed_prec_algo = criterion_cls(None, None)
        statistic_points = mixed_prec_algo.get_statistic_points(model, graph, input_output_map.keys(), subset_size)
        aggregator.register_statistic_points(statistic_points)


def register_all_statistics(
    aggregator: StatisticsAggregator,
    model: TModel,
    graph: NNCFGraph,
    subset_size: int,
    compression_algo: WeightCompression,
    enable_gptq: bool = False,
    enable_mixed_precision: bool = True,
) -> None:
    """
    Registers all required statistics for the model compression.

    :param aggregator: Aggregator to register statistics.
    :param model: Model being analyzed.
    :param graph: Model's computational graph.
    :param subset_size: Size of dataset subset for statistics.
    :param compression_algo: WeightCompression algorithm instance.
    :param enable_gptq: Whether to enable GPTQ statistics.
    :param enable_mixed_precision: Whether to enable mixed precision statistics.
    """
    compression_algo.set_backend_entity(model)
    nodes_to_compress = compression_algo.get_nodes_to_compress(graph)
    matmul_nodes_to_compress = compression_algo.get_matmul_nodes(nodes_to_compress)

    input_output_map = compression_algo.get_matmul_input_to_output_nodes_map(matmul_nodes_to_compress, graph)

    register_statistics_for_algorithm(aggregator, model, graph, subset_size, compression_algo)

    if enable_gptq:
        _register_gptq(aggregator, model, graph, matmul_nodes_to_compress, subset_size)

    if enable_mixed_precision:
        _register_mixed_precision(aggregator, model, graph, input_output_map, subset_size)


def cache_weight_compression_statistics(
    model: TModel, dataset: Dataset, subset_size: int, statistics_path: str
) -> None:
    """
    Caches compression statistics for a given model and dataset.

    :param model: Model being analyzed.
    :param dataset: Dataset to analyze model statistics.
    :param subset_size: Size of dataset subset for statistics.
    :param statistics_path: Path to save cached statistics.
    """
    config = get_weight_compression_configuration(awq=True, scale_estimation=True, lora_correction=True)
    compression_algo = WeightCompression(**config, subset_size=subset_size)

    graph = NNCFGraphFactory.create(model)
    aggregator = StatisticsAggregatorFactory.create(model, dataset)
    register_all_statistics(aggregator, model, graph, subset_size, compression_algo)
    aggregator.collect_statistics(model, graph)
    aggregator.dump_statistics(statistics_path)
