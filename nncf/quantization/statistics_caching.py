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
from pathlib import Path
from typing import Dict, List, Tuple

from nncf.api.compression import TModel
from nncf.common.factory import StatisticsAggregatorFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from nncf.data import Dataset
from nncf.parameters import SensitivityMetric
from nncf.quantization.algorithms.weight_compression.algorithm import WeightCompression
from nncf.quantization.algorithms.weight_compression.algorithm import get_weight_compression_configuration
from nncf.quantization.algorithms.weight_compression.mixed_precision import MIXED_PRECISION_CRITERIA


def register_statistics_for_algorithm(
    aggregator: StatisticsAggregator,
    model: TModel,
    graph: NNCFGraph,
    compression_algo: WeightCompression,
    matmul_input_to_output_nodes_map: Dict[Tuple[NNCFNode, int], List[NNCFNode]],
) -> None:
    """
    Registers the statistics required for the given compression algorithm.

    :param aggregator: Aggregator to register statistics.
    :param model: Model being analyzed.
    :param graph: Model's computational graph.
    :param compression_algo: WeightCompression algorithm instance.
    :param matmul_input_to_output_nodes_map: A dictionary mapping from a tuple of (activation node, port ID)
    to a list of MatMul nodes that accept the activation as input.
    """
    statistic_points = compression_algo.get_statistic_points(model, graph, matmul_input_to_output_nodes_map.keys())
    aggregator.register_statistic_points(statistic_points)


def _register_mixed_precision(
    aggregator: StatisticsAggregator,
    model: TModel,
    graph: NNCFGraph,
    matmul_input_to_output_nodes_map: Dict[Tuple[NNCFNode, int], List[NNCFNode]],
    subset_size: int,
) -> None:
    """
    Registers statistics for mixed precision compression algorithm.

    :param aggregator: Aggregator to register statistics.
    :param model: Model being analyzed.
    :param graph: Model's computational graph.
    :param matmul_input_to_output_nodes_map: A dictionary mapping from a tuple of (activation node, port ID)
    to a list of MatMul nodes that accept the activation as input.
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
        mixed_prec_algo = criterion_cls(None, None, subset_size)
        statistic_points = mixed_prec_algo.get_statistic_points(model, graph, matmul_input_to_output_nodes_map.keys())
        aggregator.register_statistic_points(statistic_points)


def register_all_statistics(
    aggregator: StatisticsAggregator,
    model: TModel,
    graph: NNCFGraph,
    subset_size: int,
    compression_algo: WeightCompression,
    enable_mixed_precision: bool = True,
) -> None:
    """
    Registers all required statistics for the model compression.

    :param aggregator: Aggregator to register statistics.
    :param model: Model being analyzed.
    :param graph: Model's computational graph.
    :param compression_algo: WeightCompression algorithm instance.
    :param enable_mixed_precision: Whether to enable mixed precision statistics.
    """
    _, matmul_input_to_output_nodes_map = compression_algo.get_compression_nodes_info(graph)

    register_statistics_for_algorithm(aggregator, model, graph, compression_algo, matmul_input_to_output_nodes_map)

    if enable_mixed_precision:
        _register_mixed_precision(aggregator, model, graph, matmul_input_to_output_nodes_map, subset_size)


def cache_weight_compression_statistics(
    model: TModel, graph: NNCFGraph, dataset: Dataset, subset_size: int, statistics_path: Path
) -> None:
    """
    Caches compression statistics for a given model and dataset.

    :param model: Model being analyzed.
    :param graph: Model's computational graph.
    :param dataset: Dataset to analyze model statistics.
    :param subset_size: Size of dataset subset for statistics.
    :param statistics_path: Path to save cached statistics.
    """
    config = get_weight_compression_configuration(awq=True, scale_estimation=True, lora_correction=True)
    compression_algo = WeightCompression(**config, subset_size=subset_size)
    compression_algo.set_backend_entity(model)
    aggregator = StatisticsAggregatorFactory.create(model, dataset)
    register_all_statistics(aggregator, model, graph, subset_size, compression_algo)
    aggregator.collect_statistics(model, graph)
    aggregator.dump_statistics(statistics_path)
