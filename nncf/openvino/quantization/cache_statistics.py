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

import openvino.runtime as ov

from nncf.common.factory import NNCFGraphFactory
from nncf.common.factory import StatisticsAggregatorFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from nncf.data import Dataset
from nncf.openvino.graph.model_utils import remove_friendly_name_duplicates
from nncf.parameters import SensitivityMetric
from nncf.quantization.algorithms.weight_compression.algorithm import WeightCompression
from nncf.quantization.algorithms.weight_compression.algorithm import get_weight_compression_configuration
from nncf.quantization.algorithms.weight_compression.gptq import GPTQ
from nncf.quantization.algorithms.weight_compression.mixed_precision import MIXED_PRECISION_CRITERIA


def register_statistics_for_algorithm(
    statistics_aggregator: StatisticsAggregator,
    model: ov.Model,
    graph: NNCFGraph,
    subset_size: int,
    compression_algorithm: WeightCompression,
) -> None:
    """Registers the statistics of the provided algorithm."""
    compression_algorithm.set_backend_entity(model)

    nodes_to_compress = [
        node
        for node in compression_algorithm.get_nodes_to_compress(graph)
        if node.metatype in compression_algorithm._backend_entity.matmul_metatypes
    ]

    matmul_input_to_output_nodes_map = compression_algorithm.get_matmul_input_to_output_nodes_map(
        nodes_to_compress, graph
    )

    statistic_points = compression_algorithm.get_statistic_points(
        model, graph, matmul_input_to_output_nodes_map.keys(), subset_size
    )
    statistics_aggregator.register_statistic_points(statistic_points)


def _register_gptq(
    statistics_aggregator: StatisticsAggregator,
    model: ov.Model,
    graph: NNCFGraph,
    matmul_nodes_to_compress: List[NNCFNode],
    subset_size: int,
) -> None:
    """Registers GPTQ compression algorithm statistics."""
    gptq_algo = GPTQ(subset_size=subset_size)
    statistic_points = gptq_algo.get_statistic_points(model, graph, matmul_nodes_to_compress)
    statistics_aggregator.register_statistic_points(statistic_points)


def _register_mixed_precision(
    statistics_aggregator: StatisticsAggregator,
    model: ov.Model,
    graph: NNCFGraph,
    matmul_input_to_output_nodes_map: Dict[Tuple[NNCFNode, int], List[NNCFNode]],
    subset_size: int,
) -> None:
    """Registers mixed precision compression algorithm statistics."""
    sensitivities_to_collect = [
        SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
        SensitivityMetric.MEAN_ACTIVATION_VARIANCE,
        SensitivityMetric.MAX_ACTIVATION_VARIANCE,
        SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE,
    ]

    for sensitivity in sensitivities_to_collect:
        criterion_cls = MIXED_PRECISION_CRITERIA.get(sensitivity)
        mixed_prec_algo = criterion_cls(None, None)
        statistic_points = mixed_prec_algo.get_statistic_points(
            model, graph, matmul_input_to_output_nodes_map.keys(), subset_size
        )
        statistics_aggregator.register_statistic_points(statistic_points)


def register_all_statistics(
    statistics_aggregator: StatisticsAggregator,
    model: ov.Model,
    graph: NNCFGraph,
    subset_size: int,
    compression_algorithm: WeightCompression,
    gptq: bool = False,  # GPTQ does not support statistics caching
    mixed_precision: bool = True,
) -> None:
    """Registers all necessary statistics for compression."""
    compression_algorithm.set_backend_entity(model)

    nodes_to_compress = compression_algorithm.get_nodes_to_compress(graph)
    matmul_nodes_to_compress = [
        node for node in nodes_to_compress if node.metatype in compression_algorithm._backend_entity.matmul_metatypes
    ]

    matmul_input_to_output_nodes_map = compression_algorithm.get_matmul_input_to_output_nodes_map(
        matmul_nodes_to_compress, graph
    )

    register_statistics_for_algorithm(statistics_aggregator, model, graph, subset_size, compression_algorithm)

    if gptq:
        _register_gptq(statistics_aggregator, model, graph, nodes_to_compress, subset_size)

    if mixed_precision:
        _register_mixed_precision(statistics_aggregator, model, graph, matmul_input_to_output_nodes_map, subset_size)


def cache_weight_compression_statistics(
    model: ov.Model, dataset: Dataset, subset_size: int, statistics_dir_path: str
) -> None:
    """Caches compression statistics for the given model and dataset for WeightCompression."""
    model = remove_friendly_name_duplicates(model)
    weight_compression_configuration = get_weight_compression_configuration(
        awq=True, scale_estimation=True, lora_correction=True
    )
    compression_algorithm = WeightCompression(**weight_compression_configuration, subset_size=subset_size)

    graph = NNCFGraphFactory.create(model)
    statistics_aggregator = StatisticsAggregatorFactory.create(model, dataset)
    register_all_statistics(statistics_aggregator, model, graph, subset_size, compression_algorithm)
    statistics_aggregator.collect_statistics(model, graph)
    statistics_aggregator.dump_statistics(statistics_dir_path)
