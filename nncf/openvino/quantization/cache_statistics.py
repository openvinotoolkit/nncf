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
from nncf.common.factory import NNCFGraphFactory
from nncf.common.factory import StatisticsAggregatorFactory
from nncf.openvino.graph.model_utils import remove_friendly_name_duplicates
from nncf.parameters import SensitivityMetric
from nncf.quantization.algorithms.weight_compression.algorithm import WeightCompression
from nncf.quantization.algorithms.weight_compression.gptq import GPTQ
from nncf.quantization.algorithms.weight_compression.mixed_precision import MIXED_PRECISION_CRITERIA


def _register_main_algorithm(
    statistics_aggregator, model, graph, subset_size, compression_algorithm, matmul_input_to_output_nodes_map
):
    statistic_points = compression_algorithm.get_statistic_points(
        model, graph, matmul_input_to_output_nodes_map.keys(), subset_size
    )
    statistics_aggregator.register_statistic_points(statistic_points)


def _register_gptq(statistics_aggregator, model, graph, matmul_nodes_to_compress, subset_size):
    gptq_algo = GPTQ(subset_size=subset_size)
    statistic_points = gptq_algo.get_statistic_points(model, graph, matmul_nodes_to_compress)
    statistics_aggregator.register_statistic_points(statistic_points)


def _register_mixed_precision(statistics_aggregator, model, graph, matmul_input_to_output_nodes_map, subset_size):
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


def register_all_statistics(statistics_aggregator, model, graph, subset_size, compression_algorithm):
    compression_algorithm._set_backend_entity(model)

    nodes_to_compress = compression_algorithm._get_nodes_to_compress(graph)
    matmul_nodes_to_compress = [
        node for node in nodes_to_compress if node.metatype in compression_algorithm._backend_entity.matmul_metatypes
    ]
    matmul_input_to_output_nodes_map = compression_algorithm._get_matmul_input_to_output_nodes_map(
        matmul_nodes_to_compress, graph
    )
    _register_main_algorithm(
        statistics_aggregator, model, graph, subset_size, compression_algorithm, matmul_input_to_output_nodes_map
    )
    _register_gptq(statistics_aggregator, model, graph, matmul_nodes_to_compress, subset_size)
    _register_mixed_precision(statistics_aggregator, model, graph, matmul_input_to_output_nodes_map, subset_size)


def register_statistics(statistics_aggregator, model, graph, subset_size, compression_algorithm):
    compression_algorithm._set_backend_entity(model)

    nodes_to_compress = compression_algorithm._get_nodes_to_compress(graph)
    matmul_nodes_to_compress = [
        node for node in nodes_to_compress if node.metatype in compression_algorithm._backend_entity.matmul_metatypes
    ]
    matmul_input_to_output_nodes_map = compression_algorithm._get_matmul_input_to_output_nodes_map(
        matmul_nodes_to_compress, graph
    )
    _register_main_algorithm(
        statistics_aggregator, model, graph, subset_size, compression_algorithm, matmul_input_to_output_nodes_map
    )


def cache_statistics(
    model,
    dataset,
    mode,
    ratio,
    group_size,
    ignored_scope,
    all_layers,
    sensitivity_metric,
    awq,
    subset_size,
    scale_estimation,
    gptq,
    lora_correction,
    backup_mode,
    advanced_parameters,
):
    model = remove_friendly_name_duplicates(model)
    compression_algorithm = WeightCompression(
        mode,
        ratio,
        group_size,
        ignored_scope,
        all_layers,
        sensitivity_metric,
        awq,
        subset_size,
        scale_estimation,
        gptq,
        lora_correction,
        backup_mode,
        advanced_parameters,
    )

    graph = NNCFGraphFactory.create(model)
    statistics_aggregator = StatisticsAggregatorFactory.create(model, dataset)

    register_all_statistics(statistics_aggregator, model, graph, subset_size, compression_algorithm)

    statistics_aggregator.collect_statistics(model, graph)
    statistics_aggregator.dump_statistics(advanced_parameters.statistics_file_path)
