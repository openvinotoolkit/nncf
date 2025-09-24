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

import torch

import nncf
from nncf import SensitivityMetric
from nncf.common.graph.graph import NNCFGraph
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.weight_compression.algorithm import WeightCompression


class WeightsCompressionPT2E(Algorithm):
    def __init__(
        self,
        quantizer,
        subset_size: int = 128,
        awq: bool = False,
        scale_estimation: bool = False,
        gptq: bool = False,
        lora_correction: bool = False,
        sensitivity_metric: nncf.SensitivityMetric = None,
        compression_format: nncf.CompressionFormat = nncf.CompressionFormat.DQ,
        advanced_parameters: nncf.AdvancedCompressionParameters = None,
    ) -> torch.fx.GraphModule:
        self._quantizer = quantizer

        wc_config = quantizer._weight_compression_configuration

        mode = wc_config.get("mode", None)
        ratio = wc_config.get("ratio", 1)
        group_size = wc_config.get("group_size", 128)
        all_layers = wc_config.get("all_layers", False)
        backup_mode = wc_config.get("backup_mode", nncf.BackupMode.INT8_ASYM)
        self._sensitivity_metric = sensitivity_metric

        self._algo = WeightCompression(
            mode=mode,
            ratio=ratio,
            group_size=group_size,
            ignored_scope=nncf.IgnoredScope(),  # only compress "nodes_to_compress"
            all_layers=all_layers,
            sensitivity_metric=self._sensitivity_metric or SensitivityMetric.WEIGHT_QUANTIZATION_ERROR,
            awq=awq,
            subset_size=subset_size,
            scale_estimation=scale_estimation,
            gptq=gptq,
            lora_correction=lora_correction,
            backup_mode=backup_mode,
            compression_format=compression_format,
            advanced_parameters=advanced_parameters,
        )

    def available_backends(self) -> list[BackendType]:
        return self._algo.available_backends()

    def apply(
        self,
        model: torch.fx.GraphModule,
        graph: NNCFGraph,
        statistic_points=None,
        dataset=None,
    ):
        self._algo.set_backend_entity(model)  # Set algo backend

        if self._sensitivity_metric is None:
            # Default case. It means that it is not defined by the user in the API
            # Hence, the annotation(Quantization parameters for all layers) from the quantizer will be used.
            all_weight_params = self._quantizer.get_weight_compression_setup(
                model, graph
            )  # Get weight compression params FROM QUANTIZER
            statistics, statistic_points = self._algo.collect_weight_compression_statistics(
                model, graph, dataset, all_weight_params, statistic_points
            )
        else:
            # Data Aware mixed precision is used. In this case, only nodes_to_compress is obtained from the quantizer
            nodes_to_compress = self._quantizer.get_nodes_to_compress(
                model, graph
            )  # Get nodes to compress FROM QUANTIZER
            all_weight_params, statistics = self._algo.get_weight_compression_parameters(
                model, graph, nodes_to_compress, statistic_points, dataset
            )

        transformed_model = self._algo.apply_wc_algos(
            model, graph, all_weight_params, statistics, dataset
        )  # Apply the wc algos FROM ALGO
        return transformed_model

    def get_statistic_points(self, model, graph: NNCFGraph) -> StatisticPointsContainer:
        return self._algo.get_statistic_points(model, graph)
