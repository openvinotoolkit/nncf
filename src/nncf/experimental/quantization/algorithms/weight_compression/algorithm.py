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
        sensitivity_metric: nncf.SensitivityMetric = SensitivityMetric.WEIGHT_QUANTIZATION_ERROR,
        compression_format: nncf.CompressionFormat = nncf.CompressionFormat.DQ,
        advanced_parameters: nncf.AdvancedCompressionParameters = None,
    ) -> torch.fx.GraphModule:
        self._quantizer = quantizer

        wc_config = self._quantizer.get_weight_compression_config()

        self._mode = wc_config.get("mode", None)
        self._awq = awq
        self._gptq = gptq
        self._scale_estimation = scale_estimation
        self._subset_size = subset_size
        self._advanced_parameters = advanced_parameters
        self._lora_correction = lora_correction
        self._ratio = wc_config.get("ratio", 1)
        self._group_size = wc_config.get("group_size", 128)
        self._all_layers = wc_config.get("all_layers", False)
        self._backup_mode = wc_config.get("backup_mode", nncf.BackupMode.INT8_ASYM)
        self._sensitivity_metric = sensitivity_metric
        self._compression_format = compression_format
        self._algo = WeightCompression(
            mode=self._mode,
            ratio=self._ratio,
            group_size=self._group_size,
            ignored_scope=nncf.IgnoredScope(),  # This is already defined in the quantizer object
            all_layers=self._all_layers,
            sensitivity_metric=self._sensitivity_metric,
            awq=self._awq,
            subset_size=self._subset_size,
            scale_estimation=self._scale_estimation,
            gptq=self._gptq,
            lora_correction=self._lora_correction,
            backup_mode=self._backup_mode,
            compression_format=self._compression_format,
            advanced_parameters=self._advanced_parameters,
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
        self._algo.set_backend_entity(model)
        
        all_weight_params, ratio_defining_params, group_size_values, skipped_weight_params = self._quantizer.get_weight_compression_parameters(
            model, graph
        )

        return self._algo.apply_with_parameters(
            model,
            graph,
            dataset,
            statistic_points,
            all_weight_params,
            ratio_defining_params,
            group_size_values,
            skipped_weight_params,
        )

    def get_statistic_points(self, model, graph: NNCFGraph) -> StatisticPointsContainer:
        return self._algo.get_statistic_points(model, graph)
