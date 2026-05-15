# Copyright (c) 2026 Intel Corporation
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

from nncf.common.factory import build_graph
from nncf.data import Dataset
from nncf.parameters import BackupMode
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.parameters import SensitivityMetric
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.algorithms.weight_compression.algorithm import WeightCompression
from nncf.scopes import IgnoredScope
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import GraphModelWrapper

DEFAULT_RANGE_TYPE = "mean_min_max"


def compress_weights_impl(
    model: GraphModelWrapper | torch.nn.Module,
    dataset: Dataset | None,
    mode: CompressWeightsMode,
    ratio: float,
    group_size: int,
    ignored_scope: IgnoredScope,
    all_layers: bool,
    sensitivity_metric: SensitivityMetric,
    awq: bool,
    subset_size: int,
    scale_estimation: bool,
    gptq: bool,
    lora_correction: bool,
    backup_mode: BackupMode,
    compression_format: CompressionFormat,
    advanced_parameters: AdvancedCompressionParameters | None = None,
) -> torch.nn.Module:
    """
    Implementation of the `compress_weights()` method for the PyTorch backend.
    """
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
        compression_format,
        advanced_parameters,
    )
    graph = build_graph(model)

    compressed_model = compression_algorithm.apply(model, graph, dataset=dataset)
    if isinstance(compressed_model, GraphModelWrapper):
        compressed_model = compressed_model.model
    return compressed_model
