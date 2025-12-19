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

from typing import Iterable, Optional, TypeVar

from nncf import AdvancedCompressionParameters
from nncf import CompressionFormat
from nncf import CompressWeightsMode
from nncf import Dataset
from nncf import SensitivityMetric
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.logging import nncf_logger
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.experimental.quantization.quantizer import Quantizer
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.weight_compression.algorithm import WeightCompression as OriginalWeightCompression
from nncf.quantization.algorithms.weight_compression.algorithm import get_weight_compression_configuration

TModel = TypeVar("TModel")


class WeightsCompression(Algorithm):
    """
    Post-training Weight Compression algorithm implementation.

    Compresses weights of Linear and Embedding layers to 8-bit integer or
    to 4-bit integer/float depending on mode, ratio and group size.
    """

    def __init__(
        self,
        quantizer: Quantizer,
        ratio: float,
        subset_size: int,
        awq: bool,
        scale_estimation: bool,
        gptq: bool,
        lora_correction: bool,
        sensitivity_metric: SensitivityMetric,
        compression_format: CompressionFormat,
        advanced_parameters: Optional[AdvancedCompressionParameters] = None,
    ) -> TModel:
        """
        :param quantizer: Quantizer to use in WeightCompression algorithm.
        :param ratio: the ratio between primary and backup precisions (e.g. 0.9 means 90% of layers specified as
            `ratio_defining_params` by the quantizer are quantized to INT4
        :param subset_size: Number of data samples to calculate activation statistics used for assigning different
            quantization precision.
        :param awq: determines whether to use or not modified AWQ algorithm.
        :param scale_estimation: determines whether to use or not scale estimation for 4 bit layers.
        :param gptq: determines whether to use or not GPTQ algorithm.
        :param lora_correction: determines whether to use or not LoRA Correction algorithm.
        :param sensitivity_metric: The sensitivity metric for assigning quantization precision to layers. In order to
            preserve the accuracy of the model, the more sensitive layers receives a higher precision.
        :param compression_format: Describes the format in which the model is saved after weight compression.
        :param advanced_parameters: advanced parameters for algorithms in compression pipeline.
        """
        self._quantizer = quantizer
        quantizer_wc_config = quantizer.get_weight_compression_config()

        mode = quantizer_wc_config.get("mode", CompressWeightsMode.INT8_ASYM)
        weight_compression_configuration = get_weight_compression_configuration(
            mode=CompressWeightsMode(mode),
            dataset=None,  # Dataset here only affects sensitivity metric. Sensitivity metric arg is guaranteed.
            ratio=ratio,
            group_size=quantizer_wc_config.get("group_size", None),
            ignored_scope=None,
            all_layers=quantizer_wc_config.get("all_layers", None),
            sensitivity_metric=sensitivity_metric,
            awq=awq,
            scale_estimation=scale_estimation,
            gptq=gptq,
            lora_correction=lora_correction,
            backup_mode=quantizer_wc_config.get("backup_mode", None),
            advanced_parameters=advanced_parameters,
        )

        self._algo = OriginalWeightCompression(
            **weight_compression_configuration,
            compression_format=compression_format,
            subset_size=subset_size,
        )

    def available_backends(self) -> list[BackendType]:
        return [BackendType.TORCH_FX]

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        self._algo.set_backend_entity(model)

        all_weight_params, ratio_defining_params, skipped_weight_params = (
            self._quantizer.get_weight_compression_parameters(model, graph)
        )
        # Collect statistics for the weights compression
        statistics, statistic_points = self._algo.collect_statistics_and_statistic_points(
            model, graph, statistic_points, dataset, ratio_defining_params, all_weight_params
        )
        # Apply Mixed precision algorithm to ratio defining parameters
        self._algo.apply_mixed_precision(ratio_defining_params, model, graph, statistic_points)
        self._algo.validate_group_size(ratio_defining_params)

        # Print statistics
        nncf_logger.info(
            self._algo.get_bitwidth_distribution_str(all_weight_params, ratio_defining_params, skipped_weight_params)
        )

        # Filter all_weight_params by excluding nodes that should remain in their original floating-point precision
        all_weight_params = [w_params for w_params in all_weight_params if w_params.compression_config is not None]
        return self._algo.apply_with_parameters(
            model,
            graph,
            dataset,
            statistics,
            all_weight_params,
        )

    def get_statistic_points(
        self,
        model: TModel,
        graph: NNCFGraph,
        nodes_and_port_ids: Iterable[tuple[NNCFNode, int]],
    ) -> StatisticPointsContainer:
        """
        Returns statistic points, for which StatisticsCollector should collect statistics.

        :param model: Model for statistics collection.
        :param graph: Model graph.
        :param nodes_and_port_ids: Nodes and port ids for which statistics should be collected.
        :return: Statistic points, for which StatisticsCollector should collect statistics.
        """
        return self._algo.get_statistic_points(model, graph, nodes_and_port_ids)
