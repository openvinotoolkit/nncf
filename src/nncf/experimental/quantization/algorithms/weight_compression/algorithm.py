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

from typing import Iterable, Optional

import torch

from nncf import AdvancedCompressionParameters
from nncf import BackupMode
from nncf import CompressionFormat
from nncf import CompressWeightsMode
from nncf import Dataset
from nncf import IgnoredScope
from nncf import SensitivityMetric
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.logging import nncf_logger
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.experimental.quantization.quantizer import Quantizer
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.weight_compression.algorithm import WeightCompression as OriginalWeightCompression


class WeightsCompression(Algorithm):
    """
    Post-training Weight Compression algorithm implementation.

    Compresses weights of Linear and Embedding layers to 8-bit integer or
    to 4-bit integer/float depending on mode, ratio and group size.
    """

    def __init__(
        self,
        mode: CompressWeightsMode,
        quantizer: Quantizer,
        ratio: float,
        group_size: int,
        ignored_scope: IgnoredScope,
        all_layers: bool,
        subset_size: int,
        awq: bool,
        scale_estimation: bool,
        gptq: bool,
        lora_correction: bool,
        backup_mode: BackupMode,
        sensitivity_metric: SensitivityMetric,
        compression_format: CompressionFormat,
        advanced_parameters: AdvancedCompressionParameters,
    ) -> torch.fx.GraphModule:
        """
        :param mode: Defines a mode for weight compression.
            INT8_SYM stands for 8-bit integer symmetric quantization of all weights.
                Weights are quantized symmetrically without zero point.
            INT8_ASYM is the same as INT8_SYM mode, but weights are quantized to a primary precision asymmetrically
                with a typical non-fixed zero point.
            INT4_SYM stands for a mixed-precision weights quantization with 4-bit integer as a primary precision.
                Weights are quantized to a primary precision symmetrically without zero point.
                All embeddings and the last layer are always compressed to a backup_mode, which is INT8_ASYM,
                by default. All others are quantized whether to 4-bit integer or to a backup_mode depending on
                criteria and the given ratio.
            INT4_ASYM is the same as INT4_SYM mode, but weights are quantized to a primary precision asymmetrically
                with a typical non-fixed zero point.
        :param quantizer: Quantizer to use in WeightCompression algorithm.
        :param ratio: the ratio between primary and backup precisions (e.g. 0.9 means 90% of layers quantized to INT4
            and the rest to backup_mode).
        :param group_size: number of weights (e.g. 128) in the channel dimension
            that share quantization parameters (scale). The value -1 means no grouping.
        :param ignored_scope: An ignored scope that defined the list of model control
            flow graph nodes to be ignored during quantization.
        :param all_layers: Indicates whether embeddings and last MatMul layers should be compressed to a primary
            precision. By default, the backup precision is assigned for the embeddings and last MatMul layers.
        :param subset_size: Number of data samples to calculate activation statistics used for assigning different
            quantization precision.
        :param awq: determines whether to use or not modified AWQ algorithm.
        :param scale_estimation: determines whether to use or not scale estimation for 4 bit layers.
        :param gptq: determines whether to use or not GPTQ algorithm.
        :param lora_correction: determines whether to use or not LoRA Correction algorithm.
        :param backup_mode: Defines a backup mode for mixed-precision weight compression.
            NONE stands for original floating-point precision of the model weights.
                In this mode, weights are retained in their original precision without any quantization.
            INT8_SYM stands for 8-bit integer symmetric quantization without zero point.
            INT8_ASYM stands for 8-bit integer asymmetric quantization with a typical non-fixed zero point.
        :param sensitivity_metric: The sensitivity metric for assigning quantization precision to layers. In order to
            preserve the accuracy of the model, the more sensitive layers receives a higher precision.
        :param compression_format: Describes the format in which the model is saved after weight compression.
        :param advanced_parameters: advanced parameters for algorithms in compression pipeline.
        """
        self._quantizer = quantizer

        self._mode = mode
        self._awq = awq
        self._gptq = gptq
        self._scale_estimation = scale_estimation
        self._subset_size = subset_size
        self._advanced_parameters = advanced_parameters
        self._lora_correction = lora_correction
        self._ratio = ratio
        self._group_size = group_size
        self._all_layers = all_layers
        self._backup_mode = backup_mode
        self._sensitivity_metric = sensitivity_metric
        self._compression_format = compression_format

        self._algo = OriginalWeightCompression(
            mode=self._mode,
            ratio=self._ratio,
            group_size=self._group_size,
            ignored_scope=ignored_scope,
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
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> torch.fx.GraphModule:
        self._algo.set_backend_entity(model)

        all_weight_params, ratio_defining_params, skipped_weight_params = (
            self._quantizer.get_weight_compression_parameters(model, graph)
        )
        # Collect statistics for the weights compression
        statistics, statistic_points = self._algo._collect_statistics_and_statistic_points(
            model, graph, statistic_points, dataset, ratio_defining_params, all_weight_params
        )
        # Apply Mixed precision algorithm to ratio defining parameters
        self._algo._apply_mixed_precision(ratio_defining_params, model, graph, statistic_points)
        self._algo._validate_group_size(ratio_defining_params)

        # Print statistics
        nncf_logger.info(
            self._algo._get_bitwidth_distribution_str(all_weight_params, ratio_defining_params, skipped_weight_params)
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
        model: torch.fx.GraphModule,
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
