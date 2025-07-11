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
import copy
import operator
from collections import OrderedDict
from collections import defaultdict
from functools import reduce
from typing import Any, Iterable, Optional, TypeVar

import nncf
from nncf import Dataset
from nncf.common.factory import StatisticsAggregatorFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.graph import get_node_names_matching_graph_pattern
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.logging import nncf_logger
from nncf.common.logging.track_progress import track
from nncf.common.scopes import should_consider_scope
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.common.utils.helpers import create_table
from nncf.experimental.common.tensor_statistics.statistics import WCTensorStatistic
from nncf.parameters import BackupMode
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.parameters import SensitivityMetric
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.advanced_parameters import convert_to_dict_recursively
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.weight_compression.awq import AWQ
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.constants import CB4_QUANTILES
from nncf.quantization.algorithms.weight_compression.gptq import GPTQ
from nncf.quantization.algorithms.weight_compression.lora_correction import LoraCorrectionAlgorithm
from nncf.quantization.algorithms.weight_compression.mixed_precision import MIXED_PRECISION_CRITERIA
from nncf.quantization.algorithms.weight_compression.scale_estimation import ScaleEstimation
from nncf.quantization.algorithms.weight_compression.weight_lowering import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.weight_lowering import get_reduction_channel_size
from nncf.scopes import IgnoredScope
from nncf.scopes import get_ignored_node_names_from_ignored_scope
from nncf.tensor import Tensor
from nncf.tensor import functions as fns
from nncf.tensor.definitions import TensorDataType

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")

INT8_MODES = [CompressWeightsMode.INT8_ASYM, CompressWeightsMode.INT8_SYM]
NON_INT8_MODES = [
    CompressWeightsMode.INT4_SYM,
    CompressWeightsMode.INT4_ASYM,
    CompressWeightsMode.NF4,
    CompressWeightsMode.E2M1,
]
SUPPORTED_DATA_TYPES = [
    TensorDataType.float16,
    TensorDataType.bfloat16,
    TensorDataType.float32,
    TensorDataType.float64,
]


def get_weight_compression_configuration(
    mode: CompressWeightsMode = CompressWeightsMode.INT8_ASYM,
    dataset: Optional[Dataset] = None,
    ratio: Optional[float] = None,
    group_size: Optional[int] = None,
    all_layers: Optional[bool] = None,
    awq: Optional[bool] = None,
    scale_estimation: Optional[bool] = None,
    gptq: Optional[bool] = None,
    lora_correction: Optional[bool] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    sensitivity_metric: Optional[SensitivityMetric] = None,
    backup_mode: Optional[BackupMode] = None,
    advanced_parameters: Optional[AdvancedCompressionParameters] = None,
) -> dict[str, Any]:
    """
    Generates a configuration dictionary for weight compression based on the provided parameters.
    """
    group_size = (
        -1
        if group_size is None and mode in INT8_MODES
        else 128
        if group_size is None and mode in NON_INT8_MODES
        else group_size
    )

    return {
        "mode": mode,
        "ratio": ratio or 1,
        "group_size": group_size,
        "all_layers": all_layers or False,
        "awq": awq or False,
        "scale_estimation": scale_estimation or False,
        "gptq": gptq or False,
        "lora_correction": lora_correction or False,
        "ignored_scope": ignored_scope or IgnoredScope(),
        "sensitivity_metric": (
            (
                SensitivityMetric.WEIGHT_QUANTIZATION_ERROR
                if dataset is None
                else SensitivityMetric.MAX_ACTIVATION_VARIANCE
            )
            if sensitivity_metric is None
            else sensitivity_metric
        ),
        "backup_mode": backup_mode or BackupMode.INT8_ASYM,
        "advanced_parameters": advanced_parameters or AdvancedCompressionParameters(),
    }


def check_user_compression_configuration(
    mode: CompressWeightsMode,
    subset_size: int,
    dataset: Optional[Dataset],
    ratio: Optional[float],
    group_size: Optional[int],
    all_layers: Optional[bool],
    awq: Optional[bool],
    scale_estimation: Optional[bool],
    gptq: Optional[bool],
    lora_correction: Optional[bool],
    ignored_scope: Optional[IgnoredScope],
    sensitivity_metric: Optional[SensitivityMetric],
    backup_mode: Optional[BackupMode],
    compression_format: Optional[CompressionFormat],
    advanced_parameters: Optional[AdvancedCompressionParameters],
) -> None:
    """
    Validates the user's weight compression configuration for correctness.
    """
    if mode in INT8_MODES:
        if (ratio and ratio != 1) or (group_size and group_size != -1):
            msg = (
                "INT8 modes require per-channel quantization of all layers in 8 bit. "
                "Default values of `ratio` (1) and `group_size` (-1) cannot be overridden."
            )
            raise nncf.ParameterNotSupportedError(msg)

        if advanced_parameters and advanced_parameters.statistics_path:
            msg = "INT8 modes do not support the `statistics_path` option in `AdvancedCompressionParameters`."
            raise nncf.ParameterNotSupportedError(msg)

        unsupported_options = {
            "all_layers": all_layers,
            "sensitivity_metric": sensitivity_metric,
            "dataset": dataset,
            "awq": awq,
            "scale_estimation": scale_estimation,
            "gptq": gptq,
            "lora_correction": lora_correction,
            "backup_mode": backup_mode,
        }
        unsupported_for_int8 = [name for name, value in unsupported_options.items() if value is not None]
        if unsupported_for_int8:
            msg = f"INT8 modes do not support {', '.join(unsupported_for_int8)} option(s). Set them to None."
            raise nncf.ParameterNotSupportedError(msg)

    if ratio is not None and not (0 <= ratio <= 1):
        msg = f"The ratio should be between 0 and 1, but ratio={ratio} is specified."
        raise nncf.ValidationError(msg)

    values_to_check = [subset_size]
    ranks = []
    if advanced_parameters:
        values_to_check.extend(
            [
                advanced_parameters.awq_params.subset_size,
                advanced_parameters.scale_estimation_params.subset_size,
                advanced_parameters.gptq_params.subset_size,
                advanced_parameters.lora_correction_params.subset_size,
            ]
        )
        ranks = [advanced_parameters.lora_adapter_rank, advanced_parameters.lora_correction_params.adapter_rank]

        codebook = advanced_parameters.codebook
        if codebook is not None:
            # OpenVINO Tensor is not support functions to validate codebook
            np_codebook = Tensor(codebook).as_numpy_tensor()
            msg = None
            if np_codebook.ndim != 1:
                msg = "The codebook must be a 1D array, but a multi-dimensional array is given."
            elif np_codebook.size < 2:
                msg = (
                    "The codebook must contain at least two unique elements,"
                    "but a single-element or empty array is given."
                )
            elif fns.any(np_codebook[:-1] >= np_codebook[1:]):
                msg = "The codebook must be a sorted 1D array with unique elements, but an unsorted array is given."
            if msg:
                raise nncf.ValidationError(msg)

    for size in values_to_check:
        if size <= 0:
            msg = f"The subset_size value should be positive, but subset_size={size} is given."
            raise nncf.ValidationError(msg)

    for rank in ranks:
        if rank <= 0:
            msg = f"The lora adapter rank should be positive, but rank={rank} is given."
            raise nncf.ValidationError(msg)

    if (
        ratio
        and dataset is None
        and sensitivity_metric is not None
        and sensitivity_metric != SensitivityMetric.WEIGHT_QUANTIZATION_ERROR
    ):
        msg = f"Mixed precision selection with sensitivity metric={sensitivity_metric.value} \
            requires a dataset, but it's not provided."
        raise nncf.ValidationError(msg)

    if lora_correction and compression_format in [
        CompressionFormat.FQ,
        CompressionFormat.FQ_LORA,
        CompressionFormat.FQ_LORA_NLS,
    ]:
        msg = "LoRA Correction algorithm is not compatible with FQ, FQ_LORA and FQ_LORA_NLS compression formats."
        raise nncf.ValidationError(msg)

    if mode == CompressWeightsMode.CODEBOOK and (advanced_parameters is None or advanced_parameters.codebook is None):
        msg = "Codebook compression mode requires codebook parameters to be specified in advanced_parameters."
        raise nncf.ValidationError(msg)


class WeightCompression(Algorithm):
    """
    Post-training Weight Compression algorithm implementation.

    Compresses weights of Linear and Embedding layers to 8-bit integer or
    to 4-bit integer/float depending on mode, ratio and group size.
    """

    def __init__(
        self,
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
        backup_mode: BackupMode = BackupMode.INT8_ASYM,
        compression_format: CompressionFormat = CompressionFormat.DQ,
        advanced_parameters: Optional[AdvancedCompressionParameters] = None,
    ):
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
            NF4 is the same as INT4_SYM mode, but primary precision is NF4 data type without zero point.
            E2M1 is the same as INT4_SYM mode, but primary precision is E2M1 data type without zero point.
        :param ratio: the ratio between primary and backup precisions (e.g. 0.9 means 90% of layers quantized to NF4
            and the rest to backup_mode).
        :param group_size: number of weights (e.g. 128) in the channel dimension
            that share quantization parameters (scale). The value -1 means no grouping.
        :param ignored_scope: An ignored scope that defined the list of model control
            flow graph nodes to be ignored during quantization.
        :param all_layers: Indicates whether embeddings and last MatMul layers should be compressed to a primary
            precision. By default, the backup precision is assigned for the embeddings and last MatMul layers.
        :param sensitivity_metric: The sensitivity metric for assigning quantization precision to layers. In order to
            preserve the accuracy of the model, the more sensitive layers receives a higher precision.
        :param awq: determines whether to use or not modified AWQ algorithm.
        :param subset_size: Number of data samples to calculate activation statistics used for assigning different
            quantization precision.
        :param scale_estimation: determines whether to use or not scale estimation for 4 bit layers.
        :param gptq: determines whether to use or not GPTQ algorithm.
        :param lora_correction: determines whether to use or not LoRA Correction algorithm.
        :param backup_mode: Defines a backup mode for mixed-precision weight compression.
            NONE stands for original floating-point precision of the model weights.
                In this mode, weights are retained in their original precision without any quantization.
            INT8_SYM stands for 8-bit integer symmetric quantization without zero point.
            INT8_ASYM stands for 8-bit integer asymmetric quantization with a typical non-fixed zero point.
        :param compression_format: Describes the format in which the model is saved after weight compression.
        :param advanced_parameters: advanced parameters for algorithms in compression pipeline.
        """
        super().__init__()
        self._mode = mode
        self._group_size = group_size
        self._ratio = ratio
        self._ignored_scope = ignored_scope
        self._backend_entity = None
        self._algorithm_key = f"CW_{hash(self)}"
        self._statistics = {}
        self._all_layers = all_layers
        self._sensitivity_metric = sensitivity_metric
        self._awq = awq
        self._subset_size = subset_size
        self._scale_estimation = scale_estimation
        self._gptq = gptq
        self._lora_correction = lora_correction
        self._backup_mode = backup_mode
        self._compression_format = compression_format
        self._advanced_parameters = (
            advanced_parameters if advanced_parameters is not None else AdvancedCompressionParameters()
        )

        criterion_cls = MIXED_PRECISION_CRITERIA.get(self._sensitivity_metric)
        self._mixed_precision_algo = criterion_cls(self._ratio, self._subset_size)
        self._statistics_path = self._advanced_parameters.statistics_path

        self._enable_flexible_group_size = self._advanced_parameters.enable_flexible_group_size
        self._min_flexible_group_size = self._advanced_parameters.min_flexible_group_size

        if self._awq:
            awq_params = self._advanced_parameters.awq_params
            self.awq_algo = AWQ(
                awq_params.subset_size,
                awq_params.percent_to_apply,
                awq_params.alpha_min,
                awq_params.alpha_max,
                awq_params.steps,
                awq_params.prefer_data_aware_scaling,
            )
        if self._gptq:
            gptq_params = self._advanced_parameters.gptq_params
            self._gptq_algo = GPTQ(
                damp_percent=gptq_params.damp_percent,
                block_size=gptq_params.block_size,
                subset_size=gptq_params.subset_size,
                scale_estimation=self._scale_estimation,
            )
        if self._scale_estimation:
            scale_estimation_params = self._advanced_parameters.scale_estimation_params
            self._scale_estimation_algo = ScaleEstimation(
                scale_estimation_params.subset_size,
                scale_estimation_params.initial_steps,
                scale_estimation_params.scale_steps,
                scale_estimation_params.weight_penalty,
            )

        self._data_aware_mixed_precision = (
            self._sensitivity_metric != SensitivityMetric.WEIGHT_QUANTIZATION_ERROR and self._ratio != 1.0
        )
        self._data_aware_compression = (
            (self._awq and self._advanced_parameters.awq_params.prefer_data_aware_scaling)
            or self._scale_estimation
            or self._lora_correction
            or self._gptq
        )

    @property
    def available_backends(self) -> list[BackendType]:
        return [BackendType.OPENVINO, BackendType.TORCH, BackendType.TORCH_FX]

    def set_ignored_scope(self, ignored_scope: IgnoredScope) -> None:
        """
        Set target ignored scope for the Weight Compression algorithm.

        :param ignored_scope: The ignored scope to set to the Weight Compression algorithm.
        """
        self._ignored_scope = ignored_scope

    def set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.weight_compression.openvino_backend import OVWeightCompressionAlgoBackend

            self._backend_entity = OVWeightCompressionAlgoBackend(model)
        elif model_backend == BackendType.TORCH:
            from nncf.quantization.algorithms.weight_compression.torch_backend import PTWeightCompressionAlgoBackend

            self._backend_entity = PTWeightCompressionAlgoBackend()
        elif model_backend == BackendType.TORCH_FX:
            from nncf.quantization.algorithms.weight_compression.torch_fx_backend import FXWeightCompressionAlgoBackend

            self._backend_entity = FXWeightCompressionAlgoBackend()
        elif model_backend == BackendType.ONNX:
            from nncf.quantization.algorithms.weight_compression.onnx_backend import ONNXWeightCompressionAlgoBackend

            self._backend_entity = ONNXWeightCompressionAlgoBackend(model)
        else:
            msg = f"Cannot return backend-specific entity because {model_backend.value} is not supported!"
            raise nncf.UnsupportedBackendError(msg)

    def get_nodes_to_compress(self, nncf_graph: NNCFGraph) -> list[NNCFNode]:
        """
        Collects nodes in the model's graph corresponding to the layers for weight compression.

        :param nncf_graph: NNCFGraph instance.
        :return: List with the data for each layer.
        """
        weighted_metatypes = (
            self._backend_entity.matmul_metatypes
            + self._backend_entity.embedding_metatypes
            + self._backend_entity.convolution_metatypes
        )

        ordered_nodes_to_compress = []
        ignored_names = get_ignored_node_names_from_ignored_scope(
            self._ignored_scope, nncf_graph, strict=self._ignored_scope.validate
        )

        autogenerated_ignored_names = get_node_names_matching_graph_pattern(
            nncf_graph, self._backend_entity.get_ignored_patterns()
        )
        ignored_names = ignored_names.union(autogenerated_ignored_names)

        for node in nncf_graph.topological_sort():
            is_node_with_weights = self._backend_entity.is_node_with_weights(node, nncf_graph)
            is_within_scope = should_consider_scope(node.node_name, ignored_names)
            if node.metatype in weighted_metatypes and is_node_with_weights and is_within_scope:
                ordered_nodes_to_compress.append(node)
        return ordered_nodes_to_compress

    def _get_ratio_defining_params(
        self, all_weight_params: list[WeightCompressionParameters], is_last_layer_shared: bool
    ) -> list[WeightCompressionParameters]:
        """
        Returns the information about weights that are used for ratio calculation between primary
        and backup precisions.

        :param all_weight_params: List of all weight parameters.
        :param is_last_layer_shared: Indicates whether the last layer which shares the weight
            should be quantized or not.
        :return: Information about each weight node that is considered for mixed precision.
        """
        if self._mode in [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM]:
            return all_weight_params

        ratio_defining_params = list(
            filter(
                lambda wp: wp.node_with_weight.metatype in self._backend_entity.matmul_metatypes,
                all_weight_params,
            )
        )

        # The last MatMul layer is quantized to 4-bits if all_layers=True
        if not self._all_layers and not is_last_layer_shared:
            ratio_defining_params = ratio_defining_params[:-1]

        # Embedding layers are quantized to 4-bits only if all_layers=True.
        if self._all_layers:
            embedding_params = list(
                filter(
                    lambda wp: wp.node_with_weight.metatype in self._backend_entity.embedding_metatypes
                    and len(wp.reduction_axes) == 1,
                    all_weight_params,
                )
            )
            ratio_defining_params.extend(embedding_params)

        return ratio_defining_params

    def _get_primary_config(self, group_size: int) -> WeightCompressionConfig:
        codebook_values = None

        if self._mode == CompressWeightsMode.CB4_F8E4M3:
            codebook_values = Tensor(CB4_QUANTILES)
        elif self._mode == CompressWeightsMode.CODEBOOK:
            codebook_values = Tensor(self._advanced_parameters.codebook)

        return WeightCompressionConfig(
            mode=self._mode,
            group_size=group_size,
            codebook_values=codebook_values,
        )

    def _set_weight_compression_config(
        self,
        ratio_defining_params: list[WeightCompressionParameters],
        model: TModel,
        graph: NNCFGraph,
        statistics_points: StatisticPointsContainer,
        group_size_values: dict[str, int],
    ) -> None:
        """
        Sets the appropriate compression configuration for weights based on some criteria.

        :param ratio_defining_params: Information about weights that are used for calculating ratio between primary and
            backup precisions.
        :param model: The model.
        :param graph: The model graph associated with the model.
        :param statistics_points: Statistics points.
        :param group_size_values: A dictionary mapping weight names to their group size values.
        """
        if self._ratio < 1 and len(ratio_defining_params) > 0:
            primary_precision_weight_params = self._mixed_precision_algo.apply(
                model, graph, statistics_points, weight_params=ratio_defining_params
            )
        else:
            primary_precision_weight_params = ratio_defining_params

        for weight_param in primary_precision_weight_params:
            weight_param.compression_config = self._get_primary_config(group_size_values[weight_param.weight_name])

        # Check if group size is valid for each weight in ratio_defining_params
        failed_nodes = []
        for w_params in ratio_defining_params:
            if w_params.compression_config is None or w_params.compression_config.group_size == -1:
                continue
            reduction_channel_size, _ = get_reduction_channel_size(w_params.weight_shape, w_params.reduction_axes)
            if reduction_channel_size % w_params.compression_config.group_size != 0:
                failed_nodes.append((w_params.node_with_weight.node_name, reduction_channel_size))
        if len(failed_nodes) > 0:
            names = ",".join(f'"{name}"' for name, _ in failed_nodes)
            msg = (
                "Failed to apply group-wise quantization with "
                f"group size value {self._group_size} and channel size value {failed_nodes[0][1]}.\n"
                "Ensure that the group size is divisible by the channel size, "
                "or include this node and others with similar issues in the ignored scope:\n"
                f"nncf.compress_weight(\n\t..., \n\tignored_scope=IgnoredScope(names=[{names}]\n\t)\n)"
            )
            raise nncf.InvalidGroupSizeError(msg)

    def _get_flexible_group_size_data(
        self, weight_params: list[WeightCompressionParameters]
    ) -> list[tuple[WeightCompressionParameters, int]]:
        """
        Compute flexible group size values.
        :param weight_params: Weight parameters for which to compute flexible group size.
        :return: A list of tuples, where each tuple pair contains a WeightCompressionParameters object and the
            group size values associated with it. If group size can't be assigned to some weight parameter
            it won't be included in the result.
        """
        flexible_group_size_not_found_weight_params = []
        group_size_data = []
        for w_params in weight_params:
            reduction_channel_size, _ = get_reduction_channel_size(w_params.weight_shape, w_params.reduction_axes)
            if reduction_channel_size % self._group_size == 0:
                # The weight can be compressed with the given group size, nothing else to do
                group_size_data.append((w_params, self._group_size))
                continue

            # Find the maximal power of two that divides reduction_channel_size
            flexible_group_size = reduction_channel_size & (~reduction_channel_size + 1)

            if flexible_group_size < self._min_flexible_group_size:
                flexible_group_size_not_found_weight_params.append(w_params)
            else:
                group_size_data.append((w_params, flexible_group_size))

        node_strings = []
        for i, (w_params, new_group_size) in enumerate(group_size_data):
            if new_group_size == self._group_size:
                continue
            weight_shape = w_params.weight_shape
            reduction_channel_size, _ = get_reduction_channel_size(weight_shape, w_params.reduction_axes)
            node_strings.append(
                f"{w_params.node_with_weight.node_name} "
                f"(weight shape: {weight_shape}, adjusted group size: {new_group_size})"
            )
        if len(node_strings) > 0:
            nncf_logger.info(
                f"Wasn't able to set the specified group size value ({self._group_size}) to some nodes. These nodes "
                f"will have an adjusted group size value:\n\t" + "\n\t".join(node_strings)
            )

        if len(flexible_group_size_not_found_weight_params) > 0:
            node_strings = [""] * len(flexible_group_size_not_found_weight_params)
            for i, w_params in enumerate(flexible_group_size_not_found_weight_params):
                weight_shape = w_params.weight_shape
                reduction_channel_size, _ = get_reduction_channel_size(weight_shape, w_params.reduction_axes)
                node_strings[i] = f"{w_params.node_with_weight.node_name} (weight shape: {weight_shape})"
            nncf_logger.warning(
                "Large enough flexible group size value cannot be found for some nodes. They will be compressed "
                "according to the backup mode. Nodes:\n\t" + "\n\t".join(node_strings)
            )

        return group_size_data

    @staticmethod
    def _proportion_str(num_weights_list: list[int], total_num_weights: int, total_num_params: int) -> str:
        """
        Generates a string with proportion between target parameters and all model parameters by number of weights.

        :param num_weights_list: List of number of weights of target model parameters.
        :param total_num_weights: The total number of weights.
        :param total_num_params: The total number of model parameters.
        :return: The string with proportion between target parameters and all model parameters by number of weights.
        """
        percentage = sum(num_weights_list) / max(total_num_weights, 1) * 100
        return f"{percentage:.0f}% ({len(num_weights_list)} / {total_num_params})"

    def _get_bitwidth_distribution_str(
        self,
        all_params: list[WeightCompressionParameters],
        ratio_defining_params: list[WeightCompressionParameters],
        ignored_scope_weight_statistics: list[int],
    ) -> str:
        """
        Generates a table that shows the ratio of weights quantized to different number of bits.

        :param all_params: Information about each weight node.
        :param ratio_defining_params: Information about weights that are used for calculating ratio between primary and
            backup precisions.
        :param ignored_scope_weight_statistics: Information about weight nodes from IgnoredScope.
        :return: A string containing the table.
        """
        dtype_vs_num_weights_map = {}
        ratio_defining_weight_names = set(wp.weight_name for wp in ratio_defining_params)
        for data in all_params:
            dtype = data.compression_config.mode if data.compression_config is not None else "float"
            n_total, n_ratio_defining = dtype_vs_num_weights_map.get(dtype, ([], []))
            if data.weight_name in ratio_defining_weight_names:
                n_ratio_defining.append(data.num_weights)
            n_total.append(data.num_weights)
            dtype_vs_num_weights_map[dtype] = (n_total, n_ratio_defining)

        if ignored_scope_weight_statistics:
            n_total, n_ratio_defining = dtype_vs_num_weights_map.get("float", ([], []))
            dtype_vs_num_weights_map["float"] = (n_total + ignored_scope_weight_statistics, n_ratio_defining)

        num_ratio_defining_weights = sum(ws.num_weights for ws in ratio_defining_params)
        num_ratio_defining_params = len(ratio_defining_params)
        num_total_weights = sum(ws.num_weights for ws in all_params) + sum(ignored_scope_weight_statistics)
        num_params = len(all_params) + len(ignored_scope_weight_statistics)
        dtype_vs_num_weights_map = OrderedDict(sorted(dtype_vs_num_weights_map.items(), reverse=True))
        # Table creation
        header = ["Weight compression mode", "% all parameters (layers)", "% ratio-defining parameters (layers)"]
        rows = []
        for bitwidth, (n_total, n_ratio_defining) in dtype_vs_num_weights_map.items():
            rows.append(
                [
                    bitwidth,
                    self._proportion_str(n_total, num_total_weights, num_params),
                    self._proportion_str(n_ratio_defining, num_ratio_defining_weights, num_ratio_defining_params),
                ]
            )

        table = create_table(header, rows)
        pretty_string = f"Statistics of the bitwidth distribution:\n{table}"
        return pretty_string

    def _get_ignored_scope_weight_statistics(self, model: TModel, graph: NNCFGraph) -> list[int]:
        """
        Collect the weight statistics for nodes in the ignored scope.

        :param model: Model for statistics collection.
        :param graph: Model graph.
        :return: A list of weight sizes for the ignored nodes.
        """
        ignored_names = get_ignored_node_names_from_ignored_scope(self._ignored_scope, graph, strict=False)
        weighted_metatypes = (
            self._backend_entity.matmul_metatypes
            + self._backend_entity.embedding_metatypes
            + self._backend_entity.convolution_metatypes
        )
        ignored_scope_weight_statistics = []
        for node_name in ignored_names:
            node = graph.get_node_by_name(node_name)
            is_node_with_weights = self._backend_entity.is_node_with_weights(node, graph)
            if not is_node_with_weights or node.metatype not in weighted_metatypes:
                continue
            for _, weight_port_id in self._backend_entity.get_weight_names_and_port_ids(node, graph):
                weight_dtype = self._backend_entity.get_weight_dtype(node, weight_port_id, model, graph)
                if weight_dtype not in SUPPORTED_DATA_TYPES:
                    continue
                weight_shape = self._backend_entity.get_weight_shape(node, weight_port_id, graph)
                weight_size = reduce(operator.mul, weight_shape, 1)
                ignored_scope_weight_statistics.append(weight_size)
        return ignored_scope_weight_statistics

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        self.set_backend_entity(model)

        nodes_to_compress = self.get_nodes_to_compress(graph)

        statistics = None
        if (self._data_aware_mixed_precision or self._data_aware_compression) and dataset:
            matmul_nodes_to_compress = [
                node for node in nodes_to_compress if node.metatype in self._backend_entity.matmul_metatypes
            ]
            matmul_input_to_output_nodes_map = self.get_matmul_input_to_output_nodes_map(
                matmul_nodes_to_compress, graph
            )
            if statistic_points is None:
                statistic_points = self.get_statistic_points(model, graph, matmul_input_to_output_nodes_map.keys())
                statistic_points = self._collect_statistics(dataset, graph, model, statistic_points)
            statistics = self._get_statistics_for_weights_compression(
                matmul_input_to_output_nodes_map, statistic_points
            )

        all_weight_params: list[WeightCompressionParameters] = []
        weight_names = set()

        is_last_layer_shared = False
        n = len(nodes_to_compress)
        for i, node in enumerate(nodes_to_compress):
            for weight_name, weight_port_id in self._backend_entity.get_weight_names_and_port_ids(node, graph):
                if weight_name in weight_names:
                    if i == n - 1:
                        is_last_layer_shared = True
                    continue

                weight_dtype = self._backend_entity.get_weight_dtype(node, weight_port_id, model, graph)
                if weight_dtype not in SUPPORTED_DATA_TYPES:
                    continue
                weight_shape = self._backend_entity.get_weight_shape(node, weight_port_id, graph)
                reduction_axes = self._backend_entity.get_reduction_axes(node, weight_port_id, graph)
                if (
                    self._group_size != -1
                    and self._all_layers
                    and node.metatype in self._backend_entity.embedding_metatypes
                    and isinstance(reduction_axes, tuple)
                    and len(reduction_axes) != 1
                ):
                    # NNCF supports multiple reduction axes only for ops with group_size != -1.
                    # Convolution ops are always kept in backup mode.
                    # Embedding layers are quantized to 4-bits only if all_layers=True.
                    # MatMul ops can't have multiple reduction axes.
                    nncf_logger.warning(
                        f"Weight compression expects a single reduction axis, but {len(reduction_axes)} given. "
                        f"Weight shape: {weight_shape}, reduction axes: {reduction_axes}, "
                        f"node name: {node.node_name}. The node will be in {self._backup_mode} mode."
                    )

                if self._backup_mode == BackupMode.NONE:
                    wc_config = None
                else:
                    mode = (
                        CompressWeightsMode.INT8_ASYM
                        if self._backup_mode == BackupMode.INT8_ASYM
                        else CompressWeightsMode.INT8_SYM
                    )
                    wc_config = WeightCompressionConfig(mode=mode)
                weight_params = WeightCompressionParameters(
                    weight_name, node, weight_port_id, weight_shape, reduction_axes, wc_config
                )
                all_weight_params.append(weight_params)
                weight_names.add(weight_name)

        ratio_defining_params = self._get_ratio_defining_params(all_weight_params, is_last_layer_shared)
        if self._enable_flexible_group_size and self._group_size != -1:
            # Compute flexible group size values if enabled
            flexible_group_size_data = self._get_flexible_group_size_data(ratio_defining_params)
            group_size_values = {w_param.weight_name: group_size for w_param, group_size in flexible_group_size_data}
            # Select a subset of ratio_defining_params that can be compressed with some group size
            ratio_defining_params = [w_param for w_param, _ in flexible_group_size_data]
        else:
            group_size_values = {w_param.weight_name: self._group_size for w_param in ratio_defining_params}
        self._set_weight_compression_config(ratio_defining_params, model, graph, statistic_points, group_size_values)
        ignored_scope_weight_statistics = self._get_ignored_scope_weight_statistics(model, graph)
        nncf_logger.info(
            self._get_bitwidth_distribution_str(
                all_weight_params, ratio_defining_params, ignored_scope_weight_statistics
            )
        )

        if self._backup_mode == BackupMode.NONE:
            # Filter all_weight_params and nodes_to_compress by excluding nodes
            # that should remain in their original floating-point precision
            nodes_names_to_exclude = {
                w_params.node_with_weight.node_name
                for w_params in all_weight_params
                if w_params.compression_config is None
            }
            all_weight_params = list(
                filter(
                    lambda w_params: w_params.node_with_weight.node_name not in nodes_names_to_exclude,
                    all_weight_params,
                )
            )
            nodes_to_compress = list(
                filter(lambda node: node.node_name not in nodes_names_to_exclude, nodes_to_compress)
            )
        if self._awq:
            self.awq_algo.apply(model, graph, all_weight_params, nodes_to_compress, statistics, self._backend_entity)
            # After applying AWQ we need to update statistics since AWQ alters the activations
            statistics = self.awq_algo.update_statistics(statistics)
            # del is used to prematurely mark non-necessary data as free for garbage collection
            del self.awq_algo

        precomputed_compressed_weights = None
        lora_correction_algo = None
        description = "Applying Weight Compression"

        if self._gptq:
            del statistics
            model, precomputed_compressed_weights = self._gptq_algo.apply(
                model=model,
                graph=graph,
                dataset=dataset,
                weight_compression_parameters=all_weight_params,
                backend_entity=self._backend_entity,
            )
        else:
            if self._scale_estimation:
                precomputed_compressed_weights = self._scale_estimation_algo.apply(
                    model=model,
                    graph=graph,
                    all_weight_params=all_weight_params,
                    statistics=statistics,
                    backend_entity=self._backend_entity,
                )

            if self._lora_correction:
                lora_correction_params = self._advanced_parameters.lora_correction_params
                lora_correction_algo = LoraCorrectionAlgorithm(statistics, lora_correction_params)
                description += " with correction of low-rank adapters"
            del statistics

        # Sort weight params to start compression with the bigger constants. This lowers peak memory footprint.
        all_weight_params = sorted(all_weight_params, key=lambda wp: wp.num_weights, reverse=True)
        all_weight_sizes = [wp.num_weights for wp in all_weight_params]

        # Compress model using weight compression parameters
        transformed_model = self._backend_entity.transform_model(
            model,
            graph,
            track(all_weight_params, description=description, weights=all_weight_sizes),
            precomputed_compressed_weights,
            lora_correction_algo,
            self._compression_format,
            self._advanced_parameters,
        )

        self._backend_entity.dump_parameters(
            model,
            parameters={
                "mode": self._mode.value,
                "group_size": self._group_size,
                "ratio": self._ratio,
                "all_layers": self._all_layers,
                "ignored_scope": self._ignored_scope,
                "sensitivity_metric": self._sensitivity_metric.value,
                "awq": self._awq,
                "scale_estimation": self._scale_estimation,
                "gptq": self._gptq,
                "lora_correction": self._lora_correction,
                "backup_mode": self._backup_mode.value,
                "compression_format": self._compression_format.value,
                "advanced_parameters": convert_to_dict_recursively(self._advanced_parameters),
            },
            algo_name="weight_compression",
        )
        return transformed_model

    def _get_activation_node_and_port(self, node: NNCFNode, nncf_graph: NNCFGraph) -> tuple[NNCFNode, int]:
        """
        This method returns the activation layer and corresponding port id for the node.

        :param node: NNCFGraph node for which the activation is sought.
        :param nncf_graph: NNCFGraph instance with the node.
        :return: Tuple with the activation node and port id.
        """
        activation_port = self._backend_entity.get_activation_port_id(node, nncf_graph)
        activation_edge = nncf_graph.get_input_edge_by_port_id(node, activation_port)
        activation_node = activation_edge.from_node
        port_id = activation_edge.output_port_id
        return activation_node, port_id

    def get_matmul_input_to_output_nodes_map(
        self, matmul_nodes: list[NNCFNode], graph: NNCFGraph
    ) -> dict[tuple[NNCFNode, int], list[NNCFNode]]:
        """
        Maps activation nodes to their corresponding MatMul nodes in the graph.

        Each weighted MatMul node takes two inputs: an activation and a weight.
        An activation node may serve as an input to multiple MatMul nodes.
        This function returns a mapping where each key is a tuple consisting of an
        activation node and its output port ID, and the value is a list of MatMul
        nodes that use this activation as input.

        :param matmul_nodes: A list of MatMul nodes from the computation graph.
        :param graph: An instance of NNCFGraph representing the computation graph.
        :return: A dictionary mapping from a tuple of (activation node, port ID)
        to a list of corresponding MatMul nodes that accept the activation as input.
        """
        matmul_input_to_output_nodes_map = defaultdict(list)
        for node in matmul_nodes:
            act_node, output_port_id = self._get_activation_node_and_port(node, graph)
            matmul_input_to_output_nodes_map[(act_node, output_port_id)].append(node)
        return matmul_input_to_output_nodes_map

    def get_compression_nodes_info(
        self, graph: NNCFGraph
    ) -> tuple[list[NNCFNode], dict[tuple[NNCFNode, int], list[NNCFNode]]]:
        """
        Retrieves the nodes to compress along with a mapping of activation nodes
        to their corresponding MatMul nodes.

        This function first identifies all nodes that can be compressed from the
        provided graph. It then filters these nodes to find those that are of
        MatMul type and generates a mapping of activation nodes to their
        corresponding MatMul nodes using the
        `get_matmul_input_to_output_nodes_map` function.

        :param graph: An instance of NNCFGraph representing the computation graph.
        :return: A tuple containing:
        - Nodes for compression.
        - A dictionary mapping from a tuple of (activation node, port ID)
        to a list of MatMul nodes that accept the activation as input.
        """
        nodes_to_compress = self.get_nodes_to_compress(graph)
        matmul_nodes_to_compress = [
            node for node in nodes_to_compress if node.metatype in self._backend_entity.matmul_metatypes
        ]
        matmul_input_to_output_nodes_map = self.get_matmul_input_to_output_nodes_map(matmul_nodes_to_compress, graph)
        return nodes_to_compress, matmul_input_to_output_nodes_map

    def _collect_statistics(
        self,
        dataset: Dataset,
        graph: NNCFGraph,
        model: TModel,
        statistic_points: StatisticPointsContainer,
    ):
        """
        Creates statistics aggregator, registers all statistics specified for algorithm, and then collect them.

        :param dataset: Dataset to collect values.
        :param graph: Model graph.
        :param model: Model for statistics collection.
        :param statistic_points: Statistics points.
        """
        statistics_aggregator = StatisticsAggregatorFactory.create(model, dataset)
        statistics_aggregator.register_statistic_points(statistic_points)
        statistics_aggregator.collect_statistics(model, graph)
        return statistics_aggregator.statistic_points

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
        statistic_container = StatisticPointsContainer()
        # Statistics for data aware algorithms
        if self._data_aware_compression:
            for node, output_port_id in nodes_and_port_ids:
                statistic_point = self._backend_entity.target_point(
                    TargetType.POST_LAYER_OPERATION, node.node_name, port_id=output_port_id
                )
                # Reduce activations across all but the last dimension. The last dimension is assumed to be the hidden
                # size dimension.
                n_dims = len(graph.get_output_edges_by_port_id(node, output_port_id)[0].tensor_shape)
                stat_collector = self._backend_entity.mean_statistic_collector(
                    reduction_axes=tuple(range(n_dims - 1)), subset_size=self._subset_size
                )
                statistic_container.add_statistic_point(
                    StatisticPoint(
                        target_point=statistic_point, tensor_collector=stat_collector, algorithm=self._algorithm_key
                    )
                )
        # Statistics for mixed precision algorithm
        if self._data_aware_mixed_precision:
            mixed_precision_statistics = self._mixed_precision_algo.get_statistic_points(
                model, graph, nodes_and_port_ids
            )
            for points in mixed_precision_statistics.values():
                for point in points:
                    statistic_container.add_statistic_point(point)

        return statistic_container

    def _get_statistics_for_weights_compression(
        self,
        matmul_input_to_output_nodes_map: dict[tuple[NNCFNode, int], list[NNCFNode]],
        statistic_points: StatisticPointsContainer,
    ) -> dict[str, WCTensorStatistic]:
        """
        Retrieve collected statistics only for WeightCompression algorithm and not for MixedPrecision.

        :param matmul_input_to_output_nodes_map: A mapping from activation node and a port id to corresponding matmul
            nodes which accept this activation as an input.
        :param statistic_points: Statistic points object.
        :return: Collected statistics.
        """
        # For each node we store statistics in a WCTensorStatistics data-class. It contains the following fields:
        #   mean_values=[mean_value_1, ..., mean_value_n]
        #   shapes=[shape_1, ..., shape_n]
        # Where mean_value is a 1D tensor representing an activation reduced over batch and sequence length dimensions,
        # shape is an original shape of an activation before reduction, n is the size of the dataset (or subset_size).
        statistics = {}
        for (act_node, output_port_id), matmul_nodes in matmul_input_to_output_nodes_map.items():
            tensor_collectors = list(
                statistic_points.get_algo_statistics_for_node(
                    act_node.node_name,
                    self._backend_entity.get_filter_fn_for_statistics(output_port_id, self._algorithm_key),
                    self._algorithm_key,
                )
            )
            # Statistics could be empty in case when the statistics is registered for another algorithm,
            # e.g. mixed precision.
            if tensor_collectors:
                assert len(tensor_collectors) == 1
                stats = tensor_collectors[0].get_statistics()

                # Each activation node may have multiple MatMul nodes which it is an input to
                for node in matmul_nodes:
                    statistics[node.node_name] = copy.deepcopy(stats)
        return statistics
