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

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import nncf
from nncf import Dataset
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.experimental.common.tensor_statistics.statistics import WCTensorStatistic
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.activation_stats import process_stats
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.weight_lowering import calculate_normalized_weight_and_fp4_scale
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_int_dequantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_int_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_nf4_dequantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_nf4_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import reshape_weight_for_grouped_quantization
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor import functions as fns

TModel = TypeVar("TModel")


class ScaleEstimation:
    """
    Scale estimation algorithm implementation.
    """

    compress_decompress_cache = {}

    def __init__(
        self,
        model: TModel,
        name_to_node_mapping: Dict[str, Any],
        all_weight_params: List[WeightCompressionParameters],
        nodes_to_compress: List[NNCFNode],
        statistics: Dict[str, WCTensorStatistic],
        subset_size: int = 32,
        initial_steps: int = 5,
        scale_steps: int = 10,
        weight_penalty: float = -1.0,
    ):
        """
        :param model: Model for applying algorithm.
        :param name_to_node_mapping: Name to node mapping for updating node weights.
        :param all_weight_params: List of all weight parameters.
        :param nodes_to_compress: List of nodes for processing.
        :param statistics: Input activation statistics for each node.
        :param subset_size: The number of samples for scale estimation.
        :param initial_steps: The number of the steps for absmax scale rectification.
        :param scale_steps: The number of the steps for grid search scale rectification
                            from 1.0 to 1.0 - 0.05 * scale_step.
        :param weight_penalty: coefficient for penalty between fp and compressed weights. If -1 then doesn't apply.
        """
        super().__init__()
        self.name_to_node_mapping = name_to_node_mapping
        self._all_weight_params = all_weight_params
        self._nodes_to_compress = nodes_to_compress
        self._statistics = statistics
        self._subset_size = subset_size
        self._initial_steps = initial_steps
        self._scale_steps = scale_steps
        self._weight_penalty = weight_penalty

        self._set_backend_entity(model)

    @property
    def available_backends(self) -> List[BackendType]:
        return [BackendType.OPENVINO]

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        :param all_weight_params: List of all weight parameters.
        :param nodes_to_compress: List of nodes for processing.
        :param activations: The input activations of the layers considered for compression.
        """

        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.weight_compression.openvino_backend import OVWeightCompressionAlgoBackend

            self._backend_entity = OVWeightCompressionAlgoBackend(model, self.name_to_node_mapping)
        else:
            raise nncf.UnsupportedBackendError(
                "Cannot return backend-specific AWQ entity because {} is not supported!".format(model_backend.value)
            )

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Estimates better scale for the int4 nodes in the model.
        Minimizes per-group difference between floating point MatMul and
        MatMul with compressed weights.
        The algorithm computes weighted scale for the group of weights in MatMul, which
        shared the same scale.

        :param model: Model for applying algorithm.
        :param graph: Model graph.
        :param statistic_points: Statistic points with collected statistics values.
        :param dataset: A representative dataset for the calibration process.
        :return: Two dictionaries for estimated scales and zero points for each weight name.
        """

        scales, zero_points = dict(), dict()

        for wp in track(self._all_weight_params, description="Applying Scale Estimation"):
            weight_name = wp.weight_name
            node_name = wp.node_with_weight.node_name
            config = wp.compression_config

            if config.num_bits != 4 or node_name not in self._statistics:
                scales[weight_name] = None
                continue

            stats = self._statistics[node_name]

            weight_data = self._backend_entity.get_weight_names_and_port_ids(wp.node_with_weight, graph)
            if len(weight_data) != 1:  # not supported by the algorithm
                continue
            _, weight_port_id = weight_data[0]

            weight = self._backend_entity.get_weight(wp.node_with_weight, weight_port_id, model, graph)

            scales[weight_name], zero_points[weight_name] = self.calculate_quantization_params(
                self._backend_entity,
                stats,
                weight,
                wp.reduction_axes,
                config,
                self._subset_size,
                self._initial_steps,
                self._scale_steps,
                self._weight_penalty,
            )

        return scales, zero_points

    @staticmethod
    def calculate_quantization_params(
        backend_entity: WeightCompressionAlgoBackend,
        statistics: WCTensorStatistic,
        weight: Tensor,
        reduction_axes: Tuple[int, ...],
        config: WeightCompressionConfig,
        subset_size: int = 32,
        initial_steps: int = 5,
        scale_steps: int = 10,
        weight_penalty: float = -1.0,
    ) -> Tensor:
        """
        Calculates the quantization parameters for a given set of weights and activations.
        This function estimates the optimal quantization scale for weight compression by
        minimizing the difference between floating-point operations and operations with
        quantized weights.

        The function uses an iterative process:
        1. Initial scale rectification based on activation statistics.
        2. A grid search to further refine the scale parameters.

        :param backend_entity: The backend-specific implementation of the weight compression algorithm.
        :param statistics: The input activations of the layer reduced over batch and sequence length dimensions,
            together with original activation tensor shapes.
        :param weight: The weight tensor that is being quantized.
        :param reduction_axes: Tuple specifying the axes along which the reduction is performed for quantization.
        :param config: Configuration parameters for the weight compression, including quantization settings.
        :param subset_size: The number of samples to use for scale estimation. Defaults to 32.
        :param initial_steps: The number of steps for initial scale rectification using activation statistics.
            Defaults to 5.
        :param scale_steps: The number of steps for refining the scale using a grid search. Defaults to 10.
        :param weight_penalty: Penalty coefficient applied to the difference between floating-point
            and quantized weights. A value of -1 disables the penalty. Defaults to -1.0.
        :return: A tensor containing the calculated quantization scales and zero points if applicable.
        """
        reduction_axis = reduction_axes[0]

        s, X = process_stats(statistics, subset_size)

        weight = weight.astype(TensorDataType.float32)
        eps = fns.finfo(weight).eps

        if reduction_axis == 0:
            weight = fns.transpose(weight)
            reduction_axis = 1

        group_size = config.group_size if config.group_size != -1 else weight.shape[reduction_axis]
        cur_config = deepcopy(config)
        cur_config.group_size = group_size

        original_weight = fns.zeros_like(weight) + weight
        if config.mode == CompressWeightsMode.NF4:
            norm_weight, scale = calculate_normalized_weight_and_fp4_scale(
                original_weight, reduction_axis, cur_config.group_size
            )
            compressed_weights = do_nf4_quantization(norm_weight, scale, is_normalized_weight=True)
            q_weights = do_nf4_dequantization(compressed_weights, scale, reduction_axis)
            zp = None
        else:
            compressed_weights, scale, zp = do_int_quantization(original_weight, reduction_axis, cur_config)
            if zp is not None:
                zp = zp.astype(scale.dtype)
            q_weights = do_int_dequantization(compressed_weights, scale, zp, reduction_axis)

        s = fns.unsqueeze(s, 0)
        s, _ = reshape_weight_for_grouped_quantization(s, reduction_axis, group_size)

        original_weight, _ = reshape_weight_for_grouped_quantization(original_weight, reduction_axis, group_size)

        # all weight in group has importance based on corresponding input activations
        importance = fns.ones_like(original_weight)
        importance = importance * s

        target, zero_mask = get_target_zero_mask(compressed_weights, zp)
        importance = fns.where(zero_mask, 0.0, importance)

        # normalize importances for every group of weights to make sum of them equal to 1.0
        denum = fns.sum(importance, axis=2, keepdims=True)
        importance = importance / (denum + eps)

        X, _ = reshape_weight_for_grouped_quantization(X, 0, group_size)
        q_weights, _ = reshape_weight_for_grouped_quantization(q_weights, reduction_axis, group_size)
        best_diffs = None
        result_scale = None

        fp_outs = fns.matmul(fns.transpose(original_weight, (1, 0, 2)), X)
        q_outs = fns.matmul(fns.transpose(q_weights, (1, 0, 2)), X)

        # metric for minimization with shape [C_OUT, N_GROUPS], N_GROUPS = C_IN / GROUP_SIZE
        min_max_scale_diffs = fns.mean((fp_outs - q_outs) ** 2, axis=-1)
        min_max_scale_diffs = fns.transpose(min_max_scale_diffs, (1, 0))
        if weight_penalty > 0.0:
            min_max_scale_diffs += weight_penalty * fns.mean((q_weights - original_weight) ** 2, axis=-1)

        zp_shape = zp.shape if zp is not None else None
        key = (config.mode, config.num_bits) + q_weights.shape + scale.shape
        if zp is not None:
            key += zp_shape
        if config.mode != CompressWeightsMode.NF4:
            if key in ScaleEstimation.compress_decompress_cache:
                compress_decompress_model = ScaleEstimation.compress_decompress_cache[key]["compress_decompress_model"]
                compress_model = ScaleEstimation.compress_decompress_cache[key]["compress_model"]
            else:
                compress_decompress_model = backend_entity.get_compress_decompress_pipeline(
                    config, q_weights.shape, scale.shape, zp_shape
                )
                compress_model = backend_entity.get_compress_pipeline(config, q_weights.shape, scale.shape, zp_shape)
                ScaleEstimation.compress_decompress_cache[key] = {
                    "compress_decompress_model": compress_decompress_model,
                    "compress_model": compress_model,
                }
        scale_sign = scale / fns.abs(scale)
        zero_scale = 0.001
        zero_mask = zero_scale * zero_mask.astype(original_weight.dtype)

        input_tensors = [original_weight.data, None]
        if zp is not None:
            input_tensors.append(zp.data)
        # iterative rectification of initial scale
        for i in range(initial_steps):
            near_to_ideal_scale = estimate_scales(original_weight, target, zero_mask, importance)
            near_to_ideal_scale = near_to_ideal_scale * scale_sign
            input_tensors[1] = near_to_ideal_scale.data

            if config.mode == CompressWeightsMode.NF4:
                g_compressed_weighs = do_nf4_quantization(original_weight, near_to_ideal_scale)
                out = do_nf4_dequantization(g_compressed_weighs, near_to_ideal_scale)
            else:
                out = compress_decompress_model(input_tensors)
            q_weights_ = fns.zeros_like(original_weight) + out
            q_outs = fns.matmul(fns.transpose(q_weights_, (1, 0, 2)), X)

            ideal_scale_diffs = fns.mean((fp_outs - q_outs) ** 2, axis=-1)
            ideal_scale_diffs = fns.transpose(ideal_scale_diffs, (1, 0))
            if weight_penalty > 0.0:
                ideal_scale_diffs += weight_penalty * fns.mean((q_weights_ - original_weight) ** 2, axis=-1)

            if best_diffs is None:
                best_diffs = min_max_scale_diffs

            mask = (ideal_scale_diffs > best_diffs).astype(best_diffs.dtype)

            best_diffs = mask * best_diffs + (1.0 - mask) * ideal_scale_diffs

            mask = fns.unsqueeze(mask, axis=2)

            if result_scale is None:
                near_to_ideal_scale = mask * scale + (1.0 - mask) * near_to_ideal_scale
            else:
                near_to_ideal_scale = mask * result_scale + (1.0 - mask) * near_to_ideal_scale
            result_scale = near_to_ideal_scale
            input_tensors[1] = near_to_ideal_scale.data

            if i < initial_steps - 1:
                if config.mode == CompressWeightsMode.NF4:
                    out = do_nf4_quantization(original_weight, near_to_ideal_scale)
                else:
                    out = compress_model(input_tensors)
                compressed_weights = fns.zeros_like(original_weight) + out
                target, zero_mask = get_target_zero_mask(compressed_weights, zp)
                zero_mask = zero_scale * zero_mask.astype(original_weight.dtype)

        # iterative rectification of scale based on grid search
        for scale_step in range(scale_steps):
            factor = 1.0 - 0.05 * scale_step
            scaled_scale = factor * scale

            input_tensors[1] = scaled_scale.data
            if config.mode == CompressWeightsMode.NF4:
                out = do_nf4_quantization(original_weight, scaled_scale)
            else:
                out = compress_model(input_tensors)
            compressed_weights = fns.zeros_like(original_weight) + out

            target, zero_mask = get_target_zero_mask(compressed_weights, zp)
            zero_mask = zero_scale * zero_mask.astype(original_weight.dtype)
            near_to_ideal_scale = estimate_scales(original_weight, target, zero_mask, importance)
            near_to_ideal_scale = near_to_ideal_scale * scale_sign

            input_tensors[1] = near_to_ideal_scale.data
            if config.mode == CompressWeightsMode.NF4:
                g_compressed_weighs = do_nf4_quantization(original_weight, near_to_ideal_scale)
                out = do_nf4_dequantization(g_compressed_weighs, near_to_ideal_scale)
            else:
                out = compress_decompress_model(input_tensors)
            q_weights_ = fns.zeros_like(original_weight) + out

            q_outs = fns.matmul(fns.transpose(q_weights_, (1, 0, 2)), X)
            ideal_scale_diffs = fns.mean((fp_outs - q_outs) ** 2, axis=-1)
            ideal_scale_diffs = fns.transpose(ideal_scale_diffs, (1, 0))
            if weight_penalty > 0.0:
                ideal_scale_diffs += weight_penalty * fns.mean((q_weights_ - original_weight) ** 2, axis=-1)

            mask = (ideal_scale_diffs > best_diffs).astype(best_diffs.dtype)

            best_diffs = mask * best_diffs + (1.0 - mask) * ideal_scale_diffs

            mask = fns.unsqueeze(mask, axis=2)

            if result_scale is None:
                near_to_ideal_scale = mask * scale + (1.0 - mask) * near_to_ideal_scale
            else:
                near_to_ideal_scale = mask * result_scale + (1.0 - mask) * near_to_ideal_scale
            result_scale = near_to_ideal_scale

        if config.group_size == -1:
            result_scale = fns.squeeze(result_scale, axis=1)
        if zp is not None and config.group_size == -1:
            zp = fns.squeeze(zp, axis=1)

        return result_scale, zp

    @staticmethod
    def activations_to_wc_statistics(activations: List[Tensor]) -> WCTensorStatistic:
        """
        Mimic the activation reducing logic from WeightCompression.get_statistic_points.

        :param activations: List of raw activations.
        :return: Instance of WCTensorStatistic class containing reduced activations and shapes.
        """
        mean_values = []
        shapes = []
        for act in activations:
            shapes.append(act.shape)
            reduction_shape = tuple(range(act.ndim - 1))
            mean_values.append(fns.mean(act, axis=reduction_shape))
        wc_statistics = WCTensorStatistic(mean_values, shapes)
        return wc_statistics


def get_target_zero_mask(compressed_weights: Tensor, zp: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """
    Computes the target values and a mask indicating zero values in the target.

    :param compressed_weights: The compressed weights tensor.
    :param zp: The zero point tensor.
    :return: The compressed weights optionally adjusted by the zero point and
        a boolean mask indicating positions in the target that are close to zero.
    """
    target = compressed_weights
    if zp is not None:
        target = target.astype(dtype=zp.dtype) - zp
    zero_mask = fns.isclose(target, 0)
    return target, zero_mask


def estimate_scales(weight: Tensor, target: Tensor, zero_mask: Tensor, importance: Tensor) -> Tensor:
    """
    Estimates scales for the given weight, target, zero mask, and importance.

    :param weight: The weights tensor.
    :param target: The target values tensor.
    :param zero_mask: A boolean mask indicating positions in the target that are close to zero.
    :param importance: The importance values tensor.
    :return: The estimated scales
    """
    ideal_scale = fns.abs(weight) / (fns.abs(target) + zero_mask)
    weighted_scale = ideal_scale * importance
    near_to_ideal_scale = fns.sum(weighted_scale, axis=2, keepdims=True)
    return near_to_ideal_scale
