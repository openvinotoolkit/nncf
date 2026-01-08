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

from copy import deepcopy
from typing import Optional, TypeVar

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistics import WCTensorStatistic
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.quantization.algorithms.weight_compression.activation_stats import process_stats
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.parameters import CompressedWeight
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_float_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_integer_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import float_quantize_dequantize_weight
from nncf.quantization.algorithms.weight_compression.weight_lowering import integer_quantize_dequantize_weight
from nncf.quantization.algorithms.weight_compression.weight_lowering import reshape_weight_for_grouped_quantization
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor import functions as fns

TModel = TypeVar("TModel")


class ScaleEstimation:
    """
    Scale estimation algorithm implementation.
    """

    def __init__(
        self,
        subset_size: int = 32,
        initial_steps: int = 5,
        scale_steps: int = 10,
        weight_penalty: float = -1.0,
    ):
        """
        :param subset_size: The number of samples for scale estimation.
        :param initial_steps: The number of the steps for absmax scale rectification.
        :param scale_steps: The number of the steps for grid search scale rectification
                            from 1.0 to 1.0 - 0.05 * scale_step.
        :param weight_penalty: coefficient for penalty between fp and compressed weights. If -1 then doesn't apply.
        """
        super().__init__()
        self._subset_size = subset_size
        self._initial_steps = initial_steps
        self._scale_steps = scale_steps
        self._weight_penalty = weight_penalty

    @property
    def available_backends(self) -> list[BackendType]:
        return [BackendType.OPENVINO, BackendType.TORCH, BackendType.ONNX]

    def _set_backend_entity(self, model: TModel) -> None:
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

            self._backend_entity = ONNXWeightCompressionAlgoBackend()
        else:
            msg = (
                "Cannot return backend-specific Scale Estimation entity because"
                f" {model_backend.value} is not supported!"
            )
            raise nncf.UnsupportedBackendError(msg)

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        all_weight_params: list[WeightCompressionParameters],
        statistics: dict[str, WCTensorStatistic],
        backend_entity: Optional[WeightCompressionAlgoBackend] = None,
    ) -> dict[str, CompressedWeight]:
        """
        Estimates better scale for the int4 nodes in the model.
        Minimizes per-group difference between floating point MatMul and
        MatMul with compressed weights.
        The algorithm computes weighted scale for the group of weights in MatMul, which
        shared the same scale.

        :param model: Model for applying algorithm.
        :param graph: Model graph.
        :param all_weight_params: List of all weight parameters.
        :param statistics: Input activation statistics for each node.
        :param statistic_points: Statistic points with collected statistics values.
        :param dataset: A representative dataset for the calibration process.
        :param backend_entity: Weight compression algorithm backend.
        :return: Two dictionaries for estimated scales and zero points for each weight name.
        """
        self._backend_entity = backend_entity
        if self._backend_entity is None:
            self._set_backend_entity(model)
        res = dict()

        for wp in track(all_weight_params, description="Applying Scale Estimation"):
            weight_name = wp.weight_name
            node_name = wp.node_with_weight.node_name
            config = wp.compression_config

            if config.num_bits != 4 or node_name not in statistics:
                res[weight_name] = CompressedWeight()
                continue

            stats = statistics[node_name]

            weight_data = self._backend_entity.get_weight_names_and_port_ids(wp.node_with_weight, graph)
            if len(weight_data) != 1:  # not supported by the algorithm
                continue
            _, weight_port_id = weight_data[0]

            weight = self._backend_entity.get_weight(wp.node_with_weight, weight_port_id, model, graph)

            scale, zero_point = self.calculate_quantization_params(
                stats,
                weight,
                wp.reduction_axes,
                config,
                self._subset_size,
                self._initial_steps,
                self._scale_steps,
                self._weight_penalty,
            )
            res[weight_name] = CompressedWeight(None, scale, zero_point, None)

        return res

    @staticmethod
    def calculate_quantization_params(
        statistics: WCTensorStatistic,
        weight: Tensor,
        reduction_axes: tuple[int, ...],
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

        X = X.astype(TensorDataType.float32)
        weight = weight.astype(TensorDataType.float32)
        eps = fns.finfo(weight).eps
        is_3d_weight = len(weight.shape) == 3

        was_transposed = False
        if reduction_axis == 0 or (reduction_axis == 1 and is_3d_weight):
            # Weights
            # 3D: [num_experts, hidden_dimension, out_features] -> [num_experts, out_features, hidden_dimension]
            # 2D: [hidden_dimension, out_features] -> [out_features, hidden_dimension]
            weight = fns.moveaxis(weight, -1, -2)
            reduction_axis = weight.ndim - 1
            was_transposed = True

        group_size = config.group_size if config.group_size != -1 else weight.shape[reduction_axis]
        cur_config = deepcopy(config)
        cur_config.group_size = group_size

        if not config.is_integer:
            q_weights, compressed_weights, scale = float_quantize_dequantize_weight(
                weight, cur_config, reduction_axis, return_compressed_weight=True
            )
            zp = None
        else:
            q_weights, compressed_weights, scale, zp = integer_quantize_dequantize_weight(
                weight, cur_config, reduction_axis, return_compressed_weight=True
            )
            if zp is not None:
                zp = zp.astype(scale.dtype)

        s = fns.unsqueeze(s, -2)
        s, _ = reshape_weight_for_grouped_quantization(s, reduction_axis, group_size)

        weight, _ = reshape_weight_for_grouped_quantization(weight, reduction_axis, group_size)

        # all weight in group has importance based on corresponding input activations
        importance = fns.ones_like(weight)
        importance = importance * s

        target, zero_mask = get_target_zero_mask(compressed_weights, zp)
        importance = fns.where(zero_mask, 0.0, importance)

        # normalize importances for every group of weights to make sum of them equal to 1.0
        denum = fns.sum(importance, axis=-1, keepdims=True)
        importance = importance / (denum + eps)

        X, _ = reshape_weight_for_grouped_quantization(X, -2, group_size)
        best_diffs = None
        result_scale = None
        fp_outs = fns.matmul(fns.moveaxis(weight, -2, -3), X)
        q_outs = fns.matmul(fns.moveaxis(q_weights, -2, -3), X)

        # metric for minimization with shape [C_OUT, N_GROUPS], N_GROUPS = C_IN / GROUP_SIZE
        # For 3D weights, it is [Batch Size, C_OUT, N_GROUPS]
        min_max_scale_diffs = fns.mean((fp_outs - q_outs) ** 2, axis=-1)
        min_max_scale_diffs = fns.moveaxis(min_max_scale_diffs, -1, -2)

        if weight_penalty > 0.0:
            min_max_scale_diffs += weight_penalty * fns.mean((q_weights - weight) ** 2, axis=-1)

        scale_sign = scale / fns.abs(scale)
        zero_scale = 0.001
        zero_mask = zero_scale * zero_mask.astype(weight.dtype)

        # iterative rectification of initial scale
        for i in range(initial_steps):
            near_to_ideal_scale = estimate_scales(weight, target, zero_mask, importance)
            near_to_ideal_scale = near_to_ideal_scale * scale_sign

            if not config.is_integer:
                q_weights_ = float_quantize_dequantize_weight(
                    weight,
                    config,
                    precomputed_scale=near_to_ideal_scale,
                )
            else:
                q_weights_ = integer_quantize_dequantize_weight(
                    weight,
                    config,
                    precomputed_scale=near_to_ideal_scale,
                    precomputed_zero_point=zp,
                )

            q_outs = fns.matmul(fns.moveaxis(q_weights_, -2, -3), X)

            ideal_scale_diffs = fns.mean((fp_outs - q_outs) ** 2, axis=-1)
            ideal_scale_diffs = fns.moveaxis(ideal_scale_diffs, -1, -2)
            if weight_penalty > 0.0:
                ideal_scale_diffs += weight_penalty * fns.mean((q_weights_ - weight) ** 2, axis=-1)

            if best_diffs is None:
                best_diffs = min_max_scale_diffs

            mask = (ideal_scale_diffs > best_diffs).astype(best_diffs.dtype)

            best_diffs = mask * best_diffs + (1.0 - mask) * ideal_scale_diffs

            mask = fns.unsqueeze(mask, axis=-1)

            if result_scale is None:
                near_to_ideal_scale = mask * scale + (1.0 - mask) * near_to_ideal_scale
            else:
                near_to_ideal_scale = mask * result_scale + (1.0 - mask) * near_to_ideal_scale
            result_scale = near_to_ideal_scale

            if i < initial_steps - 1:
                if not config.is_integer:
                    compressed_weights, _, _ = do_float_quantization(
                        weight, config, precomputed_scale=near_to_ideal_scale
                    )
                else:
                    compressed_weights, _, _ = do_integer_quantization(
                        weight,
                        config,
                        precomputed_scale=near_to_ideal_scale,
                        precomputed_zero_point=zp,
                    )
                target, zero_mask = get_target_zero_mask(compressed_weights, zp)
                zero_mask = zero_scale * zero_mask.astype(weight.dtype)

        # iterative rectification of scale based on grid search
        for scale_step in range(scale_steps):
            factor = 1.0 - 0.05 * scale_step
            scaled_scale = factor * scale

            if not config.is_integer:
                compressed_weights, _, _ = do_float_quantization(weight, config, precomputed_scale=scaled_scale)
            else:
                compressed_weights, _, _ = do_integer_quantization(
                    weight,
                    config,
                    precomputed_scale=scaled_scale,
                    precomputed_zero_point=zp,
                )

            target, zero_mask = get_target_zero_mask(compressed_weights, zp)
            zero_mask = zero_scale * zero_mask.astype(weight.dtype)
            near_to_ideal_scale = estimate_scales(weight, target, zero_mask, importance)
            near_to_ideal_scale = near_to_ideal_scale * scale_sign

            if not config.is_integer:
                q_weights_ = float_quantize_dequantize_weight(weight, config, precomputed_scale=near_to_ideal_scale)
            else:
                q_weights_ = integer_quantize_dequantize_weight(
                    weight,
                    config,
                    precomputed_scale=near_to_ideal_scale,
                    precomputed_zero_point=zp,
                )

            q_outs = fns.matmul(fns.moveaxis(q_weights_, -2, -3), X)
            ideal_scale_diffs = fns.mean((fp_outs - q_outs) ** 2, axis=-1)
            ideal_scale_diffs = fns.moveaxis(ideal_scale_diffs, -1, -2)
            if weight_penalty > 0.0:
                ideal_scale_diffs += weight_penalty * fns.mean((q_weights_ - weight) ** 2, axis=-1)

            mask = (ideal_scale_diffs > best_diffs).astype(best_diffs.dtype)

            best_diffs = mask * best_diffs + (1.0 - mask) * ideal_scale_diffs

            mask = fns.unsqueeze(mask, axis=-1)

            if result_scale is None:
                near_to_ideal_scale = mask * scale + (1.0 - mask) * near_to_ideal_scale
            else:
                near_to_ideal_scale = mask * result_scale + (1.0 - mask) * near_to_ideal_scale
            result_scale = near_to_ideal_scale

        if config.group_size == -1:
            result_scale = fns.squeeze(result_scale, axis=-2)
        if zp is not None and config.group_size == -1:
            zp = fns.squeeze(zp, axis=-2)

        if was_transposed:
            if config.group_size == -1:
                result_scale = fns.moveaxis(result_scale, -1, -2)
                if zp is not None:
                    zp = fns.moveaxis(zp, -1, -2)
            else:
                result_scale = fns.moveaxis(result_scale, (-1, -2, -3), (-2, -3, -1))
                if zp is not None:
                    zp = fns.moveaxis(zp, (-1, -2, -3), (-2, -3, -1))

        return result_scale, zp

    @staticmethod
    def activations_to_wc_statistics(activations: list[Tensor]) -> WCTensorStatistic:
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


def get_target_zero_mask(compressed_weights: Tensor, zp: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
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
    near_to_ideal_scale = fns.sum(weighted_scale, axis=-1, keepdims=True)
    return near_to_ideal_scale
