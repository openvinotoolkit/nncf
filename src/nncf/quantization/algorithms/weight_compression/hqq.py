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

from typing import TypeVar

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.logging.track_progress import track
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.parameters import CompressedWeight
from nncf.quantization.algorithms.weight_compression.weight_lowering import ReductionAxes
from nncf.quantization.algorithms.weight_compression.weight_lowering import calculate_integer_quantization_params
from nncf.quantization.algorithms.weight_compression.weight_lowering import reshape_weight_for_grouped_quantization
from nncf.tensor import Tensor
from nncf.tensor import functions as fns
from nncf.tensor.definitions import TensorBackend
from nncf.tensor.definitions import TensorDataType

TModel = TypeVar("TModel")


class HQQ:
    """
    Half-Quadratic Quantization (HQQ) algorithm implementation.

    HQQ is a data-free weight quantization algorithm that minimizes quantization error
    without requiring calibration data. It uses alternating least-squares optimization
    to find optimal scale and zero-point parameters. For asymmetric quantization, HQQ
    optimizes the zero-point as a continuous float during iterations, then rounds it to
    the nearest integer before returning so that quantization and dequantization agree.

    Reference: "Half-Quadratic Quantization of Large Machine Learning Models"
    (https://mobiusml.github.io/hqq_blog/)
    """

    def __init__(self, num_iterations: int = 20):
        """
        :param num_iterations: Number of alternating optimization iterations.
            More iterations improve quantization quality at the cost of compute time.
            Defaults to 20.
        """
        self._num_iterations = num_iterations
        self._backend_entity = None

    @property
    def available_backends(self) -> list[BackendType]:
        return [BackendType.OPENVINO, BackendType.TORCH, BackendType.ONNX]

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backend-specific logic of the algorithm.

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
            msg = (
                "Cannot return backend-specific HQQ entity because"
                f" {model_backend.value} is not supported!"
            )
            raise nncf.UnsupportedBackendError(msg)

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        all_weight_params: list[WeightCompressionParameters],
        backend_entity: WeightCompressionAlgoBackend | None = None,
    ) -> dict[str, CompressedWeight]:
        """
        Applies the HQQ algorithm to compute optimized scale and zero-point parameters.

        For each eligible weight, HQQ alternately:
          1. Quantizes the weight with the current (scale, zero_point), and
          2. Updates (scale, zero_point) via a closed-form least-squares step.

        The resulting CompressedWeight objects contain None for the compressed tensor
        (quantization is deferred) but carry the HQQ-optimized float scale and, for
        asymmetric modes, a float-valued zero point.

        :param model: Model for applying algorithm.
        :param graph: Model graph.
        :param all_weight_params: List of all weight parameters.
        :param backend_entity: Weight compression algorithm backend.
        :return: A dictionary mapping weight names to CompressedWeight instances with
            HQQ-optimized scale and zero point.
        """
        self._backend_entity = backend_entity
        if self._backend_entity is None:
            self._set_backend_entity(model)

        res = {}

        for wp in track(all_weight_params, description="Applying HQQ"):
            weight_name = wp.weight_name
            config = wp.compression_config

            if not config.is_integer:
                res[weight_name] = CompressedWeight()
                continue

            weight_data = self._backend_entity.get_weight_names_and_port_ids(wp.node_with_weight, graph)
            if len(weight_data) != 1:  # not supported by the algorithm
                continue
            _, weight_port_id = weight_data[0]

            weight = self._backend_entity.get_weight(wp.node_with_weight, weight_port_id, model, graph)

            # Convert to numpy for stable in-loop arithmetic, avoiding the OV-optimized
            # quantization path, which may not handle float zero points.
            if weight.backend == TensorBackend.ov:
                weight = weight.as_numpy_tensor()
            weight = fns.astype(weight, TensorDataType.float32)

            scale, zero_point = self._calculate_hqq_params(weight, config, wp.reduction_axes)
            res[weight_name] = CompressedWeight(None, scale, zero_point, None)

        return res

    def _calculate_hqq_params(
        self,
        weight: Tensor,
        config: WeightCompressionConfig,
        reduction_axes: ReductionAxes,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Computes HQQ-optimized scale and zero point for integer quantization.

        The algorithm alternates between two steps until convergence:
          - Quantization step: Q = clamp(round(W * inv_s + z), q_min, q_max)
          - Parameter update:
            * Asymmetric: scale is fixed (min-max init); only z is updated via
              closed-form `z = mean(Q - W * inv_s)` (per the paper).
            * Symmetric: z = None; scale is updated via `s = sum(W*Q) / sum(Q²)`.

        For the asymmetric case the zero point z is optimized as a continuous float during
        iterations (giving better reconstruction than integer-only search), then rounded and
        clipped to the valid integer range before being returned. This ensures consistency
        between quantization (which uses the returned z) and dequantization (which loads z
        as a stored integer, e.g. uint4).

        :param weight: Weight tensor in float32.
        :param config: Weight compression configuration.
        :param reduction_axes: Reduction axes for the weight tensor.
        :return: Tuple of (scale, zero_point). zero_point is an integer-valued float32
            tensor for asymmetric mode, None for symmetric mode.
        """
        group_size = config.group_size
        group_reduction_axes = reduction_axes

        # Reshape weights for grouped quantization when a group size is specified.
        if group_size != -1:
            weight, group_reduction_axes = reshape_weight_for_grouped_quantization(
                weight, reduction_axes, group_size
            )

        # Number of elements along the reduction axis (i.e. per group).
        if isinstance(group_reduction_axes, int):
            n = weight.shape[group_reduction_axes]
        else:
            n = 1
            for ax in group_reduction_axes:
                n *= weight.shape[ax]

        num_bits = config.num_bits
        is_asym = config.is_asym_mode
        level_low = 0 if is_asym else -(2 ** (num_bits - 1))
        level_high = 2**num_bits - 1 if is_asym else 2 ** (num_bits - 1) - 1

        eps = fns.finfo(weight).eps

        # Initialize with standard min-max quantization parameters.
        scale, zero_point = calculate_integer_quantization_params(weight, group_reduction_axes, config)

        # Cast integer zero point to float32 so arithmetic below is uniform.
        if zero_point is not None:
            zero_point = fns.astype(zero_point, TensorDataType.float32)

        # Pre-compute inv_scale once; scale is fixed for asymmetric iterations.
        inv_scale = 1.0 / fns.where(fns.abs(scale) < eps, eps, scale)

        for _ in range(self._num_iterations):
            # Quantization step: Q = clamp(round(W * inv_s + z), q_min, q_max)
            q_float = weight * inv_scale
            if zero_point is not None:
                q_float = q_float + zero_point
            q_float = fns.round(q_float)
            q_float = fns.clip(q_float, level_low, level_high)

            if is_asym:
                # Asymmetric: fix scale, update zero_point only (per the paper).
                # Minimizing ||W - s*(Q - z)||² w.r.t. z gives:
                #   z = mean(Q - W/s) = sum(Q - W*inv_s) / n
                zero_point = fns.sum(q_float - weight * inv_scale, axis=group_reduction_axes, keepdims=True)
                zero_point = zero_point / n
            else:
                # Symmetric OLS update for scale: minimize ||W - s*Q||².
                #   s = sum(W*Q) / sum(Q²)
                sum_qw = fns.sum(q_float * weight, axis=group_reduction_axes, keepdims=True)
                sum_qq = fns.sum(q_float * q_float, axis=group_reduction_axes, keepdims=True)
                denom = fns.where(fns.abs(sum_qq) < eps, eps, sum_qq)
                scale = sum_qw / denom
                scale = fns.where(fns.abs(scale) < eps, eps, scale)
                inv_scale = 1.0 / scale

        # Round and clip zero_point to the valid integer range so that quantization
        # and dequantization (which stores zp as uint4) use the exact same value.
        if zero_point is not None:
            zero_point = self._round_zero_point(zero_point, level_low, level_high)

        return scale, zero_point

    @staticmethod
    def _round_zero_point(zero_point: Tensor, level_low: int, level_high: int) -> Tensor:
        """
        Rounds the float zero_point to the nearest integer and clips it to the valid quantization range.

        HQQ optimizes z as a continuous value during iterations, but the OV backend stores
        zero_point as integer (uint4 for INT4). To ensure that quantization and dequantization
        use the same z, the final float z is rounded and clipped before returning.

        :param zero_point: Float zero_point tensor from HQQ iterations.
        :param level_low: Minimum valid zero_point value.
        :param level_high: Maximum valid zero_point value.
        :return: Rounded and clipped zero_point.
        """
        return fns.clip(fns.round(zero_point), level_low, level_high)
