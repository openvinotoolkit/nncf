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
    to jointly find optimal scale and zero-point parameters, producing floating-point
    zero points for asymmetric quantization.

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
          - Quantization step: Q = clamp(round(W / s + z), q_min, q_max)
          - Parameter update:
            * Asymmetric (W ≈ s * (Q - z)): joint closed-form least-squares for s and z.
            * Symmetric (W ≈ s * Q): closed-form update for s alone.

        For the asymmetric case the zero point z is float-valued (not rounded to an integer),
        which gives HQQ better reconstruction quality than standard min-max initialization.

        :param weight: Weight tensor in float32.
        :param config: Weight compression configuration.
        :param reduction_axes: Reduction axes for the weight tensor.
        :return: Tuple of (scale, zero_point). zero_point is float for asymmetric mode,
            None for symmetric mode.
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

        for _ in range(self._num_iterations):
            # Quantization step: Q = clamp(round(W / s + z), q_min, q_max)
            q_float = weight / scale
            if zero_point is not None:
                q_float = q_float + zero_point
            q_float = fns.round(q_float)
            q_float = fns.clip(q_float, level_low, level_high)

            if is_asym:
                # Asymmetric least-squares update for (s, z): minimize ||W - s*(Q - z)||^2.
                # Letting b = s*z, normal equations give:
                #   det = n * sum_QQ - sum_Q^2
                #   s   = (n * sum_QW - sum_Q * sum_W) / det
                #   z   = (sum_Q * sum_QW - sum_QQ * sum_W) / (det * s)
                sum_q = fns.sum(q_float, axis=group_reduction_axes, keepdims=True)
                sum_w = fns.sum(weight, axis=group_reduction_axes, keepdims=True)
                sum_qq = fns.sum(q_float * q_float, axis=group_reduction_axes, keepdims=True)
                sum_qw = fns.sum(q_float * weight, axis=group_reduction_axes, keepdims=True)

                det = n * sum_qq - sum_q * sum_q
                safe_det = fns.where(fns.abs(det) < eps, eps, det)

                new_scale = (n * sum_qw - sum_q * sum_w) / safe_det
                new_scale = fns.where(fns.abs(new_scale) < eps, eps, new_scale)
                new_zero_point = (sum_q * sum_qw - sum_qq * sum_w) / (safe_det * new_scale)

                scale = new_scale
                zero_point = new_zero_point

            else:
                # Symmetric least-squares update for s: minimize ||W - s*Q||^2.
                #   s = sum(W*Q) / sum(Q^2)
                sum_qw = fns.sum(q_float * weight, axis=group_reduction_axes, keepdims=True)
                sum_qq = fns.sum(q_float * q_float, axis=group_reduction_axes, keepdims=True)
                denom = fns.where(fns.abs(sum_qq) < eps, eps, sum_qq)
                scale = sum_qw / denom
                scale = fns.where(fns.abs(scale) < eps, eps, scale)

        return scale, zero_point
