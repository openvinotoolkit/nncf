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

import math
from typing import Optional, TypeVar

import numpy as np

import nncf
from nncf import Dataset
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.logging import nncf_logger
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.layerwise.engine import LayerwiseEngine
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.parameters import CompressedWeight
from nncf.quantization.algorithms.weight_compression.scale_estimation import ScaleEstimation
from nncf.quantization.algorithms.weight_compression.weight_lowering import calculate_float_quantization_params
from nncf.quantization.algorithms.weight_compression.weight_lowering import calculate_integer_quantization_params
from nncf.quantization.algorithms.weight_compression.weight_lowering import float_quantize_dequantize_weight
from nncf.quantization.algorithms.weight_compression.weight_lowering import integer_quantize_dequantize_weight
from nncf.tensor import Tensor
from nncf.tensor import functions as fns
from nncf.tensor.definitions import TensorDataType

TModel = TypeVar("TModel")


class GPTQ:
    """
    GPTQ algorithm implementation
    """

    def __init__(
        self, damp_percent: float = 0.1, block_size: int = 128, subset_size: int = 128, scale_estimation: bool = False
    ):
        """
        :param damp_percent: The percent of the average Hessian diagonal to use for dampening,
            recommended value is 0.1.
        :param block_size: The size of the blocks used during quantization. Defaults to 128.
        :param subset_size: Number of data samples to calculate Hessian. Defaults to 128.
        """
        self._damp_percent = damp_percent
        self._block_size = block_size
        self._subset_size = subset_size
        self._scale_estimation = scale_estimation
        self._backend = None
        self._backend_entity = None

        self._layerwise_engine = LayerwiseEngine(subset_size=self._subset_size)

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backend-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """
        self._backend = get_backend(model)
        if self._backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.weight_compression.openvino_backend import OVWeightCompressionAlgoBackend

            self._backend_entity = OVWeightCompressionAlgoBackend(model)
        else:
            msg = f"Cannot return backend-specific entity because {self._backend.value} is not supported!"
            raise nncf.UnsupportedBackendError(msg)

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        dataset: Dataset,
        weight_compression_parameters: list[WeightCompressionParameters],
        statistic_points: Optional[StatisticPointsContainer] = None,
        backend_entity: Optional[WeightCompressionAlgoBackend] = None,
    ) -> tuple[TModel, dict[str, CompressedWeight]]:
        """
        Applies the GPTQ algorithm to quantize the weights of the given model.

        :param model: The model to quantize.
        :param graph: The model graph.
        :param dataset: The dataset to use for quantization.
        :param weight_compression_parameters: Parameters for weight compression.
        :param statistic_points: Optional container for statistic points.
        :param backend_entity: Weight compression algorithm backend.
        :return: The quantized model and its scales and zero points.
        """
        self._backend_entity = backend_entity
        if self._backend_entity is None:
            self._set_backend_entity(model)

        res = {}

        target_nodes = []
        target_nodes_wc_params_map = {}
        matmul_metatypes = self._backend_entity.matmul_metatypes
        for wc_params in weight_compression_parameters:
            if wc_params.node_with_weight.metatype in matmul_metatypes:
                target_nodes.append(wc_params.node_with_weight)
                target_nodes_wc_params_map[wc_params.node_with_weight] = wc_params

        target_node_iterator = self._layerwise_engine.create_iterator_through_target_nodes(
            model, graph, target_nodes, dataset, statistic_points
        )

        description = "Applying GPTQ"
        if self._scale_estimation:
            description += " with Scale Estimation"
        for node, inputs in track(target_node_iterator, total=len(target_nodes), description=description):
            wc_params = target_nodes_wc_params_map[node]
            if wc_params.compression_config.mode in [
                CompressWeightsMode.INT8_ASYM,
                CompressWeightsMode.INT8_SYM,
            ]:
                continue

            if self._backend_entity.matmul_has_transposed_activations(wc_params.node_with_weight, graph):
                msg = "Transposed activations are not supported yet for the GPTQ algorithm"
                raise nncf.UnsupportedModelError(msg)

            _, input_tensors = next(iter(inputs.items()))
            weight_tensor = self._backend_entity.get_weight(
                wc_params.node_with_weight, wc_params.weight_port_id, model, graph
            )
            weight_tensor = fns.astype(weight_tensor, TensorDataType.float32)

            is_3d_weight = len(weight_tensor.shape) == 3

            node = wc_params.node_with_weight
            hessian = self._calculate_hessian(node, input_tensors, is_3d_weight)
            weight_tensor = fns.unsqueeze(weight_tensor, 0) if not is_3d_weight else weight_tensor
            scales = []
            zero_points = []
            weights = []
            for batch_idx in range(hessian.shape[0]):
                batch_hessian = hessian[batch_idx]
                batch_weight = weight_tensor[batch_idx]
                reduction_axes = wc_params.reduction_axes
                assert len(reduction_axes) == 1, "2D reduction axes is not currently supported in GPTQ"
                wc_params.reduction_axes = (reduction_axes[0] - 1,) if is_3d_weight else reduction_axes
                input_tensor = input_tensors[batch_idx] if is_3d_weight else input_tensors
                batch_quantized_weight, batch_scale, batch_zero_point = self._quantize_weights(
                    wc_params, batch_hessian, batch_weight, input_tensor
                )
                wc_params.reduction_axes = reduction_axes
                weights.append(batch_quantized_weight)
                scales.append(batch_scale)
                zero_points.append(batch_zero_point)
            scale = fns.stack(scales, axis=0) if is_3d_weight else scales[0]
            zero_point = fns.stack(zero_points, axis=0) if is_3d_weight and None not in zero_points else zero_points[0]
            weight = fns.stack(weights, axis=0) if is_3d_weight else weights[0]
            self._backend_entity.set_weight(wc_params.node_with_weight, wc_params.weight_port_id, model, graph, weight)
            res[wc_params.weight_name] = CompressedWeight(None, scale, zero_point, None)

        return model, res

    def get_statistic_points(
        self,
        model: TModel,
        graph: NNCFGraph,
        target_nodes: list[NNCFNode],
        backend_entity: Optional[WeightCompressionAlgoBackend] = None,
    ) -> StatisticPointsContainer:
        """
        Returns statistic points, for which StatisticsCollector should collect statistics.

        :param model: The model for statistics collection.
        :param graph: The model graph.
        :param backend_entity: Weight compression algorithm backend.
        :return: Statistic points, for which StatisticsCollector should collect statistics.
        """
        self._backend_entity = backend_entity
        if self._backend_entity is None:
            self._set_backend_entity(model)

        matmul_metatypes = self._backend_entity.matmul_metatypes
        filtered_nodes = []
        for node in target_nodes:
            if node.metatype in matmul_metatypes:
                filtered_nodes.append(node)

        return self._layerwise_engine.get_statistic_points(model, graph, filtered_nodes)

    def _calculate_hessian(self, node: NNCFNode, inputs: list[Tensor], is_3d_weight: bool = False) -> Tensor:
        """
        Calculates the Hessian matrix for the given node and inputs.

        :param node: The target node for Hessian calculation.
        :param inputs: List of input tensors.
        :return: The Hessian matrix as a tensor.
        """
        nsamples = 0

        if node.metatype in self._backend_entity.convolution_metatypes:
            msg = "Convolution metatypes are not supported"
            raise nncf.UnsupportedModelError(msg)
        if node.layer_attributes.input_attributes["transpose"]:
            msg = "Transposed input is not supported"
            raise nncf.UnsupportedModelError(msg)
        # Make hessian 3D. Such that for 2D weights it is only 1 batch and can be squeezed later.
        # For 3D weights this dimension matches the weights dimensions
        hessian_batch = 1 if not is_3d_weight else np.multiply.reduce(inputs[0].shape[:-2])
        hessian = fns.zeros(
            (hessian_batch, inputs[0].shape[-1], inputs[0].shape[-1]),
            backend=inputs[0].backend,
            dtype=TensorDataType.float32,
        )

        for inp in inputs:
            is_3d_act = len(inp.shape) == 3
            # For 3D weights case, batch size will always be 1. Each "batch"/expert of the activation is treated as
            # single 2D matmuls
            batch_size = 1 if not is_3d_act and not is_3d_weight else inp.shape[0]
            if node.metatype in self._backend_entity.matmul_metatypes:
                # For 3D act + 2D weight case we should reshape activation to 2D to match weight
                # For 3D act + 3D weight it should remain in 3D and the last 2 dimensions should be activation per
                # batch/0-th dimension
                if is_3d_act and not is_3d_weight:
                    inp = inp.reshape((-1, inp.shape[-1]))
                inp = fns.moveaxis(inp, -1, -2)
            hessian *= nsamples / (nsamples + batch_size)
            nsamples += batch_size
            inp = fns.astype(inp, TensorDataType.float32) * math.sqrt(2 / nsamples)
            hessian += fns.matmul(inp, fns.moveaxis(inp, -1, -2))

        return hessian

    def _quantize_weights(
        self,
        wc_params: WeightCompressionParameters,
        hessian: Tensor,
        weight_tensor: Tensor,
        inputs: list[Tensor],
    ):
        """
        Quantizes the weights of the model based on the calculated Hessian matrix.

        :param model: The model to quantize.
        :param graph: The model graph.
        :param wc_params: Parameters for weight compression.
        :param hessian: The Hessian matrix.
        :return: Scales and zero points used for quantization.
        """
        if wc_params.node_with_weight.metatype in self._backend_entity.convolution_metatypes:
            msg = "Convolution metatypes are not supported"
            raise RuntimeError(msg)
        if not wc_params.node_with_weight.layer_attributes.constant_attributes[wc_params.weight_port_id]["transpose"]:
            msg = "Transpose is not supported"
            raise RuntimeError(msg)

        if len(hessian.shape) == 3 and hessian.shape[0] == 1:
            hessian = fns.squeeze(hessian)
            msg = "The hessian passed to quantize_weights is 3D. It should be 2D"
            nncf_logger.warning(msg=msg)
        assert len(hessian.shape) == 2, "Hessian should be 2D"

        dead_indices = fns.diag(hessian) == 0
        hessian[dead_indices, dead_indices] = 1
        weight_tensor[:, dead_indices] = 0

        scales = []
        zero_points = []

        losses = fns.zeros_like(weight_tensor)
        quantized_tensor = fns.zeros_like(weight_tensor)

        columns = hessian.shape[0]
        group_size = (
            wc_params.compression_config.group_size
            if wc_params.compression_config.group_size != -1
            else weight_tensor.shape[1]
        )
        reduction_axes = wc_params.reduction_axes
        block_compression_config = WeightCompressionConfig(
            mode=wc_params.compression_config.mode, codebook_values=wc_params.compression_config.codebook_values
        )

        damp = self._damp_percent * fns.mean(fns.diag(hessian))
        diag_indices = fns.arange(columns, backend=hessian.backend, device=hessian.device)
        hessian[diag_indices, diag_indices] += damp
        hessian = fns.linalg.cholesky(hessian)
        hessian = fns.linalg.cholesky_inverse(hessian)
        hessian = fns.linalg.cholesky(hessian, upper=True)
        hessian_inv = hessian

        for i1 in range(0, columns, self._block_size):
            i2 = min(i1 + self._block_size, columns)
            count = i2 - i1

            weight_block = weight_tensor[:, i1:i2].clone()
            quantized_block = fns.zeros_like(weight_block)
            error_block = fns.zeros_like(weight_block)
            loss_block = fns.zeros_like(weight_block)
            hessian_inv_block = hessian_inv[i1:i2, i1:i2]

            for i in range(count):
                weight_col = weight_block[:, i]
                hessian_diag_val = hessian_inv_block[i, i]

                if (i1 + i) % group_size == 0:
                    if not block_compression_config.is_integer:
                        scale = calculate_float_quantization_params(
                            weight_tensor[:, (i1 + i) : (i1 + i + group_size)], reduction_axes, block_compression_config
                        )
                        scales.append(scale)
                    else:
                        if self._scale_estimation and block_compression_config.num_bits == 4:
                            activations = [inp[..., (i1 + i) : (i1 + i + group_size)] for inp in inputs]
                            # TODO(anazir): Make it work for 3D weights
                            wc_statistics = ScaleEstimation.activations_to_wc_statistics(activations)
                            scale, zero_point = ScaleEstimation.calculate_quantization_params(
                                wc_statistics,
                                weight_tensor[:, (i1 + i) : (i1 + i + group_size)],
                                reduction_axes,
                                block_compression_config,
                            )
                        else:
                            scale, zero_point = calculate_integer_quantization_params(
                                weight_tensor[:, (i1 + i) : (i1 + i + group_size)],
                                reduction_axes,
                                block_compression_config,
                            )
                        scales.append(scale)
                        zero_points.append(zero_point)

                if not block_compression_config.is_integer:
                    quantized_col = float_quantize_dequantize_weight(
                        fns.unsqueeze(weight_col, 1),
                        block_compression_config,
                        precomputed_scale=scales[-1],
                    )
                else:
                    quantized_col = integer_quantize_dequantize_weight(
                        fns.unsqueeze(weight_col, 1),
                        block_compression_config,
                        precomputed_scale=scales[-1],
                        precomputed_zero_point=zero_points[-1],
                    )
                quantized_col = fns.flatten(quantized_col)
                quantized_block[:, i] = quantized_col
                loss_block[:, i] = (weight_col - quantized_col) ** 2 / hessian_diag_val**2

                error_col = (weight_col - quantized_col) / hessian_diag_val
                weight_block[:, i:] -= fns.matmul(
                    fns.unsqueeze(error_col, 1), fns.unsqueeze(hessian_inv_block[i, i:], 0)
                )
                error_block[:, i] = error_col

            quantized_tensor[:, i1:i2] = quantized_block
            losses[:, i1:i2] = loss_block / 2

            weight_tensor[:, i2:] -= fns.matmul(error_block, hessian_inv[i1:i2, i2:])

        quantized_tensor = quantized_tensor.reshape(weight_tensor.shape).astype(weight_tensor.dtype)

        scales = fns.stack(scales, axis=1)
        if wc_params.compression_config.group_size == -1:
            scales = fns.squeeze(scales, axis=-1)
        if wc_params.compression_config.mode in [
            CompressWeightsMode.INT8_ASYM,
            CompressWeightsMode.INT4_ASYM,
        ]:
            zero_points = fns.stack(zero_points, axis=1)
            if wc_params.compression_config.group_size == -1:
                zero_points = fns.squeeze(zero_points, axis=-1)
        else:
            zero_points = None
        return weight_tensor, scales, zero_points
