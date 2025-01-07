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

import math
from typing import Dict, List, Optional, Tuple, TypeVar

import nncf
from nncf import Dataset
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.layerwise.engine import LayerwiseEngine
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.scale_estimation import ScaleEstimation
from nncf.quantization.algorithms.weight_compression.weight_lowering import calculate_integer_quantization_params
from nncf.quantization.algorithms.weight_compression.weight_lowering import calculate_nf4_scale
from nncf.quantization.algorithms.weight_compression.weight_lowering import calculate_quantized_weight
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_int_dequantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_nf4_dequantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_nf4_quantization
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
            raise nncf.UnsupportedBackendError(
                f"Cannot return backend-specific entity because {self._backend.value} is not supported!"
            )

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        dataset: Dataset,
        weight_compression_parameters: List[WeightCompressionParameters],
        statistic_points: Optional[StatisticPointsContainer] = None,
        backend_entity: Optional[WeightCompressionAlgoBackend] = None,
    ) -> Tuple[TModel, Dict[str, Tensor], Dict[str, Tensor]]:
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

        scales = {}
        zero_points = {}

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
        for node, inputs in track(target_node_iterator, total=len(target_nodes), description="Applying GPTQ"):
            wc_params = target_nodes_wc_params_map[node]
            if wc_params.compression_config.mode in [
                CompressWeightsMode.INT8_ASYM,
                CompressWeightsMode.INT8_SYM,
            ]:
                continue
            _, input_tensors = next(iter(inputs.items()))
            hessian = self._calculate_hessian(node, input_tensors)
            scale, zero_point = self._quantize_weights(model, graph, wc_params, hessian, input_tensors)
            scales[wc_params.weight_name] = scale
            zero_points[wc_params.weight_name] = zero_point

        return model, scales, zero_points

    def get_statistic_points(
        self,
        model: TModel,
        graph: NNCFGraph,
        target_nodes: List[NNCFNode],
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

    def _calculate_hessian(self, node: NNCFNode, inputs: List[Tensor]) -> Tensor:
        """
        Calculates the Hessian matrix for the given node and inputs.

        :param node: The target node for Hessian calculation.
        :param inputs: List of input tensors.
        :return: The Hessian matrix as a tensor.
        """
        nsamples = 0

        if node.metatype in self._backend_entity.convolution_metatypes:
            raise nncf.UnsupportedModelError("Convolution metatypes are not supported")
        if node.layer_attributes.input_attributes["transpose"]:
            raise nncf.UnsupportedModelError("Transposed input is not supported")

        hessian = fns.zeros(
            (inputs[0].shape[-1], inputs[0].shape[-1]), backend=inputs[0].backend, dtype=TensorDataType.float32
        )

        for inp in inputs:
            batch_size = 1 if len(inp.shape) == 2 else inp.shape[0]
            if node.metatype in self._backend_entity.matmul_metatypes:
                if len(inp.shape) == 3:
                    inp = inp.reshape((-1, inp.shape[-1]))
                inp = fns.transpose(inp)
            hessian *= nsamples / (nsamples + batch_size)
            nsamples += batch_size
            inp = fns.astype(inp, TensorDataType.float32) * math.sqrt(2 / nsamples)
            hessian += fns.matmul(inp, fns.transpose(inp))

        return hessian

    def _quantize_weights(
        self,
        model: TModel,
        graph: NNCFGraph,
        wc_params: WeightCompressionParameters,
        hessian: Tensor,
        inputs: List[Tensor],
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
            raise RuntimeError("Convolution metatypes are not supported")
        if not wc_params.node_with_weight.layer_attributes.constant_attributes[wc_params.weight_port_id]["transpose"]:
            raise RuntimeError("Transpose is not supported")

        weight_tensor = self._backend_entity.get_weight(
            wc_params.node_with_weight, wc_params.weight_port_id, model, graph
        )
        weight_tensor = fns.astype(weight_tensor, TensorDataType.float32)

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
        block_compression_config = WeightCompressionConfig(mode=wc_params.compression_config.mode)

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
                    if block_compression_config.mode == CompressWeightsMode.NF4:
                        scale = calculate_nf4_scale(weight_tensor[:, (i1 + i) : (i1 + i + group_size)], reduction_axes)
                        scales.append(scale)
                    else:
                        if self._scale_estimation and block_compression_config.num_bits == 4:
                            activations = [inp[..., (i1 + i) : (i1 + i + group_size)] for inp in inputs]
                            wc_statistics = ScaleEstimation.activations_to_wc_statistics(activations)
                            scale, zero_point = ScaleEstimation.calculate_quantization_params(
                                self._backend_entity,
                                wc_statistics,
                                weight_tensor[:, (i1 + i) : (i1 + i + group_size)],
                                reduction_axes,
                                wc_params.compression_config,
                            )
                            scales.append(scale.squeeze(axis=1))
                            zero_points.append(zero_point if zero_point is None else zero_point.squeeze(axis=1))
                        else:
                            scale, zero_point = calculate_integer_quantization_params(
                                weight_tensor[:, (i1 + i) : (i1 + i + group_size)],
                                reduction_axes,
                                block_compression_config,
                            )
                            scales.append(scale)
                            zero_points.append(zero_point)
                if block_compression_config.mode == CompressWeightsMode.NF4:
                    compressed_weights = do_nf4_quantization(
                        fns.unsqueeze(weight_col, 1), scales[-1], is_normalized_weight=False
                    )
                    quantized_col = do_nf4_dequantization(compressed_weights, scales[-1], reduction_axis=-1)
                else:
                    compressed_weights = calculate_quantized_weight(
                        fns.unsqueeze(weight_col, 1), block_compression_config, scales[-1], zero_points[-1]
                    )
                    quantized_col = do_int_dequantization(compressed_weights, scales[-1], zero_points[-1])
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
        self._backend_entity.set_weight(
            wc_params.node_with_weight, wc_params.weight_port_id, model, graph, quantized_tensor
        )

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
        return scales, zero_points
