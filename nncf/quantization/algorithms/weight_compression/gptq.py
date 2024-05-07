# Copyright (c) 2024 Intel Corporation
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
from typing import List, Optional, TypeVar

import nncf
from nncf import Dataset
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.experimental.tensor import Tensor
from nncf.experimental.tensor import functions as fns
from nncf.experimental.tensor.definitions import TensorDataType
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.layerwise.engine import LayerwiseEngine
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.weight_lowering import calculate_integer_quantization_params
from nncf.quantization.algorithms.weight_compression.weight_lowering import calculate_quantized_weight
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_dequantization

TModel = TypeVar("TModel")


class GPTQ:
    """
    GPTQ algorithm implementation
    """

    def __init__(
        self,
        damp_percent: float = 0.1,
        block_size: int = 128,
        subset_size: int = 128,
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
        self._backend = None
        self._backend_entity = None

        self._layerwise_engine = LayerwiseEngine(subset_size=self._subset_size)

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """
        self._backend = get_backend(model)
        if self._backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.weight_compression.openvino_backend import OVWeightCompressionAlgoBackend

            self._backend_entity = OVWeightCompressionAlgoBackend(model)
        else:
            raise nncf.UnsupportedBackendError(
                "Cannot return backend-specific entity because {} is not supported!".format(self._backend.value)
            )

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        dataset: Dataset,
        weight_compression_parameters: List[WeightCompressionParameters],
        statistics_points: Optional[StatisticPointsContainer] = None,
        backend_entity: Optional[WeightCompressionAlgoBackend] = None,
    ) -> None:
        """
        Applies the GPTQ algorithm to quantize the weights of the given model.

        :param model: The model to quantize.
        :param graph: The model graph.
        :param dataset: The dataset to use for quantization.
        :param weight_compression_parameters: Parameters for weight compression.
        :param statistics_points: Optional container for statistic points.
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
            model, graph, target_nodes, dataset, statistics_points
        )
        for node, inputs in track(target_node_iterator, total=len(target_nodes), description="Applying GPTQ"):
            wc_params = target_nodes_wc_params_map[node]
            if wc_params.compression_config.group_size == -1:
                continue
            assert len(inputs) == 1
            _, input_tensors = next(iter(inputs.items()))
            H = self._calculate_hessian(node, input_tensors)
            scale, zero_point = self._quantize_weights(model, graph, wc_params, H)
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
            raise RuntimeError("Convolution metatypes are not supported")
        # TODO: backend specific code
        if node.layer_attributes.input_attributes["transpose"]:
            raise RuntimeError("Transpose is not supported")

        # TODO: workaround create zeros tensor
        # check weight transpose
        H = fns.zeros((inputs[0].shape[-1], inputs[0].shape[-1]), backend=inputs[0].backend)

        for inp in inputs:
            batch_size = 1 if len(inp.shape) == 2 else inp.shape[0]

            if node.metatype in self._backend_entity.matmul_metatypes:
                # check transpose
                if len(inp.shape) == 3:
                    inp = inp.reshape((-1, inp.shape[-1]))
                inp = fns.transpose(inp)
            if node.metatype in self._backend_entity.convolution_metatypes:
                pass
                # TODO: Require implementaiton on numpy
                # unfold = nn.Unfold(
                #     self.layer.kernel_size,
                #     dilation=self.layer.dilation,
                #     padding=self.layer.padding,
                #     stride=self.layer.stride,
                # )
                # inp = unfold(inp)
                # inp = inp.permute([1, 0, 2])
                # inp = inp.flatten(1)
            H *= nsamples / (nsamples + batch_size)
            nsamples += batch_size
            # inp = inp.float()
            inp = fns.astype(inp, TensorDataType.float32) * math.sqrt(2 / nsamples)
            # self.H += 2 / self.nsamples * inp.matmul(inp.t())
            H += fns.matmul(inp, fns.transpose(inp))

        return H

    def _quantize_weights(self, model: TModel, graph: NNCFGraph, wc_params: WeightCompressionParameters, H: Tensor):
        """
        Quantizes the weights of the model based on the calculated Hessian matrix.

        :param model: The model to quantize.
        :param graph: The model graph.
        :param wc_params: Parameters for weight compression.
        :param H: The Hessian matrix.
        :return: Scales and zero points used for quantization.
        """
        if wc_params.node_with_weight.metatype in self._backend_entity.convolution_metatypes:
            raise RuntimeError("Convolution metatypes are not supported")
        # TODO: backend specific code
        if not wc_params.node_with_weight.layer_attributes.constant_attributes[wc_params.weight_port_id]["transpose"]:
            raise RuntimeError("Transpose is not supported")

        W = self._backend_entity.get_weight(wc_params.node_with_weight, wc_params.weight_port_id, model, graph)

        # if isinstance(self.layer, nn.Conv2d):
        #    W = W.flatten(1)
        # if isinstance(self.layer, transformers.Conv1D):
        #    W = W.t()
        # W = W.float()
        W = fns.astype(W, TensorDataType.float32)

        # TODO: handle it if group size is not used.
        # if not self.quantizer.ready():
        #     self.quantizer.find_params(W, weight=True)

        dead = fns.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        scales = []
        zero_points = []

        Losses = fns.zeros_like(W)
        Q = fns.zeros_like(W)

        columns = H.shape[0]
        group_size = wc_params.compression_config.group_size
        reduction_axes = wc_params.reduction_axes
        block_compression_config = WeightCompressionConfig(mode=wc_params.compression_config.mode)

        damp = self._damp_percent * fns.mean(fns.diag(H))
        # TODO: workaround to create arange tensor
        diag = fns.arange(columns, backend=H.backend, device=H.device)
        H[diag, diag] += damp
        H = fns.linalg.cholesky(H)
        H = fns.linalg.cholesky_inverse(H)
        H = fns.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, columns, self._block_size):
            i2 = min(i1 + self._block_size, columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = fns.zeros_like(W1)
            Err1 = fns.zeros_like(W1)
            Losses1 = fns.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if group_size != -1 and (i1 + i) % group_size == 0:
                    scale, zero_point = calculate_integer_quantization_params(
                        W[:, (i1 + i) : (i1 + i + group_size)], reduction_axes, block_compression_config
                    )
                    scales.append(scale)
                    zero_points.append(zero_point)

                compressed_weights = calculate_quantized_weight(
                    fns.unsqueeze(w, 1), scales[-1], zero_points[-1], block_compression_config
                )
                q = do_dequantization(compressed_weights, scales[-1], zero_points[-1])
                q = fns.flatten(q)
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= fns.matmul(fns.unsqueeze(err1, 1), fns.unsqueeze(Hinv1[i, i:], 0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= fns.matmul(Err1, Hinv[i1:i2, i2:])

        # TODO support Conv1D
        # if isinstance(self.layer, transformers.Conv1D):
        #     Q = Q.t()

        Q = Q.reshape(W.shape).astype(W.dtype)
        self._backend_entity.set_weight(wc_params.node_with_weight, wc_params.weight_port_id, model, graph, Q)

        # TODO: support group_size = -1
        # if scale == []:
        #     scale.append(self.quantizer.scale)
        #     zero.append(self.quantizer.zero)

        scales = fns.stack(scales, axis=1)
        if wc_params.compression_config.mode in [
            CompressWeightsMode.INT8_SYM,
            CompressWeightsMode.INT4_SYM,
        ]:
            zero_points = fns.squeeze(zero_points[0])
        else:
            zero_points = fns.stack(zero_points, axis=1)
        return scales, zero_points
