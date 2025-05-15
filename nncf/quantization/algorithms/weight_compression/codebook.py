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
from typing import Any, Optional, TypeVar

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.logging.track_progress import track
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.common import Codebook
from nncf.quantization.algorithms.weight_compression.common import CompressedWeight
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.handle_errors import handle_invalid_group_size_error
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_float_quantization
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor import functions as fns

TModel = TypeVar("TModel")


class CodebookCompression:
    """
    Codebook estimation algorithm implementation.
    """

    def __init__(
        self,
        initial_codebook: Tensor,
        dst_type: Any,
    ):
        """
        :param initial_codebook: codebook for compression.
        """
        super().__init__()
        self._initial_codebook = initial_codebook
        self._dst_type = dst_type

    @property
    def available_backends(self) -> list[BackendType]:
        return [BackendType.OPENVINO]

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.weight_compression.openvino_backend import OVWeightCompressionAlgoBackend

            self._backend_entity = OVWeightCompressionAlgoBackend(model)
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
        :param backend_entity: Weight compression algorithm backend.
        :return: Two dictionaries for estimated scales and zero points for each weight name.
        """
        self._backend_entity = backend_entity
        if self._backend_entity is None:
            self._set_backend_entity(model)

        res = {}
        invalid_node_names = []
        first_caught_error = None
        for wp in track(all_weight_params, description="Applying Codebook Compression"):
            if wp.compression_config.mode != CompressWeightsMode.CODEBOOK:
                continue
            weight_name = wp.weight_name
            config = wp.compression_config

            weight_data = self._backend_entity.get_weight_names_and_port_ids(wp.node_with_weight, graph)
            if len(weight_data) != 1:  # not supported by the algorithm
                continue
            _, weight_port_id = weight_data[0]

            weight = self._backend_entity.get_weight(wp.node_with_weight, weight_port_id, model, graph)

            try:
                indexes, scale, codebook = self.calculate_quantization_params(weight, wp.reduction_axes, config)
                res[weight_name] = CompressedWeight(indexes, scale, None, Codebook(codebook, self._dst_type))
            except nncf.InvalidGroupSizeError as error:
                first_caught_error = error
                invalid_node_names.append(wp.node_with_weight.node_name)

        if first_caught_error:
            handle_invalid_group_size_error(first_caught_error, invalid_node_names)

        return res

    def calculate_quantization_params(
        self,
        weight: Tensor,
        reduction_axes: tuple[int, ...],
        config: WeightCompressionConfig,
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
        :return: A tensor containing the calculated quantization scales and zero points if applicable.
        """
        reduction_axis = reduction_axes[0]

        weight = weight.astype(TensorDataType.float32)

        codebook = fns.tensor(
            self._initial_codebook, backend=weight.backend, dtype=TensorDataType.float32, device=weight.device
        )

        if reduction_axis == 0:
            weight = fns.transpose(weight)
            reduction_axis = 1

        group_size = config.group_size if config.group_size != -1 else weight.shape[reduction_axis]
        cur_config = deepcopy(config)
        cur_config.group_size = group_size

        max_val = fns.max(fns.abs(codebook))
        norm_weight, scale = do_float_quantization(weight, cur_config, reduction_axis, max_val=max_val)

        orig_shape = norm_weight.shape

        norm_weight = fns.unsqueeze(norm_weight.flatten(), 1)

        dist = (norm_weight - fns.unsqueeze(codebook, 0)) ** 2

        indexes = dist.data.argmin(-1)
        indexes = fns.reshape(indexes, orig_shape)

        return indexes, scale, codebook
