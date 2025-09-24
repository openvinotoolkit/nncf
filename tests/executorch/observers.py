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


from abc import ABC
from abc import abstractmethod
from typing import Optional

import torch
from torch.ao.quantization.observer import ObserverBase

from nncf.experimental.torch.fx.node_utils import get_tensor_constant_from_node
from nncf.experimental.torch.fx.transformations import constant_update
from nncf.experimental.torch.fx.transformations import module_insertion
from nncf.experimental.torch.fx.transformations import node_removal
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_integer_quantization
from nncf.tensor.tensor import Tensor as NNCFTensor
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.commands import TargetType
from nncf.torch.quantization.layers import BaseWeightsDecompressor
from nncf.torch.quantization.layers import INT4AsymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT4SymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT8AsymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT8SymmetricWeightsDecompressor


class WeightObserverBase(ObserverBase, ABC):
    """
    Base implementation of an NNCF observer that defines the rules for compressing layer
    weights into the OpenVINO representation.
    """

    def __init__(
        self,
        wc_param: WeightCompressionParameters,
        dtype: torch.dtype,
        **kwargs,
    ) -> None:
        """
        :param wc_param: Weight compression parameters container.
        :param dtype: target dtype for the quantization.
        """
        super().__init__(dtype=dtype, is_dynamic=False)
        self._wc_param = wc_param

    def calculate_qparams(
        self,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Calculates quantization parameters: quantized weight, quantization scale and quantization zero point.

        :param weight: FP weight to be used for calculating qparams.
        :return: A tuple containing the quantized weight, quantization scale and quantization zero point.
        """
        wc_param = self._wc_param
        wc_config = wc_param.compression_config
        reduction_axes = wc_param.reduction_axes
        q_weight, scale, zp = do_integer_quantization(NNCFTensor(weight), wc_config, reduction_axes=reduction_axes)
        zp = zp.data if zp is not None else None
        return q_weight.data, scale.data, zp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def convert(self, model: torch.fx.GraphModule, observer_node: torch.fx.Node) -> None:
        """
        Replaces the given observer node from the given model with a quantized
        weight and a OpenVINO specific decompression module.

        :param model: A `torch.fx.GraphModule` representing the statically traced model
                    with observer nodes attached and calibrated.
        :param observer_node: The `torch.fx.Node` corresponding to the observer module for
                            the weight that is being transformed into a compressed representation.
        """
        weight_node = observer_node.args[0]
        original_weight = get_tensor_constant_from_node(weight_node, model)
        q_weight, scale, zero_point = self.calculate_qparams(original_weight)

        decompressor = self._create_decompressor(scale, zero_point, q_weight, original_weight)
        packed_q_weight = decompressor.pack_weight(q_weight)

        # Weight port id is 0 since observer is inserted for a single weight only.
        constant_update(model, observer_node, packed_q_weight, input_port_id=0)

        compressed_weight_name = observer_node.all_input_nodes[0].name
        decompressor_suffix = "_".join(compressed_weight_name.replace(".", "_").split("_")[:-2])
        decompressor_name = f"{decompressor.quantization_mode}_weights_decompressor_{decompressor_suffix}"

        module_insertion(
            model,
            decompressor,
            [
                PTTargetPoint(
                    TargetType.OPERATOR_POST_HOOK,
                    target_node_name=compressed_weight_name,
                )
            ],
            decompressor_name,
        )
        node_removal(model, observer_node, 0)

    @abstractmethod
    def _create_decompressor(
        self,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        q_weight: torch.Tensor,
        original_weight: torch.Tensor,
    ) -> BaseWeightsDecompressor:
        """
        Returns a respective NNCF decompressor for different types of quantization.

        :param scale: Calculated scale quantization parameter.
        :param zero_point: Calculated zero_point quantization parameter.
        :param q_weight: Calculated quantized weight.
        :param original_weight: FP weight.
        :return: NNCF observer according to the qmode which creates the decompression subgraph supported by OpenVINO.
        """


class INT4WeightObserver(WeightObserverBase):
    """
    OpenVINO INT4 Weight Compression observer.
    """

    def _create_decompressor(
        self,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        q_weight: torch.Tensor,
        original_weight: torch.Tensor,
    ) -> BaseWeightsDecompressor:
        if zero_point is None:
            return INT4SymmetricWeightsDecompressor(scale, q_weight.shape, original_weight.shape, original_weight.dtype)
        return INT4AsymmetricWeightsDecompressor(
            scale,
            zero_point,
            q_weight.shape,
            original_weight.shape,
            original_weight.dtype,
        )


class INT8WeightObserver(WeightObserverBase):
    """
    OpenVINO INT8 Weight Compression per channel observer.
    """

    def _create_decompressor(
        self,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        q_weight: torch.Tensor,
        original_weight: torch.Tensor,
    ) -> BaseWeightsDecompressor:
        if zero_point is None:
            return INT8SymmetricWeightsDecompressor(scale, original_weight.dtype)
        return INT8AsymmetricWeightsDecompressor(scale, zero_point, original_weight.dtype)
