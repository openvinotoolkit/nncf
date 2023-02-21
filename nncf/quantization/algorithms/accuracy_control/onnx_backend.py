"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from typing import List
from typing import Any

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.quantization.algorithms.accuracy_control.backend import AccuracyControlAlgoBackend
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXOpMetatype
from nncf.onnx.graph.node_utils import get_bias_value
from nncf.onnx.graph.node_utils import is_node_with_bias
from nncf.onnx.graph.transformations.command_creation import create_bias_correction_command


class ONNXAccuracyControlAlgoBackend(AccuracyControlAlgoBackend):
    """
    Implementation of the `AccuracyControlAlgoBackend` for ONNX backend.
    """

    # Metatypes

    @staticmethod
    def get_quantizer_metatypes() -> List[ONNXOpMetatype]:
        raise NotImplementedError

    @staticmethod
    def get_const_metatypes() -> List[ONNXOpMetatype]:
        raise NotImplementedError

    @staticmethod
    def get_quantizable_metatypes() -> List[ONNXOpMetatype]:
        raise NotImplementedError

    @staticmethod
    def get_quantize_agnostic_metatypes() -> List[ONNXOpMetatype]:
        raise NotImplementedError

    @staticmethod
    def get_shape_of_metatypes() -> List[ONNXOpMetatype]:
        raise NotImplementedError

    # Creation of commands

    @staticmethod
    def create_command_to_remove_quantizer(quantizer_node: NNCFNode):
        raise NotImplementedError

    @staticmethod
    def create_command_to_update_bias(node_with_bias: NNCFNode, bias_value: Any, nncf_graph: NNCFGraph):
        return create_bias_correction_command(node_with_bias, bias_value)

    @staticmethod
    def create_command_to_update_weight(node_with_weight: NNCFNode, weight_value: Any):
        raise NotImplementedError

    # Manipulations with bias value and weights

    @staticmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return is_node_with_bias(node)

    @staticmethod
    def is_node_with_weight(node: NNCFNode) -> bool:
        raise NotImplementedError

    @staticmethod
    def get_bias_value(node_with_bias: NNCFNode, nncf_graph: NNCFGraph, model) -> Any:
        return get_bias_value(node_with_bias, model)

    @staticmethod
    def get_weight_value(node_with_weight: NNCFNode, nncf_graph: NNCFGraph, model) -> Any:
        raise NotImplementedError

    # Preparation of model

    @staticmethod
    def prepare_for_inference(model) -> Any:
        raise NotImplementedError
