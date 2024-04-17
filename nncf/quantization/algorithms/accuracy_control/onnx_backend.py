# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

import numpy as np
import onnx

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.onnx.engine import ONNXEngine
from nncf.onnx.graph.metatypes import onnx_metatypes
from nncf.onnx.graph.metatypes.groups import INPUTS_QUANTIZABLE_OPERATIONS
from nncf.onnx.graph.metatypes.groups import OPERATIONS_WITH_WEIGHTS
from nncf.onnx.graph.metatypes.groups import QUANTIZE_AGNOSTIC_OPERATIONS
from nncf.onnx.graph.metatypes.groups import QUANTIZE_DEQUANTIZE_OPERATIONS
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXOpMetatype
from nncf.onnx.graph.node_utils import get_bias_value
from nncf.onnx.graph.node_utils import is_node_with_bias
from nncf.onnx.graph.onnx_helper import get_tensor_value
from nncf.quantization.algorithms.accuracy_control.backend import AccuracyControlAlgoBackend
from nncf.quantization.algorithms.accuracy_control.backend import PreparedModel


class ONNXPreparedModel(PreparedModel):
    """
    Implementation of the `PreparedModel` for ONNX backend.
    """

    def __init__(self, model: onnx.ModelProto):
        self._model = model
        self._engine = None

    @property
    def model_for_inference(self) -> onnx.ModelProto:
        return self._model

    @property
    def engine(self) -> ONNXEngine:
        if self._engine is None:
            self._engine = ONNXEngine(self._model)
        return self._engine


class ONNXAccuracyControlAlgoBackend(AccuracyControlAlgoBackend):
    """
    Implementation of the `AccuracyControlAlgoBackend` for ONNX backend.
    """

    # Metatypes

    @staticmethod
    def get_op_with_weights_metatypes() -> List[ONNXOpMetatype]:
        return OPERATIONS_WITH_WEIGHTS

    @staticmethod
    def get_quantizer_metatypes() -> List[ONNXOpMetatype]:
        return QUANTIZE_DEQUANTIZE_OPERATIONS

    @staticmethod
    def get_const_metatypes() -> List[ONNXOpMetatype]:
        return [onnx_metatypes.ONNXConstantMetatype]

    @staticmethod
    def get_quantizable_metatypes() -> List[ONNXOpMetatype]:
        return INPUTS_QUANTIZABLE_OPERATIONS

    @staticmethod
    def get_quantize_agnostic_metatypes() -> List[ONNXOpMetatype]:
        return QUANTIZE_AGNOSTIC_OPERATIONS + [onnx_metatypes.ONNXConcatMetatype]

    @staticmethod
    def get_shapeof_metatypes() -> List[ONNXOpMetatype]:
        return [onnx_metatypes.ONNXShapeMetatype]

    @staticmethod
    def get_start_nodes_for_activation_path_tracing(nncf_graph: NNCFGraph) -> List[NNCFNode]:
        return nncf_graph.get_input_nodes()

    # Manipulations with bias value and weights

    @staticmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return is_node_with_bias(node)

    @staticmethod
    def is_node_with_weight(node: NNCFNode) -> bool:
        return node.metatype in OPERATIONS_WITH_WEIGHTS and node.layer_attributes.has_weight()

    @staticmethod
    def get_bias_value(node_with_bias: NNCFNode, nncf_graph: NNCFGraph, model: onnx.ModelProto) -> np.ndarray:
        return get_bias_value(node_with_bias, model)

    @staticmethod
    def get_weight_value(node_with_weight: NNCFNode, model: onnx.ModelProto, port_id: int) -> np.ndarray:
        assert node_with_weight.layer_attributes.has_weight()
        weight_name = node_with_weight.layer_attributes.weight_attrs[port_id]["name"]
        return get_tensor_value(model, weight_name)

    @staticmethod
    def get_weight_tensor_port_ids(node: NNCFNode) -> List[Optional[int]]:
        return list(node.layer_attributes.weight_attrs.keys())

    @staticmethod
    def get_model_size(model: onnx.ModelProto) -> int:
        model_size = 0
        for initializer in model.graph.initializer:
            model_size += onnx.numpy_helper.to_array(initializer).nbytes
        for node in model.graph.node:
            for attr in node.attribute:
                if attr.HasField("t"):
                    model_size += onnx.numpy_helper.to_array(attr.t).nbytes
                for t in attr.tensors:
                    model_size += onnx.numpy_helper.to_array(t).nbytes
        return model_size
