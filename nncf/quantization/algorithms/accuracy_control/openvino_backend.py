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

from typing import List, Optional

import numpy as np
import openvino.runtime as ov
from openvino import Type
from openvino.properties.hint import inference_precision

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.openvino.engine import OVCompiledModelEngine
from nncf.openvino.graph.layer_attributes import OVLayerAttributes
from nncf.openvino.graph.metatypes.groups import CONSTANT_OPERATIONS
from nncf.openvino.graph.metatypes.groups import FAKE_QUANTIZE_OPERATIONS
from nncf.openvino.graph.metatypes.groups import INPUTS_QUANTIZABLE_OPERATIONS
from nncf.openvino.graph.metatypes.groups import OPERATIONS_WITH_WEIGHTS
from nncf.openvino.graph.metatypes.groups import QUANTIZE_AGNOSTIC_OPERATIONS
from nncf.openvino.graph.metatypes.groups import SHAPEOF_OPERATIONS
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConcatMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVOpMetatype
from nncf.openvino.graph.model_utils import get_start_nodes_for_activation_path_tracing
from nncf.openvino.graph.model_utils import model_has_state
from nncf.openvino.graph.node_utils import get_bias_value
from nncf.openvino.graph.node_utils import get_weight_value
from nncf.openvino.graph.node_utils import is_node_with_bias
from nncf.quantization.algorithms.accuracy_control.backend import AccuracyControlAlgoBackend
from nncf.quantization.algorithms.accuracy_control.backend import PreparedModel


class OVPreparedModel(PreparedModel):
    """
    Implementation of the `PreparedModel` for OpenVINO backend.
    """

    def __init__(self, model: ov.Model, use_fp32_precision: bool = True):
        self._stateful = model_has_state(model)
        config = None
        if use_fp32_precision:
            config = {inference_precision: Type.f32}
        self._compiled_model = ov.compile_model(model, device_name="CPU", config=config)
        self._engine = None

    @property
    def model_for_inference(self) -> ov.CompiledModel:
        return self._compiled_model

    @property
    def engine(self) -> OVCompiledModelEngine:
        if self._engine is None:
            self._engine = OVCompiledModelEngine(self._compiled_model, self._stateful)
        return self._engine


class OVAccuracyControlAlgoBackend(AccuracyControlAlgoBackend):
    """
    Implementation of the `AccuracyControlAlgoBackend` for OpenVINO backend.
    """

    # Metatypes

    @staticmethod
    def get_op_with_weights_metatypes() -> List[OVOpMetatype]:
        return OPERATIONS_WITH_WEIGHTS

    @staticmethod
    def get_quantizer_metatypes() -> List[OVOpMetatype]:
        return FAKE_QUANTIZE_OPERATIONS

    @staticmethod
    def get_const_metatypes() -> List[OVOpMetatype]:
        return CONSTANT_OPERATIONS

    @staticmethod
    def get_quantizable_metatypes() -> List[OVOpMetatype]:
        return INPUTS_QUANTIZABLE_OPERATIONS

    @staticmethod
    def get_quantize_agnostic_metatypes() -> List[OVOpMetatype]:
        return QUANTIZE_AGNOSTIC_OPERATIONS + [OVConcatMetatype]

    @staticmethod
    def get_shapeof_metatypes() -> List[OVOpMetatype]:
        return SHAPEOF_OPERATIONS

    @staticmethod
    def get_start_nodes_for_activation_path_tracing(nncf_graph: NNCFGraph) -> List[NNCFNode]:
        return get_start_nodes_for_activation_path_tracing(nncf_graph)

    # Manipulations with bias value and weights

    @staticmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return is_node_with_bias(node, nncf_graph)

    @staticmethod
    def is_node_with_weight(node: NNCFNode) -> bool:
        return node.metatype in OPERATIONS_WITH_WEIGHTS and isinstance(node.layer_attributes, OVLayerAttributes)

    @staticmethod
    def get_bias_value(node_with_bias: NNCFNode, nncf_graph: NNCFGraph, model: ov.Model) -> np.ndarray:
        return get_bias_value(node_with_bias, nncf_graph, model)

    @staticmethod
    def get_weight_value(node_with_weight: NNCFNode, model: ov.Model, port_id: int) -> np.ndarray:
        return get_weight_value(node_with_weight, model, port_id)

    @staticmethod
    def get_weight_tensor_port_ids(node: NNCFNode) -> List[Optional[int]]:
        return node.layer_attributes.get_const_port_ids()

    @staticmethod
    def get_model_size(model: ov.Model) -> int:
        model_size = 0
        for op in model.get_ops():
            if op.get_type_name() == "Constant":
                model_size += op.data.nbytes

        return model_size
