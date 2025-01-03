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

from typing import Any, List, Optional

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.quantization.algorithms.accuracy_control.backend import AccuracyControlAlgoBackend
from nncf.quantization.algorithms.accuracy_control.backend import TModel
from tests.common.quantization.metatypes import CONSTANT_METATYPES
from tests.common.quantization.metatypes import QUANTIZABLE_METATYPES
from tests.common.quantization.metatypes import QUANTIZE_AGNOSTIC_METATYPES
from tests.common.quantization.metatypes import QUANTIZER_METATYPES
from tests.common.quantization.metatypes import WEIGHT_LAYER_METATYPES
from tests.common.quantization.metatypes import ShapeOfTestMetatype


class AABackendForTests(AccuracyControlAlgoBackend):
    @staticmethod
    def get_op_with_weights_metatypes() -> List[OperatorMetatype]:
        return WEIGHT_LAYER_METATYPES

    @staticmethod
    def get_quantizer_metatypes() -> List[OperatorMetatype]:
        return QUANTIZER_METATYPES

    @staticmethod
    def get_const_metatypes() -> List[OperatorMetatype]:
        return CONSTANT_METATYPES

    @staticmethod
    def get_quantizable_metatypes() -> List[OperatorMetatype]:
        return QUANTIZABLE_METATYPES

    @staticmethod
    def get_start_nodes_for_activation_path_tracing(nncf_graph: NNCFGraph) -> List[NNCFNode]:
        return nncf_graph.get_input_nodes()

    @staticmethod
    def get_quantize_agnostic_metatypes() -> List[OperatorMetatype]:
        return QUANTIZE_AGNOSTIC_METATYPES

    @staticmethod
    def get_shapeof_metatypes() -> List[OperatorMetatype]:
        return [ShapeOfTestMetatype]

    @staticmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return False

    @staticmethod
    def is_node_with_weight(node: NNCFNode) -> bool:
        return False

    @staticmethod
    def get_bias_value(node_with_bias: NNCFNode, nncf_graph: NNCFGraph, model: TModel) -> Any:
        return None

    @staticmethod
    def get_weight_value(node_with_weight: NNCFNode, model: TModel, port_id: int) -> Any:
        return None

    @staticmethod
    def get_weight_tensor_port_ids(node: NNCFNode) -> List[Optional[int]]:
        return None

    @staticmethod
    def get_model_size(model: TModel) -> int:
        return 0
