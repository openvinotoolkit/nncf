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

import numpy as np
import openvino.runtime as ov
import pytest

from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.openvino.engine import OVCompiledModelEngine
from nncf.openvino.graph.metatypes.groups import CONSTANT_OPERATIONS
from nncf.openvino.graph.metatypes.groups import FAKE_QUANTIZE_OPERATIONS
from nncf.openvino.graph.metatypes.groups import INPUTS_QUANTIZABLE_OPERATIONS
from nncf.openvino.graph.metatypes.groups import OPERATIONS_WITH_WEIGHTS
from nncf.openvino.graph.metatypes.groups import QUANTIZE_AGNOSTIC_OPERATIONS
from nncf.openvino.graph.metatypes.groups import SHAPEOF_OPERATIONS
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConcatMetatype
from nncf.openvino.graph.model_utils import get_start_nodes_for_activation_path_tracing
from nncf.quantization.algorithms.accuracy_control.openvino_backend import OVAccuracyControlAlgoBackend
from nncf.quantization.algorithms.accuracy_control.openvino_backend import OVPreparedModel
from tests.openvino.native.models import ConvModel as OvConvModel
from tests.openvino.native.models import LinearModel as OvLinearModel


@pytest.fixture
def ov_prepared_model() -> OvLinearModel:
    model = OvLinearModel().ov_model
    return OVPreparedModel(model)


@pytest.fixture
def ov_graph_and_model() -> NNCFGraph:
    model = OvConvModel().ov_model
    return NNCFGraphFactory.create(model), model


def testOvPreparedModelInit(ov_prepared_model: OVPreparedModel):
    assert isinstance(ov_prepared_model.model_for_inference, ov.CompiledModel)
    assert ov_prepared_model._engine is None
    assert isinstance(ov_prepared_model.engine, OVCompiledModelEngine)
    assert isinstance(ov_prepared_model.engine, OVCompiledModelEngine)


def test_ov_accuracy_control_algo_backend_static_methods():
    assert OVAccuracyControlAlgoBackend.get_op_with_weights_metatypes() == OPERATIONS_WITH_WEIGHTS
    assert OVAccuracyControlAlgoBackend.get_quantizer_metatypes() == FAKE_QUANTIZE_OPERATIONS
    assert OVAccuracyControlAlgoBackend.get_const_metatypes() == CONSTANT_OPERATIONS
    assert OVAccuracyControlAlgoBackend.get_quantizable_metatypes() == INPUTS_QUANTIZABLE_OPERATIONS
    assert OVAccuracyControlAlgoBackend.get_quantize_agnostic_metatypes() == QUANTIZE_AGNOSTIC_OPERATIONS + [
        OVConcatMetatype
    ]
    assert OVAccuracyControlAlgoBackend.get_shapeof_metatypes() == SHAPEOF_OPERATIONS


def test_ov_accuracy_control_algo_backend_static_methods_with_graph(ov_graph_and_model):
    ov_graph, model = ov_graph_and_model
    assert OVAccuracyControlAlgoBackend.get_start_nodes_for_activation_path_tracing(
        ov_graph
    ) == get_start_nodes_for_activation_path_tracing(ov_graph)
    conv_node: NNCFNode = ov_graph.get_node_by_key("4 Conv")
    add_node: NNCFNode = ov_graph.get_node_by_key("3 Add")

    assert OVAccuracyControlAlgoBackend.is_node_with_bias(conv_node, ov_graph)
    assert OVAccuracyControlAlgoBackend.is_node_with_weight(conv_node)
    assert isinstance(OVAccuracyControlAlgoBackend.get_bias_value(conv_node, ov_graph, model), np.ndarray)
    assert OVAccuracyControlAlgoBackend.get_weight_tensor_port_ids(conv_node) == [1]
    assert isinstance(OVAccuracyControlAlgoBackend.get_weight_value(conv_node, model, 1), np.ndarray)
    assert OVAccuracyControlAlgoBackend.get_model_size(model) == 116

    assert not OVAccuracyControlAlgoBackend.is_node_with_bias(add_node, ov_graph)
    assert not OVAccuracyControlAlgoBackend.is_node_with_weight(add_node)
