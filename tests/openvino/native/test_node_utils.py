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
import pytest
from openvino.runtime import opset13 as opset

from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.graph import NNCFNode
from nncf.openvino.graph.layer_attributes import OVLayerAttributes
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from nncf.openvino.graph.node_utils import get_weight_channel_axes
from nncf.openvino.graph.node_utils import get_weight_value
from nncf.openvino.graph.node_utils import get_weighted_layer_attributes
from nncf.openvino.graph.node_utils import is_node_with_bias
from tests.openvino.native.models import ConvModel
from tests.openvino.native.models import ConvNotBiasModel
from tests.openvino.native.models import FPModel
from tests.openvino.native.models import MatMul2DModel
from tests.openvino.native.models import MatMul2DNotBiasModel


def test_get_weight_value_const_with_convert():
    model = FPModel(const_dtype="FP16").ov_model
    nncf_graph = NNCFGraphFactory.create(model)
    node_with_weight = nncf_graph.get_node_by_name("MatMul")

    actual_value = get_weight_value(node_with_weight, model, port_id=1)
    assert actual_value.dtype == np.float16


@pytest.mark.parametrize(
    "model_to_create, is_with_bias, node_name",
    [
        [ConvNotBiasModel, True, "Conv"],
        [ConvModel, True, "Conv"],
        # TODO: add group conv to node with bias
        # [DepthwiseConv3DModel, True, 'Conv3D'],
        # [DepthwiseConv4DModel, True, 'Conv4D'],
        # [DepthwiseConv5DModel, True, 'Conv5D'],
        [MatMul2DModel, True, "MatMul"],
        [MatMul2DNotBiasModel, True, "MatMul"],
    ],
)
def test_is_node_with_bias(model_to_create, is_with_bias, node_name):
    model = model_to_create().ov_model
    nncf_graph = GraphConverter.create_nncf_graph(model)
    node = nncf_graph.get_node_by_name(node_name)
    assert is_node_with_bias(node, nncf_graph) == is_with_bias


@pytest.mark.parametrize(
    "weights_port_id, transpose, shape, expected_channel_axes",
    [
        (0, False, (1,), []),
        (0, True, (1,), []),
        (1, False, (1,), []),
        (1, True, (1,), []),
        (0, False, (1, 1), [0]),
        (0, True, (1, 1), [1]),
        (1, False, (1, 1), [1]),
        (1, True, (1, 1), [0]),
        (0, False, (1, 1, 1, 1), [0, 1, 2]),
        (0, True, (1, 1, 1, 1), [0, 1, 3]),
        (1, False, (1, 1, 1, 1), [0, 1, 3]),
        (1, True, (1, 1, 1, 1), [0, 1, 2]),
    ],
)
def test_get_weight_channel_axes_for_matmul(weights_port_id, transpose, shape, expected_channel_axes):
    input_1 = opset.parameter([1, 1], name="Input", dtype=np.float32)
    constant_1 = opset.constant(np.ones(shape).astype(np.float32))
    inputs_ = (input_1, constant_1) if weights_port_id == 1 else (constant_1, input_1)
    matmul_1 = opset.matmul(*inputs_, transpose_a=transpose, transpose_b=transpose, name="MatMul")

    constant_attrs = {weights_port_id: {"transpose": transpose, "shape": shape}}
    attributes = {
        NNCFNode.ID_NODE_ATTR: 0,
        NNCFNode.NODE_NAME_ATTR: "test",
        NNCFNode.METATYPE_ATTR: OVMatMulMetatype,
        NNCFNode.LAYER_ATTRIBUTES: OVLayerAttributes(
            layer_attributes=get_weighted_layer_attributes(matmul_1, OVMatMulMetatype, constant_attrs),
            constant_attributes=constant_attrs,
        ),
    }
    node = NNCFNode(attributes)
    actual_channel_axes = get_weight_channel_axes(node)

    assert len(actual_channel_axes) == len(expected_channel_axes)
    assert all(a == b for a, b in zip(actual_channel_axes, expected_channel_axes))
