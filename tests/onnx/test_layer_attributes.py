# Copyright (c) 2026 Intel Corporation
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

from nncf.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.onnx.graph.nncf_graph_builder import ONNXLayerAttributes
from tests.onnx.common import ModelBuilder


def _build_identity_model():
    mb = ModelBuilder()
    x = mb.add_input("X", (3, 3, 3))
    x = mb.add_identity(x)
    mb.add_output(x, (3, 3, 3))
    model = mb.build(opset_version=19, ir_version=9)

    return model


def _build_conv_model():
    mb = ModelBuilder()
    x = mb.add_input("X", (1, 3, 3, 3))
    x = mb.add_conv(
        x,
        mb.add_initializer(np.zeros((3, 3, 1, 1), dtype=np.float32)),
        mb.add_initializer(np.zeros((3,), dtype=np.float32)),
        (1, 1),
    )
    mb.add_output(x, (1, 3, 3, 3))
    model = mb.build(opset_version=19, ir_version=9)

    return model


def _build_shared_weight_model():
    mb = ModelBuilder()
    x = mb.add_input("X", (3, 4))
    w = mb.add_initializer(np.zeros((4, 3), dtype=np.float32))
    x = mb.add_matmul(x, w)
    x = mb.add_matmul(x, mb.add_transpose(w, [1, 0]))
    mb.add_output(x, (3, 4))
    model = mb.build(opset_version=19, ir_version=9)

    return model


@pytest.mark.parametrize(
    "build_model,node_name,reference_attributes",
    [
        (_build_identity_model, "Identity_0", ONNXLayerAttributes()),
        (
            _build_conv_model,
            "Conv_0",
            ONNXLayerAttributes(
                weight_attrs={
                    1: {"name": "Initializer_0", "shape": [3, 3, 1, 1], "weight_input_shape": [3, 3, 1, 1], "dtype": 1}
                },
                bias_attrs={"name": "Initializer_1"},
            ),
        ),
        (
            _build_shared_weight_model,
            "MatMul_0",
            ONNXLayerAttributes(
                weight_attrs={1: {"name": "Initializer_0", "shape": [4, 3], "weight_input_shape": [4, 3], "dtype": 1}}
            ),
        ),
        (
            _build_shared_weight_model,
            "MatMul_2",
            ONNXLayerAttributes(
                weight_attrs={1: {"name": "Initializer_0", "shape": [4, 3], "weight_input_shape": [3, 4], "dtype": 1}}
            ),
        ),
    ],
)
def test_layer_attributes(build_model, node_name, reference_attributes):
    model = build_model()
    graph = GraphConverter.create_nncf_graph(model)
    node = graph.get_node_by_name(node_name)
    assert node.layer_attributes.__dict__ == reference_attributes.__dict__
