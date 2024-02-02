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

from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionMetatype
from nncf.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.onnx.graph.node_utils import get_bias_value
from nncf.onnx.graph.node_utils import transpose_axis
from tests.onnx.models import OneConvolutionalIdentityBiasModel
from tests.onnx.models import OneConvolutionalModel


@pytest.mark.parametrize("model", [OneConvolutionalModel(), OneConvolutionalIdentityBiasModel()])
def test_get_bias_value(model):
    onnx_model = model.onnx_model
    nncf_graph = GraphConverter.create_nncf_graph(onnx_model)
    # Only one Convolution in test models
    conv_node = nncf_graph.get_nodes_by_metatypes([ONNXConvolutionMetatype])[0]
    bias_value = get_bias_value(conv_node, onnx_model)
    assert np.allclose(bias_value, model.conv_bias)


@pytest.mark.parametrize(
    "shape, axis, expected_channel_axis",
    [
        ((1, 3, 5, 5), -1, 0),
        ((1, 3, 5, 5), 1, 2),
        ((1, 3, 5, 5), 0, 3),
        ((1, 3, 5, 5), 2, 1),
        ((1, 3, 5, 5), -2, 1),
        ((1,), -1, 0),
        ((1, 1), -1, 0),
        ((1, 1), 1, 0),
        ((1, 1), 0, 1),
    ],
)
def test_transpose_axis(shape, axis, expected_channel_axis):
    assert expected_channel_axis == transpose_axis(shape, axis)
