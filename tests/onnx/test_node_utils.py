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
import numpy as np
import pytest

from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionMetatype
from nncf.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.onnx.graph.node_utils import get_bias_value
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
