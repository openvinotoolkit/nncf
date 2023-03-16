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

import numpy as np

from nncf.experimental.openvino_native.graph.nncf_graph_builder import GraphConverter
from nncf.experimental.openvino_native.graph.node_utils import get_weight_value
from nncf.experimental.openvino_native.graph.node_utils import is_node_with_bias
from nncf.common.factory import NNCFGraphFactory
from tests.openvino.native.models import FPModel
from tests.openvino.native.models import ConvNotBiasModel
from tests.openvino.native.models import ConvModel


def test_get_weight_value_const_with_convert():
    model = FPModel(const_dtype='FP16').ov_model
    nncf_graph = NNCFGraphFactory.create(model)
    node_with_weight = nncf_graph.get_node_by_name('MatMul')

    actual_value = get_weight_value(node_with_weight, nncf_graph, model)
    assert actual_value.dtype == np.uint16


def test_is_node_with_bias():
    model = ConvNotBiasModel().ov_model
    nncf_graph = GraphConverter.create_nncf_graph(model)
    node = nncf_graph.get_node_by_name('Conv')
    assert not is_node_with_bias(node, nncf_graph)

    model = ConvModel().ov_model
    nncf_graph = GraphConverter.create_nncf_graph(model)
    node = nncf_graph.get_node_by_name('Conv')
    assert is_node_with_bias(node, nncf_graph)
