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

import pytest
import numpy as np
import openvino.runtime as ov
from openvino.runtime import opset9 as opset

from nncf.experimental.openvino_native.graph.nncf_graph_builder import GraphConverter
from nncf.experimental.openvino_native.graph.nncf_graph_builder import OVConstantLayerAttributes


def get_conv(input_1, node_name, input_shape, kernel=None):
    strides = [1, 1]
    pads = [0, 0]
    dilations = [1, 1]
    if kernel is None:
        kernel = np.ones((input_shape[1], input_shape[1], 1, 1)).astype(np.float32)
    return opset.convolution(input_1, kernel,
                             strides, pads, pads, dilations, name=node_name)


def get_convert_conv(input_1, node_name, input_shape):
    kernel = np.ones((input_shape[1], input_shape[1], 1, 1)).astype(np.float64)
    const = opset.constant(kernel)
    convert = opset.convert(const, np.float32)
    return get_conv(input_1, node_name, input_shape, convert)


def get_one_layer_model(op_name: str, node_creator, input_shape):
    input_1 = opset.parameter(input_shape, name='Input')
    op = node_creator(input_1, op_name, input_shape)
    result = opset.result(op, name="Result")
    model = ov.Model([result], [input_1])
    return model


@pytest.mark.parametrize('node_creator, ref_layer_attrs',
                         [(get_conv, OVConstantLayerAttributes(1, (3, 3, 1, 1))),
                          (get_convert_conv, OVConstantLayerAttributes(1, (3, 3, 1, 1)))])
def test_layer_attributes(node_creator, ref_layer_attrs):
    input_shape = [1, 3, 3, 3]
    op_name = 'test_node'
    ov_model = get_one_layer_model(op_name, node_creator, input_shape)
    nncf_graph = GraphConverter.create_nncf_graph(ov_model)
    node = nncf_graph.get_node_by_name(op_name)
    assert node.layer_attributes.__dict__ == ref_layer_attrs.__dict__
