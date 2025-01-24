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

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import openvino.runtime as ov
import pytest
from openvino.runtime import opset13 as opset

from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import GenericWeightedLayerAttributes
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.openvino.graph.layer_attributes import OVLayerAttributes
from nncf.openvino.graph.layout import OVLayoutElem
from nncf.openvino.graph.layout import get_conv_weights_layout_from_node
from nncf.openvino.graph.layout import get_linear_activations_layout_from_node
from nncf.openvino.graph.layout import get_linear_weights_layout_from_node
from nncf.openvino.graph.nncf_graph_builder import GraphConverter


def get_conv(input_1, node_name, input_shape, kernel=None):
    strides = [1, 1]
    pads = [0, 0]
    dilations = [1, 1]
    if kernel is None:
        shape = (input_shape[1] + 1, input_shape[1], 2, 1)
        kernel = opset.constant(np.ones(shape), dtype=np.float32, name="Const")
    return opset.convolution(input_1, kernel, strides, pads, pads, dilations, name=node_name)


def get_group_conv(input_1, node_name, input_shape):
    shape = (input_shape[1] // 2, input_shape[1], 2, 1, 1)
    kernel = opset.constant(np.ones(shape), dtype=np.float32, name="Const")
    return get_depthwise_conv(input_1, node_name, input_shape, kernel)


def get_depthwise_conv(input_1, node_name, input_shape, kernel=None):
    strides = [1, 2]
    pads = [0, 1]
    dilations = [3, 1]
    if kernel is None:
        shape = (input_shape[1], input_shape[1], 1, 1, 1)
        kernel = opset.constant(np.ones(shape), dtype=np.float32, name="Const")
    return opset.group_convolution(input_1, kernel, strides, pads, pads, dilations, name=node_name)


def get_transpose_conv(input_1, node_name, input_shape, kernel=None):
    strides = [1, 1]
    pads = [0, 0]
    dilations = [1, 1]
    if kernel is None:
        shape = (input_shape[1], input_shape[1] + 1, 2, 1)
        kernel = opset.constant(np.ones(shape), dtype=np.float32, name="Const")
    return opset.convolution_backprop_data(
        input_1, kernel, strides, pads_begin=pads, pads_end=pads, dilations=dilations, name=node_name
    )


def get_transpose_group_conv(input_1, node_name, input_shape, kernel=None):
    strides = [1, 2]
    pads = [0, 1]
    dilations = [3, 1]
    if kernel is None:
        shape = (input_shape[1], 1, input_shape[1], 1, 1)
        kernel = opset.constant(np.ones(shape), dtype=np.float32, name="Const")
    return opset.group_convolution_backprop_data(
        input_1, kernel, strides, pads_begin=pads, pads_end=pads, dilations=dilations, name=node_name
    )


def get_convert_conv(input_1, node_name, input_shape):
    shape = (input_shape[1] + 1, input_shape[1], 1, 1)
    const = opset.constant(np.ones(shape), dtype=np.float64, name="Const")
    convert = opset.convert(const, np.float32)
    return get_conv(input_1, node_name, input_shape, convert)


def get_matmul_b(input_1, node_name, input_shape):
    return get_matmul(input_1, node_name, input_shape, transpose_b=True)


def get_matmul_a(input_1, node_name, input_shape):
    return get_matmul(input_1, node_name, input_shape, transpose_a=True)


def get_matmul_b_swapped(input_1, node_name, input_shape):
    return get_matmul(input_1, node_name, input_shape, transpose_b=True, swap_inputs=True)


def get_matmul_a_swapped(input_1, node_name, input_shape):
    return get_matmul(input_1, node_name, input_shape, transpose_a=True, swap_inputs=True)


def get_matmul(input_1, node_name, input_shape, transpose_a=False, transpose_b=False, swap_inputs=False):
    channel_position = 1 if transpose_a else -1
    data_shape = [input_shape[channel_position], 1]
    if transpose_b:
        data_shape = data_shape[::-1]
    data = opset.constant(np.ones(tuple(data_shape)), dtype=np.float32, name="Const")
    a, b = (data, input_1) if swap_inputs else (input_1, data)
    return opset.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b, name=node_name)


def get_1d_matmul(input_1, node_name, input_shape):
    data_shape = (input_shape[-1],)
    data = opset.constant(np.ones(tuple(data_shape)), dtype=np.float32, name="Const")
    return opset.matmul(input_1, data, transpose_a=False, transpose_b=False, name=node_name)


def get_add(input_1, node_name, input_shape):
    data_shape = [1] * len(input_shape)
    data = opset.constant(np.ones(tuple(data_shape)), dtype=np.float32, name="Const")
    return opset.add(input_1, data, name=node_name)


def get_lstm(input_1, node_name, input_shape):
    batch_size, _, input_size = input_shape
    hidden_size = 4
    num_directions = 1
    hs = opset.constant(np.ones((batch_size, num_directions, hidden_size)), dtype=np.float32, name="hs")
    cs = opset.constant(np.ones((batch_size, num_directions, hidden_size)), dtype=np.float32, name="cs")
    seq_len_const = opset.constant(np.ones((batch_size)), dtype=np.int32, name="seq_len_const")
    w = opset.constant(np.ones((num_directions, 4 * hidden_size, input_size)), dtype=np.float32, name="w")
    r = opset.constant(np.ones((num_directions, 4 * hidden_size, hidden_size)), dtype=np.float32, name="r")
    b = opset.constant(np.ones((num_directions, 4 * hidden_size)), dtype=np.float32, name="b")
    return opset.lstm_sequence(
        input_1, hs, cs, seq_len_const, w, r, b, hidden_size, "forward", name=node_name
    ).outputs()[0]


def get_shape_node(input_, op_name, input_shape):
    return opset.shape_of(input_, name=op_name)


def get_one_layer_model(op_name: str, node_creator, input_shape):
    input_1 = opset.parameter(input_shape, name="Input")
    op = node_creator(input_1, op_name, input_shape)
    result = opset.result(op, name="Result")
    model = ov.Model([result], [input_1])
    return model


@dataclass
class LayerAttributesTestCase:
    node_creator: Callable
    input_shape: Tuple[int, ...]
    act_port_id: int
    ref_layer_attrs: OVLayerAttributes
    ref_weights_layout: Tuple[OVLayoutElem]
    ref_acts_layout: Tuple[OVLayoutElem]


TEST_CASES_CONV = [
    LayerAttributesTestCase(
        get_conv,
        (1, 3, 3, 3),
        0,
        OVLayerAttributes(
            {1: {"name": "Const", "shape": (4, 3, 2, 1), "dtype": "f32"}},
            ConvolutionLayerAttributes(
                weight_requires_grad=False,
                in_channels=3,
                out_channels=4,
                kernel_size=(2, 1),
                stride=(1, 1),
                dilations=[1, 1],
                groups=1,
                transpose=False,
                padding_values=(0, 0, 0, 0),
            ),
            {},
        ),
        (
            OVLayoutElem.C_OUT,
            OVLayoutElem.C_IN,
            OVLayoutElem.SPATIAL,
            OVLayoutElem.SPATIAL,
        ),
        {},
    ),
    LayerAttributesTestCase(
        get_convert_conv,
        (1, 3, 3, 3),
        0,
        OVLayerAttributes(
            {1: {"name": "Const", "shape": (4, 3, 1, 1), "dtype": "f32"}},
            ConvolutionLayerAttributes(
                weight_requires_grad=False,
                in_channels=3,
                out_channels=4,
                kernel_size=(1, 1),
                stride=(1, 1),
                dilations=[1, 1],
                groups=1,
                transpose=False,
                padding_values=(0, 0, 0, 0),
            ),
            {},
        ),
        (
            OVLayoutElem.C_OUT,
            OVLayoutElem.C_IN,
            OVLayoutElem.SPATIAL,
            OVLayoutElem.SPATIAL,
        ),
        {},
    ),
    LayerAttributesTestCase(
        get_depthwise_conv,
        (1, 3, 3, 3),
        0,
        OVLayerAttributes(
            {1: {"name": "Const", "shape": (3, 3, 1, 1, 1), "dtype": "f32"}},
            ConvolutionLayerAttributes(
                weight_requires_grad=False,
                in_channels=1,
                out_channels=3,
                kernel_size=(1, 1),
                stride=(1, 2),
                dilations=[3, 1],
                groups=3,
                transpose=False,
                padding_values=(0, 1, 0, 1),
            ),
            {},
        ),
        (
            OVLayoutElem.GROUPS,
            OVLayoutElem.C_OUT,
            OVLayoutElem.C_IN,
            OVLayoutElem.SPATIAL,
            OVLayoutElem.SPATIAL,
        ),
        {},
    ),
    LayerAttributesTestCase(
        get_group_conv,
        (1, 10, 3, 3),
        0,
        OVLayerAttributes(
            {1: {"name": "Const", "shape": (5, 10, 2, 1, 1), "dtype": "f32"}},
            ConvolutionLayerAttributes(
                weight_requires_grad=False,
                in_channels=2,
                out_channels=10,
                kernel_size=(1, 1),
                stride=(1, 2),
                dilations=[3, 1],
                groups=5,
                transpose=False,
                padding_values=(0, 1, 0, 1),
            ),
            {},
        ),
        (
            OVLayoutElem.GROUPS,
            OVLayoutElem.C_OUT,
            OVLayoutElem.C_IN,
            OVLayoutElem.SPATIAL,
            OVLayoutElem.SPATIAL,
        ),
        {},
    ),
    LayerAttributesTestCase(
        get_transpose_conv,
        (1, 3, 3, 3),
        0,
        OVLayerAttributes(
            {1: {"name": "Const", "shape": (3, 4, 2, 1), "dtype": "f32"}},
            ConvolutionLayerAttributes(
                weight_requires_grad=False,
                in_channels=3,
                out_channels=4,
                kernel_size=(2, 1),
                stride=(1, 1),
                dilations=[1, 1],
                groups=1,
                transpose=True,
                padding_values=(0, 0, 0, 0),
            ),
            {},
        ),
        (
            OVLayoutElem.C_IN,
            OVLayoutElem.C_OUT,
            OVLayoutElem.SPATIAL,
            OVLayoutElem.SPATIAL,
        ),
        {},
    ),
    LayerAttributesTestCase(
        get_transpose_group_conv,
        (1, 3, 3, 3),
        0,
        OVLayerAttributes(
            {1: {"name": "Const", "shape": (3, 1, 3, 1, 1), "dtype": "f32"}},
            ConvolutionLayerAttributes(
                weight_requires_grad=False,
                in_channels=1,
                out_channels=3,
                kernel_size=(1, 1),
                stride=(1, 2),
                dilations=[3, 1],
                groups=3,
                transpose=True,
                padding_values=(0, 1, 0, 1),
            ),
            {},
        ),
        (
            OVLayoutElem.GROUPS,
            OVLayoutElem.C_IN,
            OVLayoutElem.C_OUT,
            OVLayoutElem.SPATIAL,
            OVLayoutElem.SPATIAL,
        ),
        {},
    ),
]


TEST_CASES_LINEAR = [
    LayerAttributesTestCase(
        get_matmul_b,
        (1, 3, 4),
        0,
        OVLayerAttributes(
            {1: {"name": "Const", "shape": (1, 4), "dtype": "f32", "transpose": True}},
            LinearLayerAttributes(
                weight_requires_grad=False,
                in_features=4,
                out_features=1,
                with_bias=False,
            ),
            {"transpose": False},
        ),
        (OVLayoutElem.C_OUT, OVLayoutElem.C_IN),
        (OVLayoutElem.SPATIAL, OVLayoutElem.C_OUT, OVLayoutElem.C_IN),
    ),
    LayerAttributesTestCase(
        get_matmul_a,
        (1, 3, 4),
        0,
        OVLayerAttributes(
            {1: {"name": "Const", "shape": (3, 1), "dtype": "f32", "transpose": False}},
            LinearLayerAttributes(
                weight_requires_grad=False,
                in_features=3,
                out_features=1,
                with_bias=False,
            ),
            {"transpose": True},
        ),
        (OVLayoutElem.C_IN, OVLayoutElem.C_OUT),
        (OVLayoutElem.SPATIAL, OVLayoutElem.C_IN, OVLayoutElem.C_OUT),
    ),
    LayerAttributesTestCase(
        get_matmul_a_swapped,
        (1, 3, 4),
        1,
        OVLayerAttributes(
            {0: {"name": "Const", "shape": (3, 1), "dtype": "f32", "transpose": True}},
            LinearLayerAttributes(
                weight_requires_grad=False,
                in_features=3,
                out_features=1,
                with_bias=False,
            ),
            {"transpose": False},
        ),
        (OVLayoutElem.C_IN, OVLayoutElem.C_OUT),
        (OVLayoutElem.SPATIAL, OVLayoutElem.C_IN, OVLayoutElem.C_OUT),
    ),
    LayerAttributesTestCase(
        get_matmul_b_swapped,
        (1, 3, 4),
        1,
        OVLayerAttributes(
            {0: {"name": "Const", "shape": (1, 4), "dtype": "f32", "transpose": False}},
            LinearLayerAttributes(
                weight_requires_grad=False,
                in_features=4,
                out_features=1,
                with_bias=False,
            ),
            {"transpose": True},
        ),
        (OVLayoutElem.C_OUT, OVLayoutElem.C_IN),
        (OVLayoutElem.SPATIAL, OVLayoutElem.C_OUT, OVLayoutElem.C_IN),
    ),
    LayerAttributesTestCase(
        get_1d_matmul,
        (1, 3, 4),
        0,
        OVLayerAttributes(
            {1: {"name": "Const", "shape": (4,), "dtype": "f32", "transpose": False}},
            LinearLayerAttributes(
                weight_requires_grad=False,
                in_features=4,
                out_features=None,
                with_bias=False,
            ),
            {"transpose": False},
        ),
        (OVLayoutElem.C_IN,),
        (OVLayoutElem.SPATIAL, OVLayoutElem.C_OUT, OVLayoutElem.C_IN),
    ),
]


TEST_CASES_NO_WEGIHTS_LAYOUT = [
    LayerAttributesTestCase(get_shape_node, (1, 3, 3, 3), 0, None, None, None),
    LayerAttributesTestCase(
        get_add,
        (1, 3, 4, 5),
        0,
        OVLayerAttributes(
            {1: {"name": "Const", "shape": (1, 1, 1, 1), "dtype": "f32"}},
            GenericWeightedLayerAttributes(False, weight_shape=(1, 1, 1, 1)),
            {},
        ),
        None,
        {},
    ),
    LayerAttributesTestCase(
        get_lstm,
        (2, 3, 4),
        0,
        OVLayerAttributes(
            {
                1: {"name": "hs", "shape": (2, 1, 4), "dtype": "f32"},
                2: {"name": "cs", "shape": (2, 1, 4), "dtype": "f32"},
                4: {"name": "w", "shape": (1, 16, 4), "dtype": "f32"},
                5: {"name": "r", "shape": (1, 16, 4), "dtype": "f32"},
            },
            None,
            {},
        ),
        None,
        {},
    ),
]


def _get_node_to_test(test_descriptor: LayerAttributesTestCase):
    op_name = "test_node"
    ov_model = get_one_layer_model(op_name, test_descriptor.node_creator, test_descriptor.input_shape)
    nncf_graph = GraphConverter.create_nncf_graph(ov_model)
    return nncf_graph.get_node_by_name(op_name)


@pytest.mark.parametrize("test_descriptor", TEST_CASES_LINEAR + TEST_CASES_NO_WEGIHTS_LAYOUT)
def test_layer_attributes(test_descriptor: LayerAttributesTestCase):
    node = _get_node_to_test(test_descriptor)
    if test_descriptor.ref_layer_attrs is None:
        assert node.layer_attributes is None
    else:
        assert node.layer_attributes.__dict__ == test_descriptor.ref_layer_attrs.__dict__


@pytest.mark.parametrize("test_descriptor", TEST_CASES_CONV)
def test_get_conv_weights_layout_from_node(test_descriptor: LayerAttributesTestCase):
    node = _get_node_to_test(test_descriptor)
    for _ in range(2):  # To test get_conv_weights_layout_from_node is a clean function
        weights_layout = get_conv_weights_layout_from_node(node)
        assert weights_layout == test_descriptor.ref_weights_layout


@pytest.mark.parametrize("test_descriptor", TEST_CASES_LINEAR)
def test_get_linear_weights_layout_from_node(test_descriptor: LayerAttributesTestCase):
    node = _get_node_to_test(test_descriptor)
    for _ in range(2):  # To test get_linear_weights_layout_from_node is a clean function
        weights_layout = get_linear_weights_layout_from_node(node)
        assert weights_layout == test_descriptor.ref_weights_layout


@pytest.mark.parametrize("test_descriptor", TEST_CASES_LINEAR)
def test_get_linear_activations_layout_from_node(test_descriptor: LayerAttributesTestCase):
    node = _get_node_to_test(test_descriptor)
    for _ in range(2):  # To test get_linear_activations_layout_from_node is a clean function
        acts_layout = get_linear_activations_layout_from_node(
            node, test_descriptor.act_port_id, test_descriptor.input_shape
        )
        assert acts_layout == test_descriptor.ref_acts_layout
