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
import openvino.runtime as ov
import pytest
from openvino.runtime import opset9 as opset

from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import GenericWeightedLayerAttributes
from nncf.common.graph.layer_attributes import LayoutElem
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.openvino.graph.layer_attributes import OVLayerAttributes
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


@pytest.mark.parametrize(
    "node_creator, input_shape, ref_layer_attrs",
    [
        (
            get_conv,
            (1, 3, 3, 3),
            OVLayerAttributes(
                {1: {"name": "Const", "shape": (4, 3, 2, 1)}},
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
                    weights_layout=(LayoutElem.C_OUT, LayoutElem.C_IN, LayoutElem.SPATIAL, LayoutElem.SPATIAL),
                ),
                {},
            ),
        ),
        (
            get_convert_conv,
            (1, 3, 3, 3),
            OVLayerAttributes(
                {1: {"name": "Const", "shape": (4, 3, 1, 1)}},
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
                    weights_layout=(LayoutElem.C_OUT, LayoutElem.C_IN, LayoutElem.SPATIAL, LayoutElem.SPATIAL),
                ),
                {},
            ),
        ),
        (
            get_depthwise_conv,
            (1, 3, 3, 3),
            OVLayerAttributes(
                {1: {"name": "Const", "shape": (3, 3, 1, 1, 1)}},
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
                    weights_layout=(
                        LayoutElem.GROUPS,
                        LayoutElem.C_OUT,
                        LayoutElem.C_IN,
                        LayoutElem.SPATIAL,
                        LayoutElem.SPATIAL,
                    ),
                ),
                {},
            ),
        ),
        (
            get_group_conv,
            (1, 10, 3, 3),
            OVLayerAttributes(
                {1: {"name": "Const", "shape": (5, 10, 2, 1, 1)}},
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
                    weights_layout=(
                        LayoutElem.GROUPS,
                        LayoutElem.C_OUT,
                        LayoutElem.C_IN,
                        LayoutElem.SPATIAL,
                        LayoutElem.SPATIAL,
                    ),
                ),
                {},
            ),
        ),
        (
            get_transpose_conv,
            (1, 3, 3, 3),
            OVLayerAttributes(
                {1: {"name": "Const", "shape": (3, 4, 2, 1)}},
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
                    weights_layout=(LayoutElem.C_IN, LayoutElem.C_OUT, LayoutElem.SPATIAL, LayoutElem.SPATIAL),
                ),
                {},
            ),
        ),
        (
            get_transpose_group_conv,
            (1, 3, 3, 3),
            OVLayerAttributes(
                {1: {"name": "Const", "shape": (3, 1, 3, 1, 1)}},
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
                    weights_layout=(
                        LayoutElem.GROUPS,
                        LayoutElem.C_IN,
                        LayoutElem.C_OUT,
                        LayoutElem.SPATIAL,
                        LayoutElem.SPATIAL,
                    ),
                ),
                {},
            ),
        ),
        (get_shape_node, (1, 3, 3, 3), None),
        (
            get_matmul_b,
            (1, 3, 4),
            OVLayerAttributes(
                {1: {"name": "Const", "shape": (1, 4), "transpose": True}},
                LinearLayerAttributes(
                    weight_requires_grad=False,
                    in_features=4,
                    out_features=1,
                    with_bias=False,
                    weights_layout=[LayoutElem.C_OUT, LayoutElem.C_IN],
                ),
                {"transpose": False},
            ),
        ),
        (
            get_matmul_a,
            (1, 3, 4),
            OVLayerAttributes(
                {1: {"name": "Const", "shape": (3, 1), "transpose": False}},
                LinearLayerAttributes(
                    weight_requires_grad=False,
                    in_features=3,
                    out_features=1,
                    with_bias=False,
                    weights_layout=[LayoutElem.C_IN, LayoutElem.C_OUT],
                ),
                {"transpose": True},
            ),
        ),
        (
            get_matmul_a_swapped,
            (1, 3, 4),
            OVLayerAttributes(
                {0: {"name": "Const", "shape": (3, 1), "transpose": True}},
                LinearLayerAttributes(
                    weight_requires_grad=False,
                    in_features=3,
                    out_features=1,
                    with_bias=False,
                    weights_layout=[LayoutElem.C_IN, LayoutElem.C_OUT],
                ),
                {"transpose": False},
            ),
        ),
        (
            get_matmul_b_swapped,
            (1, 3, 4),
            OVLayerAttributes(
                {0: {"name": "Const", "shape": (1, 4), "transpose": False}},
                LinearLayerAttributes(
                    weight_requires_grad=False,
                    in_features=4,
                    out_features=1,
                    with_bias=False,
                    weights_layout=[LayoutElem.C_OUT, LayoutElem.C_IN],
                ),
                {"transpose": True},
            ),
        ),
        (
            get_1d_matmul,
            (1, 3, 4),
            OVLayerAttributes(
                {1: {"name": "Const", "shape": (4,), "transpose": False}},
                LinearLayerAttributes(
                    weight_requires_grad=False,
                    in_features=4,
                    out_features=None,
                    with_bias=False,
                    weights_layout=[LayoutElem.C_IN],
                ),
                {"transpose": False},
            ),
        ),
        (
            get_add,
            (1, 3, 4, 5),
            OVLayerAttributes(
                {1: {"name": "Const", "shape": (1, 1, 1, 1)}},
                GenericWeightedLayerAttributes(False, weight_shape=(1, 1, 1, 1)),
                {},
            ),
        ),
        (
            get_lstm,
            (2, 3, 4),
            OVLayerAttributes(
                {
                    1: {"name": "hs", "shape": (2, 1, 4)},
                    2: {"name": "cs", "shape": (2, 1, 4)},
                    4: {"name": "w", "shape": (1, 16, 4)},
                    5: {"name": "r", "shape": (1, 16, 4)},
                },
                None,
                {},
            ),
        ),
    ],
)
def test_layer_attributes(node_creator, input_shape, ref_layer_attrs):
    op_name = "test_node"
    ov_model = get_one_layer_model(op_name, node_creator, input_shape)
    nncf_graph = GraphConverter.create_nncf_graph(ov_model)
    node = nncf_graph.get_node_by_name(op_name)
    if ref_layer_attrs is None:
        assert node.layer_attributes is None
    else:
        assert node.layer_attributes.__dict__ == ref_layer_attrs.__dict__
