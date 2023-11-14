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
from nncf.openvino.graph.layer_attributes import OVLayerAttributes
from nncf.openvino.graph.nncf_graph_builder import GraphConverter


def get_conv(node_name, input_shape, kernel=None):
    strides = [1, 1]
    pads = [0, 0]
    dilations = [1, 1]
    if kernel is None:
        shape = (input_shape[1] + 1, input_shape[1], 2, 1)
        kernel = opset.constant(np.ones(shape), dtype=np.float32, name="Const")
    input_ = opset.parameter(input_shape, name="Input0")
    output = opset.convolution(input_, kernel, strides, pads, pads, dilations, name=node_name)
    return [input_], [output]


def get_group_conv(node_name, input_shape, kernel=None):
    strides = [1, 2]
    pads = [0, 1]
    dilations = [3, 1]
    if kernel is None:
        shape = (input_shape[1], input_shape[1], 1, 1, 1)
        kernel = opset.constant(np.ones(shape), dtype=np.float32, name="Const")
    input_1 = opset.parameter(input_shape, name="Input0")
    output = opset.group_convolution(input_1, kernel, strides, pads, pads, dilations, name=node_name)
    return [input_1], [output]


def get_transpose_conv(node_name, input_shape, kernel=None):
    strides = [1, 1]
    pads = [0, 0]
    dilations = [1, 1]
    if kernel is None:
        shape = (input_shape[1], input_shape[1] + 1, 2, 1)
        kernel = opset.constant(np.ones(shape), dtype=np.float32, name="Const")
    input_1 = opset.parameter(input_shape, name="Input0")
    output = opset.convolution_backprop_data(
        input_1, kernel, strides, pads_begin=pads, pads_end=pads, dilations=dilations, name=node_name
    )
    return [input_1], [output]


def get_transpose_group_conv(node_name, input_shape, kernel=None):
    strides = [1, 2]
    pads = [0, 1]
    dilations = [3, 1]
    if kernel is None:
        shape = (input_shape[1], 1, input_shape[1], 1, 1)
        kernel = opset.constant(np.ones(shape), dtype=np.float32, name="Const")
    input_ = opset.parameter(input_shape, name="Input0")
    output = opset.group_convolution_backprop_data(
        input_, kernel, strides, pads_begin=pads, pads_end=pads, dilations=dilations, name=node_name
    )
    return [input_], [output]


def get_convert_conv(node_name, input_shape):
    shape = (input_shape[1] + 1, input_shape[1], 1, 1)
    const = opset.constant(np.ones(shape), dtype=np.float64, name="Const")
    convert = opset.convert(const, np.float32)
    return get_conv(node_name, input_shape, convert)


def get_matmul_b(node_name, input_shape):
    return get_matmul(node_name, input_shape, transpose_b=True)


def get_matmul_a(node_name, input_shape):
    return get_matmul(node_name, input_shape, transpose_a=True)


def get_matmul(node_name, input_shape, transpose_a=False, transpose_b=False):
    channel_position = 1 if transpose_a else -1
    data_shape = [input_shape[channel_position], 1]
    if transpose_b:
        data_shape = data_shape[::-1]
    data = opset.constant(np.ones(tuple(data_shape)), dtype=np.float32, name="Const")
    input_ = opset.parameter(input_shape, name="Input0")
    output = opset.matmul(input_, data, transpose_a=transpose_a, transpose_b=transpose_b, name=node_name)
    return [input_], [output]


def get_shape_node(op_name, input_shape):
    input_ = opset.parameter(input_shape, name="Input0")
    output = opset.shape_of(input_, name=op_name)
    return [input_], [output]


def get_concat_node(op_name, input_shape):
    num_inputs = 3
    inputs = [opset.parameter(input_shape, name=f"Input{i}") for i in range(num_inputs)]
    output = opset.concat(inputs, axis=1, name=op_name)
    return inputs, [output]


def get_one_layer_model(op_name: str, node_creator, input_shape):
    inputs, outputs = node_creator(op_name, input_shape)
    results = [opset.result(o, name=f"Result{i}") for i, o in enumerate(outputs)]
    model = ov.Model(results, inputs)
    return model


@pytest.mark.parametrize(
    "node_creator, input_shape, ref_layer_attrs",
    [
        (
            get_conv,
            (1, 3, 3, 3),
            OVLayerAttributes(
                {1: {"name": "Const", "shape": (4, 3, 2, 1)}},
                {
                    1: ConvolutionLayerAttributes(
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
                },
                {},
            ),
        ),
        (
            get_convert_conv,
            (1, 3, 3, 3),
            OVLayerAttributes(
                {1: {"name": "Const", "shape": (4, 3, 1, 1)}},
                {
                    1: ConvolutionLayerAttributes(
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
                },
                {},
            ),
        ),
        (
            get_group_conv,
            (1, 3, 3, 3),
            OVLayerAttributes(
                {1: {"name": "Const", "shape": (3, 3, 1, 1, 1)}},
                {
                    1: ConvolutionLayerAttributes(
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
                },
                {},
            ),
        ),
        (
            get_transpose_conv,
            (1, 3, 3, 3),
            OVLayerAttributes(
                {1: {"name": "Const", "shape": (3, 4, 2, 1)}},
                {
                    1: ConvolutionLayerAttributes(
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
                },
                {},
            ),
        ),
        (
            get_transpose_group_conv,
            (1, 3, 3, 3),
            OVLayerAttributes(
                {1: {"name": "Const", "shape": (3, 1, 3, 1, 1)}},
                {
                    1: ConvolutionLayerAttributes(
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
                },
                {},
            ),
        ),
        (get_shape_node, (1, 3, 3, 3), None),
        (
            get_matmul_b,
            (1, 3, 4),
            OVLayerAttributes(
                {1: {"name": "Const", "shape": (1, 4), "transpose": True}},
                {1: GenericWeightedLayerAttributes(False, (1, 4))},
                {"transpose": False},
            ),
        ),
        (
            get_matmul_a,
            (1, 3, 4),
            OVLayerAttributes(
                {1: {"name": "Const", "shape": (3, 1), "transpose": False}},
                {1: GenericWeightedLayerAttributes(False, (3, 1))},
                {"transpose": True},
            ),
        ),
        (
            get_concat_node,
            (1, 3, 4),
            OVLayerAttributes(
                {},
                {},
                {},
            ),
        ),
    ][::-1],
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
