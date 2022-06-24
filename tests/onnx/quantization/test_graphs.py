"""
 Copyright (c) 2022 Intel Corporation
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

from dataclasses import dataclass
from typing import List

import numpy as np
import onnx
import pytest

#pylint: disable=no-member
def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:
    # (TensorProto)
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())

    return initializer_tensor


@dataclass
class TestCase:
    input_shape: List[int]
    model: onnx.ModelProto


@pytest.fixture
def fixture_fxt_reshape_weight_graph(name='fxt_reshape_weight_graph'):
    # This graph pattern is in inception-v1-12:
    # https://github.com/onnx/models/tree/main/vision/classification/inception_and_googlenet/inception_v1
    #
    # reshaped_X      reshaped_W
    #          \     /
    #           GEMM

    # IO tensors (ValueInfoProto).
    model_input_name = "X"
    model_input_channels = 10
    X = onnx.helper.make_tensor_value_info(model_input_name,
                                           onnx.TensorProto.FLOAT,
                                           [1, model_input_channels])
    model_output_name = "Y"
    model_output_channels = 5
    Y = onnx.helper.make_tensor_value_info(model_output_name,
                                           onnx.TensorProto.FLOAT,
                                           [1, model_output_channels])

    w_tensor = create_initializer_tensor(
        name="W",
        tensor_array=np.random.standard_normal(
            [1, 1, model_input_channels, model_output_channels]),
        data_type=onnx.TensorProto.FLOAT)

    w_shape_tensor = create_initializer_tensor(
        name="w_shape",
        tensor_array=np.array([model_input_channels, model_output_channels]),
        data_type=onnx.TensorProto.INT64)

    z_tensor = create_initializer_tensor(
        name="z_tensor",
        tensor_array=np.random.standard_normal([1, model_input_channels]),
        data_type=onnx.TensorProto.FLOAT)

    reshaped_w_node = onnx.helper.make_node(
        op_type="Reshape",
        inputs=["W", "w_shape"],
        outputs=["reshaped_w"],
    )

    added_x_node = onnx.helper.make_node(
        op_type="Add",
        inputs=["X", "z_tensor"],
        outputs=["added_x"],
    )

    gemm_node = onnx.helper.make_node(
        'Gemm',
        inputs=['added_x', 'reshaped_w'],
        outputs=['logit']
    )

    softmax_node = onnx.helper.make_node(
        'Softmax',
        inputs=['logit'],
        outputs=['Y'],
    )

    graph_def = onnx.helper.make_graph(
        nodes=[reshaped_w_node, added_x_node, gemm_node, softmax_node],
        name="Net",
        inputs=[X],
        outputs=[Y],
        initializer=[w_tensor, w_shape_tensor, z_tensor],
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="onnx-example")
    model_def.opset_import[0].version = 13

    model_def = onnx.shape_inference.infer_shapes(model_def)

    onnx.checker.check_model(model_def)

    yield TestCase(input_shape=[1, model_input_channels], model=model_def)


#def test_fxt_reshape_weight_graph(fxt_reshape_weight_graph: TestCase):
#    input_shape = fxt_reshape_weight_graph.input_shape
#    model = fxt_reshape_weight_graph.model
#
#    # TODO: PTQ succeeds, but this raises errors. Need to revisit.
#    # onnx.save(quantized_model, "tmp.onnx")
#    # infer_model(input_shape, quantized_model)
