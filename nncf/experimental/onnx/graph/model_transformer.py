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
from copy import deepcopy
import onnx
import numpy as np
from nncf.experimental.onnx.graph.onnx_graph import ONNXGraph

from nncf.experimental.post_training.graph.model_transformer import ModelTransformer
from nncf.experimental.onnx.graph.transformations.layout import ONNXTransformationLayout
from nncf.experimental.onnx.compressed_model import CompressedModel

from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.experimental.onnx.graph.transformations.commands import ONNXInsertionCommand


class ONNXModelTransformer(ModelTransformer):
    def __init__(self, model: CompressedModel):
        self.model = model
        self.model.transformed_model = deepcopy(self.model.original_model)

    def transform(self, model: CompressedModel, transformation_layout: ONNXTransformationLayout) -> CompressedModel:
        for transform in transformation_layout.transformations:
            self._apply_transformation(transform)
        return model

    def _apply_transformation(self, transformation: ONNXInsertionCommand):
        q_name, dq_name = self._insert(transformation.target_point, transformation.tensor)
        transformation.q_name = q_name
        transformation.dq_name = dq_name
        self.model.transformations.append(transformation)

    def _insert(self, target_point, tensor):
        def find_node_index(node_name, onnx_model):
            for i, node in enumerate(onnx_model.graph.node):
                if node.name == node_name:
                    return i
            return 0

        name = target_point
        shape = tensor.shape
        zero_point = np.zeros_like(tensor).tolist()
        # scale = np.ones_like(tensor).tolist()
        onnx_scale = onnx.helper.make_tensor('scale_' + name, onnx.TensorProto.FLOAT, [], [1])
        # onnx_zero_point = onnx.helper.make_tensor('zero_point_' + name, onnx.TensorProto.UINT8, [], 0)
        quantizer = onnx.helper.make_node(
            'QuantizeLinear',  # name
            [name, 'scale_' + name, 'zero_point_' + name],  # inputs
            ['q_output_' + name],  # outputs
            name='QuantizeLinear' + name,
            axis=0
        )

        dequantizer = onnx.helper.make_node(
            'DequantizeLinear',  # name
            ['q_output_' + name, 'scale_' + name, 'zero_point_' + name],  # inputs
            ['dq_output_' + name],  # outputs
            name='QuantizeLinear' + name,
            axis=0
        )
        onnx_graph = ONNXGraph(self.model.transformed_model)
        input_nodes = onnx_graph.get_nodes_by_input(name)

        for node in input_nodes:
            for i, inp in enumerate(node.input):
                if inp == name:
                    node.input[i] = 'dq_output_' + name
        self.model.transformed_model.graph.initializer.extend([onnx_scale])
        # self.model.transformed_model.graph.initializer.extend([onnx_zero_point])
        i = find_node_index(input_nodes[0].name, self.model.transformed_model)
        self.model.transformed_model.graph.node.insert(i, quantizer)
        self.model.transformed_model.graph.node.insert(i + 1, dequantizer)

    def update_quantizer_parameters(self, quantizer_name):
