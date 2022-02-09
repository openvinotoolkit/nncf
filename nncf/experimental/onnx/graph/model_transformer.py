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

from nncf.experimental.onnx.graph.onnx_graph import ONNXGraph

from nncf.experimental.post_training.graph.model_transformer import ModelTransformer
from nncf.experimental.onnx.graph.transformations.layout import ONNXTransformationLayout
from nncf.experimental.onnx.compressed_model import CompressedModel

from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.experimental.onnx.graph.transformations.commands import ONNXQuantizerInsertionCommand
from nncf.experimental.onnx.graph.transformations.commands import ONNXInsertionCommand


class ONNXModelTransformer(ModelTransformer):
    def __init__(self, model: CompressedModel):
        self.model = model
        original_model = self.model.original_model
        self.model.compressed_model = deepcopy(original_model)

    def transform(self, model: CompressedModel, transformation_layout: ONNXTransformationLayout) -> CompressedModel:
        for transform in transformation_layout.transformations:
            self._apply_transformation(transform)
        return model

    def _apply_transformation(self, transformation: TransformationCommand):
        if isinstance(transformation, ONNXQuantizerInsertionCommand):
            self._insert_quantizer(transformation)
            self.model.transformations.append(transformation)

    def _insert_quantizer(self, transformation: ONNXInsertionCommand):
        def find_node_index(node_name, onnx_model):
            for i, node in enumerate(onnx_model.graph.node):
                if node.name == node_name:
                    return i
            return 0

        if transformation.parameters[0].size != 1:
            per_channel = True
        else:
            per_channel = False

        scale = transformation.parameters[0]
        zero_points = transformation.parameters[1]

        target_point = transformation.target_point

        quantizer_name = 'QuantizeLinear_' + target_point
        dequantizer_name = 'DequantizeLinear_' + target_point

        if per_channel:
            onnx_scale = onnx.helper.make_tensor('scale_' + target_point, onnx.TensorProto.FLOAT, scale.shape, scale)
            onnx_zero_point = onnx.helper.make_tensor('zero_point_' + target_point, onnx.TensorProto.INT8, scale.shape,
                                                      zero_points)
            axis = 0
            quantizer = onnx.helper.make_node(
                'QuantizeLinear',
                [target_point, 'scale_' + target_point, 'zero_point_' + target_point],  # inputs
                ['q_output_' + target_point],  # outputs
                name=quantizer_name,
                axis=axis
            )
            dequantizer = onnx.helper.make_node(
                'DequantizeLinear',
                ['q_output_' + target_point, 'scale_' + target_point, 'zero_point_' + target_point],  # inputs
                ['dq_output_' + target_point],  # outputs
                name=dequantizer_name,
                axis=axis
            )
        else:
            onnx_scale = onnx.helper.make_tensor('scale_' + target_point, onnx.TensorProto.FLOAT, [], [scale])
            onnx_zero_point = onnx.helper.make_tensor('zero_point_' + target_point, onnx.TensorProto.INT8, [],
                                                      [zero_points])
            quantizer = onnx.helper.make_node(
                'QuantizeLinear',
                [target_point, 'scale_' + target_point, 'zero_point_' + target_point],  # inputs
                ['q_output_' + target_point],  # outputs
                name=quantizer_name
            )
            dequantizer = onnx.helper.make_node(
                'DequantizeLinear',
                ['q_output_' + target_point, 'scale_' + target_point, 'zero_point_' + target_point],  # inputs
                ['dq_output_' + target_point],  # outputs
                name=dequantizer_name
            )

        onnx_graph = ONNXGraph(self.model.compressed_model)
        input_nodes = onnx_graph.get_nodes_by_input(target_point)

        for node in input_nodes:
            for i, inp in enumerate(node.input):
                if inp == target_point:
                    node.input[i] = 'dq_output_' + target_point

        self.model.compressed_model.graph.initializer.extend([onnx_scale])
        self.model.compressed_model.graph.initializer.extend([onnx_zero_point])
        i = find_node_index(input_nodes[0].name, self.model.compressed_model)
        self.model.compressed_model.graph.node.insert(i, quantizer)
        self.model.compressed_model.graph.node.insert(i + 1, dequantizer)
