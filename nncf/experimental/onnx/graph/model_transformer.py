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

from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.quantization.structs import QuantizationMode

from nncf.experimental.onnx.graph.onnx_graph import ONNXGraph

from nncf.experimental.post_training.graph.model_transformer import ModelTransformer
from nncf.experimental.onnx.graph.transformations.layout import ONNXTransformationLayout
from nncf.experimental.post_training.compressed_model import CompressedModel

from nncf.experimental.onnx.graph.transformations.commands import ONNXQuantizerInsertionCommand
from nncf.experimental.onnx.graph.transformations.commands import ONNXUpdateBias
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
        if isinstance(transformation, ONNXUpdateBias):
            self._update_bias(transformation)
            self.model.transformations.append(transformation)

    def _update_bias(self, transformation: ONNXUpdateBias):

        def find_node_index(node_name, onnx_model):
            for i, node in enumerate(onnx_model.graph.node):
                if node.name == node_name:
                    return i
            return 0

        onnx_graph = ONNXGraph(self.model.compressed_model)

        target_point = transformation.target_point
        bias_tensor = transformation.bias_tensor

        node = onnx_graph.get_node_by_name(target_point)

        bias_name = 'bias_' + target_point
        onnx_bias = onnx.helper.make_tensor(bias_name, onnx.TensorProto.FLOAT, bias_tensor.shape, bias_tensor)

        t = node.op_type
        n = node.name
        outputs = node.output
        inputs = node.input
        attrs = node.attribute

        node_inputs = [_input for _input in inputs]

        # Should be bias in BN and CONV layers
        if len(node.input) > 3:
            node_inputs[2] = bias_name
        else:
            node_inputs += [bias_name]

        node_attrs = {}

        for attr in node.attribute:
            attr_key = attr.name
            attr_value = onnx.helper.get_attribute_value(attr)
            node_attrs[attr_key] = attr_value

        new_node = onnx.helper.make_node(
            t,
            name=n,
            inputs=node_inputs,
            outputs=outputs,
            **node_attrs
        )
        print(f'bias_name = {bias_name}')

        i = find_node_index(n, self.model.compressed_model)
        self.model.compressed_model.graph.node.remove(node)
        self.model.compressed_model.graph.initializer.extend([onnx_bias])
        self.model.compressed_model.graph.node.insert(i, new_node)

    def _insert_quantizer(self, transformation: ONNXInsertionCommand):
        def find_node_index(node_name, onnx_model):
            for i, node in enumerate(onnx_model.graph.node):
                if node.name == node_name:
                    return i
            return 0

        if isinstance(transformation.parameters[0], list):
            per_channel = True
        else:
            per_channel = False

        scale = transformation.parameters[0]
        zero_points = transformation.parameters[1]
        mode = transformation.parameters[2]

        target_point = transformation.target_point

        quantizer_name = 'QuantizeLinear_' + target_point
        dequantizer_name = 'DequantizeLinear_' + target_point

        if per_channel:
            onnx_scale = onnx.helper.make_tensor('scale_' + target_point, onnx.TensorProto.FLOAT, (len(scale),), scale)
            if mode == QuantizationMode.ASYMMETRIC:
                onnx_zero_point = onnx.helper.make_tensor('zero_point_' + target_point, onnx.TensorProto.UINT8,
                                                          (len(scale),),
                                                          zero_points)
            else:
                onnx_zero_point = onnx.helper.make_tensor('zero_point_' + target_point, onnx.TensorProto.INT8,
                                                          (len(scale),),
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
                axis=axis,
            )
        else:
            onnx_scale = onnx.helper.make_tensor('scale_' + target_point, onnx.TensorProto.FLOAT, [], [scale])
            if mode == QuantizationMode.ASYMMETRIC:
                onnx_zero_point = onnx.helper.make_tensor('zero_point_' + target_point, onnx.TensorProto.UINT8, [],
                                                          [zero_points])
            else:
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

        # TODO:NEED TO ADJUST LOGIC FOR INCEPTION_v3
        onnx_graph = ONNXGraph(self.model.compressed_model)
        try:
            input_nodes = onnx_graph.get_nodes_by_input(target_point)
        except RuntimeError as e:
            print(e)
            # TODO:SKIP THE BAD NODE
            return
        #     input_nodes = onnx_graph.get_nodes_by_output(target_point)
        # finally:

        for node in input_nodes:
            for i, inp in enumerate(node.input):
                if inp == target_point:
                    node.input[i] = 'dq_output_' + target_point

        self.model.compressed_model.graph.initializer.extend([onnx_scale])
        self.model.compressed_model.graph.initializer.extend([onnx_zero_point])
        i = find_node_index(input_nodes[0].name, self.model.compressed_model)
        self.model.compressed_model.graph.node.insert(i, quantizer)
        self.model.compressed_model.graph.node.insert(i + 1, dequantizer)
