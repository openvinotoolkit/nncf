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
            self._insert_quantizer_dequantizer(transformation)
            self.model.transformations.append(transformation)
        if isinstance(transformation, ONNXUpdateBias):
            self._update_bias(transformation)
            self.model.transformations.append(transformation)

    def _update_bias(self, transformation: ONNXUpdateBias):

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

        i = onnx_graph.get_node_index(n)
        self.model.compressed_model.graph.node.remove(node)
        self.model.compressed_model.graph.initializer.extend([onnx_bias])
        self.model.compressed_model.graph.node.insert(i, new_node)

    def _insert_quantizer_dequantizer(self, transformation: ONNXQuantizerInsertionCommand):
        target_point = transformation.target_point
        scale = transformation.quantizer_parameters.scale
        zero_point = transformation.quantizer_parameters.zero_point
        mode = transformation.quantizer_parameters.mode

        per_channel = True if isinstance(scale, list) else False

        zero_point = [zero_point] if not isinstance(zero_point, list) else zero_point
        tensor_type = onnx.TensorProto.UINT8 if mode == QuantizationMode.ASYMMETRIC else onnx.TensorProto.INT8
        scale = [scale] if not isinstance(scale, list) else scale
        axis = 0 if per_channel else None

        quantizer_name = 'QuantizeLinear_' + target_point
        dequantizer_name = 'DequantizeLinear_' + target_point
        scale_tensor_name = 'scale_' + target_point
        zero_point_tensor_name = 'zero_point_' + target_point

        onnx_scale = onnx.helper.make_tensor(scale_tensor_name, onnx.TensorProto.FLOAT, (len(scale),), scale)
        onnx_zero_point = onnx.helper.make_tensor(zero_point_tensor_name, tensor_type, (len(scale),), zero_point)

        quantizer = onnx.helper.make_node(
            'QuantizeLinear',
            [target_point, scale_tensor_name, zero_point_tensor_name],  # inputs
            ['q_output_' + target_point],  # outputs
            name=quantizer_name,
            axis=axis
        )
        dequantizer = onnx.helper.make_node(
            'DequantizeLinear',
            ['q_output_' + target_point, scale_tensor_name, zero_point_tensor_name],  # inputs
            ['dq_output_' + target_point],  # outputs
            name=dequantizer_name,
            axis=axis,
        )

        # TODO:NEED TO ADJUST LOGIC FOR INCEPTION_v3
        onnx_graph = ONNXGraph(self.model.compressed_model)
        try:
            input_nodes = onnx_graph.get_nodes_by_input(target_point)
        except RuntimeError as e:
            print(e)
            # TODO:SKIP THE BAD NODE
            return

        for node in input_nodes:
            for i, inp in enumerate(node.input):
                if inp == target_point:
                    node.input[i] = 'dq_output_' + target_point

        self.model.compressed_model.graph.initializer.extend([onnx_scale])
        self.model.compressed_model.graph.initializer.extend([onnx_zero_point])
        insert_index = onnx_graph.get_node_index(input_nodes[0].name)
        self.model.compressed_model.graph.node.insert(insert_index, quantizer)
        self.model.compressed_model.graph.node.insert(insert_index + 1, dequantizer)
