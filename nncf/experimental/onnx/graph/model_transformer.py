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

from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.quantization.structs import QuantizationMode

from nncf.experimental.onnx.graph.onnx_graph import ONNXGraph

from nncf.experimental.onnx.graph.transformations.layout import ONNXTransformationLayout
from nncf.experimental.post_training.compressed_model import CompressedModel

from nncf.experimental.onnx.graph.transformations.commands import ONNXQuantizerInsertionCommand


class ONNXModelTransformer(ModelTransformer):
    def __init__(self, model: CompressedModel):
        super().__init__(model)
        original_model = self._model.original_model
        self._model.compressed_model = deepcopy(original_model)

    def transform(self, transformation_layout: ONNXTransformationLayout) -> CompressedModel:
        for transform in transformation_layout.transformations:
            self._apply_transformation(transform)
        return self._model

    def _apply_transformation(self, transformation: TransformationCommand):
        if isinstance(transformation, ONNXQuantizerInsertionCommand):
            self._insert_quantizer_dequantizer(transformation)
            self._model.transformations.append(transformation)

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
        dims = [len(scale)] if per_channel else []

        quantizer_name = 'QuantizeLinear_' + target_point
        dequantizer_name = 'DequantizeLinear_' + target_point
        scale_tensor_name = 'scale_' + target_point
        zero_point_tensor_name = 'zero_point_' + target_point

        onnx_scale = onnx.helper.make_tensor(scale_tensor_name, onnx.TensorProto.FLOAT, dims, scale)
        onnx_zero_point = onnx.helper.make_tensor(zero_point_tensor_name, tensor_type, dims, zero_point)

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
        onnx_graph = ONNXGraph(self._model.compressed_model)
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

        self._model.compressed_model.graph.initializer.extend([onnx_scale])
        self._model.compressed_model.graph.initializer.extend([onnx_zero_point])
        insert_index = onnx_graph.get_node_index(input_nodes[0].name)
        self._model.compressed_model.graph.node.insert(insert_index, quantizer)
        self._model.compressed_model.graph.node.insert(insert_index + 1, dequantizer)
