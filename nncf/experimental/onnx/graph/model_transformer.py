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

from typing import TypeVar

from copy import deepcopy
import onnx

from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.quantization.structs import QuantizationMode

from nncf.experimental.onnx.graph.onnx_graph import ONNXGraph

from nncf.experimental.onnx.graph.transformations.layout import ONNXTransformationLayout

from nncf.experimental.onnx.graph.transformations.commands import ONNXQuantizerInsertionCommand

ModelType = TypeVar('ModelType')


# pylint: disable=no-member

class ONNXModelTransformer(ModelTransformer):
    QUANTIZER_NAME_PREFIX = 'QuantizeLinear_'
    DEQUANTIZER_NAME_PREFIX = 'DequantizeLinear_'
    SCALE_TENSOR_NAME_PREFIX = 'scale_'
    ZERO_POINT_NAME_PREFIX = 'zero_point_'

    def __init__(self, model: onnx.ModelProto):
        super().__init__(model)
        self.transformed_model = deepcopy(model)

    def transform(self, transformation_layout: ONNXTransformationLayout) -> onnx.ModelProto:
        for transform in transformation_layout.transformations:
            self._apply_transformation(transform)
        return self.transformed_model

    def _apply_transformation(self, transformation: TransformationCommand):
        if isinstance(transformation, ONNXQuantizerInsertionCommand):
            self._insert_quantizer_dequantizer(transformation)

    def _insert_quantizer_dequantizer(self, transformation: ONNXQuantizerInsertionCommand):
        onnx_graph = ONNXGraph(self.transformed_model)
        if transformation.target_point.type == TargetType.OPERATION_WITH_WEIGHTS:
            target_edge_name = onnx_graph.get_weight_tensor_with_initializer(
                transformation.target_point.target_node_name)
            if target_edge_name is None:
                # TODO (kshpv): need to discover whether we could delete checking weight_initializer_name on None
                # the same as in ONNXMinMaxQuantization
                return
        elif transformation.target_point.type == TargetType.PRE_LAYER_OPERATION:
            target_edge_name = onnx_graph.get_node_edges(transformation.target_point.target_node_name)['input'][0]
        elif transformation.target_point.type == TargetType.POST_LAYER_OPERATION:
            target_edge_name = onnx_graph.get_node_edges(transformation.target_point.target_node_name)['output'][0]
        else:
            raise RuntimeError(
                'Could not find the edge corresponding to node {}'.format(
                    transformation.target_point.target_node_name))
        scale = transformation.quantizer_parameters.scale
        zero_point = transformation.quantizer_parameters.zero_point
        mode = transformation.quantizer_parameters.mode

        per_channel = isinstance(scale, list)

        zero_point = [zero_point] if not isinstance(zero_point, list) else zero_point
        tensor_type = onnx.TensorProto.UINT8 if mode == QuantizationMode.ASYMMETRIC else onnx.TensorProto.INT8
        scale = [scale] if not isinstance(scale, list) else scale

        axis = 0 if per_channel else None
        dims = [len(scale)] if per_channel else []

        quantizer_name = ONNXModelTransformer.QUANTIZER_NAME_PREFIX + target_edge_name
        dequantizer_name = ONNXModelTransformer.DEQUANTIZER_NAME_PREFIX + target_edge_name
        scale_tensor_name = ONNXModelTransformer.SCALE_TENSOR_NAME_PREFIX + target_edge_name
        zero_point_tensor_name = ONNXModelTransformer.ZERO_POINT_NAME_PREFIX + target_edge_name

        onnx_scale = onnx.helper.make_tensor(scale_tensor_name, onnx.TensorProto.FLOAT, dims, scale)
        onnx_zero_point = onnx.helper.make_tensor(zero_point_tensor_name, tensor_type, dims, zero_point)

        quantizer = onnx.helper.make_node(
            'QuantizeLinear',
            [target_edge_name, scale_tensor_name, zero_point_tensor_name],  # inputs
            ['q_output_' + target_edge_name],  # outputs
            name=quantizer_name,
            axis=axis
        )
        dequantizer = onnx.helper.make_node(
            'DequantizeLinear',
            ['q_output_' + target_edge_name, scale_tensor_name, zero_point_tensor_name],  # inputs
            ['dq_output_' + target_edge_name],  # outputs
            name=dequantizer_name,
            axis=axis,

        )

        # TODO (kshpv): need to carefully look through the logic of nodes searching.
        #  The model with the possible issues is inception_v3.

        input_nodes = onnx_graph.get_nodes_by_input(target_edge_name)
        if not input_nodes:
            raise RuntimeError(
                f'Can not add the quantizer to the {target_edge_name} edge. This edge does not have end node.')

        for node in input_nodes:
            for i, inp in enumerate(node.input):
                if inp == target_edge_name:
                    node.input[i] = 'dq_output_' + target_edge_name

        self.transformed_model.graph.initializer.extend([onnx_scale])
        self.transformed_model.graph.initializer.extend([onnx_zero_point])
        insert_index = onnx_graph.get_node_index(input_nodes[0].name)
        self.transformed_model.graph.node.insert(insert_index, quantizer)
        self.transformed_model.graph.node.insert(insert_index + 1, dequantizer)
