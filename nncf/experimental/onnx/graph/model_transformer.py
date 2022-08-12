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
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.utils.logger import logger as nncf_logger
from nncf.experimental.onnx.graph.onnx_graph import ONNXGraph
from nncf.experimental.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.experimental.onnx.graph.transformations.layout import ONNXTransformationLayout
from nncf.experimental.onnx.graph.transformations.commands import ONNXQuantizerInsertionCommand
from nncf.experimental.onnx.graph.transformations.commands import ONNXOutputInsertionCommand

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
        self.quantizer_insertion_commands = []  # type: List[ONNXQuantizerInsertionCommand]
        self.output_insertion_commands = []  # type: List[ONNXOutputInsertionCommand]

    def transform(self, transformation_layout: ONNXTransformationLayout) -> onnx.ModelProto:
        for transformation in transformation_layout.transformations:
            if isinstance(transformation, ONNXQuantizerInsertionCommand):
                self._add_quantizer_insertion_transformation(transformation)
            elif isinstance(transformation, ONNXOutputInsertionCommand):
                self._add_output_transformation(transformation)
        self._apply_transformations()
        return self.transformed_model

    def _add_quantizer_insertion_transformation(self, transformation: ONNXQuantizerInsertionCommand):
        self.quantizer_insertion_commands.append(transformation)

    def _add_output_transformation(self, transformation: ONNXOutputInsertionCommand):
        self.output_insertion_commands.append(transformation)

    def _apply_transformations(self):
        if self.quantizer_insertion_commands:
            self._apply_quantizer_insertion_transformations()
            self.quantizer_insertion_commands = []
        if self.output_insertion_commands:
            self._apply_outputs_transformations()
            self.output_insertion_commands = []

    def _apply_outputs_transformations(self):
        def select_model_inputs_outputs(model, outputs=None):
            """
            Takes a model and changes its outputs.

            :param model: *ONNX* model
            :param outputs: new outputs
            :return: modified model
            """
            if outputs is None:
                raise RuntimeError("Parameter outputs cannot be None.")
            onnx_graph = ONNXGraph(model)
            var_out = []
            for out in outputs:
                # shape should be None; if you place not None, some models will have inference problems (e.g. Mask RCNN)
                type_proto = onnx.helper.make_tensor_type_proto(onnx_graph.get_edge_dtype(out),
                                                                shape=None)
                value_info = onnx.helper.make_value_info(name=out, type_proto=type_proto)
                var_out.append(value_info)

            graph = onnx.helper.make_graph(nodes=model.graph.node,
                                           name=model.graph.name,
                                           inputs=model.graph.input,
                                           outputs=var_out,
                                           initializer=model.graph.initializer,
                                           value_info=model.graph.value_info)
            onnx_model = onnx.helper.make_model(graph,
                                                ir_version=model.ir_version,
                                                producer_name=model.producer_name,
                                                producer_version=model.producer_version,
                                                domain=model.domain,
                                                model_version=model.model_version,
                                                doc_string=model.doc_string)
            if len(model.metadata_props) > 0:
                values = {p.key: p.value for p in model.metadata_props}
                onnx.helper.set_model_props(onnx_model, values)

            if len(onnx_model.graph.input) != len(model.graph.input):
                raise RuntimeError("Input mismatch {} != {}".format(
                    len(onnx_model.input), len(model.input)))
            # fix opset import
            del onnx_model.opset_import[:]
            for oimp in model.opset_import:
                op_set = onnx_model.opset_import.add()
                op_set.domain = oimp.domain
                op_set.version = oimp.version
            return onnx_model

        onnx_graph = ONNXGraph(self.transformed_model)
        nncf_graph = GraphConverter.create_nncf_graph(self.transformed_model)
        model_outputs = [output.name for output in onnx_graph.get_model_outputs()]
        extra_model_outputs = []
        input_edge_names = []

        for transformation in self.output_insertion_commands:

            node_name = transformation.target_point.target_node_name
            if NNCFGraphNodeType.INPUT_NODE in node_name:
                nncf_node_name = nncf_graph.get_node_by_name(transformation.target_point.target_node_name)
                onnx_nodes_after_input_node = [edge.to_node for edge in nncf_graph.get_output_edges(nncf_node_name)]
                for onnx_node_name in onnx_nodes_after_input_node:
                    input_edge_names.append(onnx_graph.get_node_edges(onnx_node_name.node_name)['input'][0])
                extra_model_outputs.extend(input_edge_names)
                input_edge_names = []
            else:
                if transformation.target_point.type == TargetType.POST_LAYER_OPERATION:
                    edge_name = onnx_graph.get_node_edges(node_name)['output'][0]
                elif transformation.target_point.type == TargetType.PRE_LAYER_OPERATION:
                    edge_name = onnx_graph.get_node_edges(node_name)['input'][0]
                else:
                    raise RuntimeError
                extra_model_outputs.append(edge_name)
            extra_model_outputs.extend(input_edge_names)

        model_with_intermediate_outputs = select_model_inputs_outputs(self.transformed_model,
                                                                      outputs=[*extra_model_outputs,
                                                                               *model_outputs])
        self.transformed_model = model_with_intermediate_outputs

    def _apply_quantizer_insertion_transformations(self):
        # TODO: optimize
        for transformation in self.quantizer_insertion_commands:
            self._insert_quantizer_dequantizer(transformation)

    def _insert_quantizer_dequantizer(self, transformation: ONNXQuantizerInsertionCommand):
        # TODO (kshpv): remove many branches
        # pylint: disable=too-many-branches
        onnx_graph = ONNXGraph(self.transformed_model)
        target_edge_names = set()
        if transformation.target_point.type == TargetType.OPERATION_WITH_WEIGHTS:
            try:
                target_edge_names.add(onnx_graph.get_weight_tensor_with_initializer(
                    transformation.target_point.target_node_name))
            except RuntimeError as er:
                nncf_logger.exception(er)
                return
        elif transformation.target_point.type == TargetType.PRE_LAYER_OPERATION:
            target_edge_names.add(
                onnx_graph.get_node_edges(transformation.target_point.target_node_name)['input'][0])
        elif transformation.target_point.type == TargetType.POST_LAYER_OPERATION:
            if NNCFGraphNodeType.INPUT_NODE in transformation.target_point.target_node_name:  # ADD INPUT NODE CASE

                nncf_graph = GraphConverter.create_nncf_graph(self.transformed_model)
                nncf_node_name = nncf_graph.get_node_by_name(transformation.target_point.target_node_name)
                onnx_nodes_after_input_node = [edge.to_node for edge in nncf_graph.get_output_edges(nncf_node_name)]
                for onnx_node_name in onnx_nodes_after_input_node:
                    if onnx_graph.get_node_edges(onnx_node_name.node_name)['input'][0] not in target_edge_names:
                        target_edge_names.add(onnx_graph.get_node_edges(onnx_node_name.node_name)['input'][0])
            else:
                target_edge_names.add(
                    onnx_graph.get_node_edges(transformation.target_point.target_node_name)['output'][0])
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

        target_edge_name = next(iter(target_edge_names))
        quantizer_name = ONNXModelTransformer.QUANTIZER_NAME_PREFIX + target_edge_name
        dequantizer_name = ONNXModelTransformer.DEQUANTIZER_NAME_PREFIX + target_edge_name
        scale_tensor_name = ONNXModelTransformer.SCALE_TENSOR_NAME_PREFIX + target_edge_name
        zero_point_tensor_name = ONNXModelTransformer.ZERO_POINT_NAME_PREFIX + target_edge_name

        onnx_scale = onnx.helper.make_tensor(scale_tensor_name, onnx.TensorProto.FLOAT, dims, scale)
        onnx_zero_point = onnx.helper.make_tensor(zero_point_tensor_name, tensor_type, dims, zero_point)

        quantizer = onnx.helper.make_node(
            'QuantizeLinear',
            inputs=[target_edge_name, scale_tensor_name, zero_point_tensor_name],
            outputs=['q_output_' + target_edge_name],
            name=quantizer_name,
            axis=axis
        )
        # If several nodes on one edge
        dequantizer_outputs = ['dq_output_' + st for st in target_edge_names]
        dequantizer = onnx.helper.make_node(
            'DequantizeLinear',
            inputs=['q_output_' + target_edge_name, scale_tensor_name, zero_point_tensor_name],
            outputs=dequantizer_outputs,
            name=dequantizer_name,
            axis=axis,

        )

        # TODO (kshpv): need to carefully look through the logic of nodes searching.
        #  The model with the possible issues is inception_v3.
        # If several nodes on one edge
        for target_edge_name in target_edge_names:
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
