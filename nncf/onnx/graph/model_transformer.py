"""
 Copyright (c) 2023 Intel Corporation
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
from typing import List, Optional, Tuple, Set

from copy import deepcopy
from collections import Counter
import onnx
import numpy as np

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.onnx.graph.onnx_graph import ONNXGraph
from nncf.onnx.graph.transformations.commands import ONNXBiasCorrectionCommand
from nncf.onnx.graph.transformations.commands import ONNXModelExtractionCommand
from nncf.onnx.graph.transformations.commands import ONNXOutputInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXQuantizerInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXQDQNodeRemovingCommand
from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.model_transformer import ModelTransformer


# pylint: disable=no-member
class ONNXModelTransformer(ModelTransformer):
    QUANTIZER_NAME_PREFIX = 'QuantizeLinear_'
    DEQUANTIZER_NAME_PREFIX = 'DequantizeLinear_'
    SCALE_TENSOR_NAME_PREFIX = 'scale_'
    ZERO_POINT_NAME_PREFIX = 'zero_point_'

    def __init__(self, model: onnx.ModelProto):
        super().__init__(model)
        self._model = deepcopy(model)
        self._nncf_graph = NNCFGraphFactory.create(self._model)

    def transform(self, transformation_layout: TransformationLayout) -> onnx.ModelProto:
        """
        Applies transformations by type-callback on the model

        :param transformations: lisf of the TransformationCommand transformations
        """
        quantizer_insert_transformations = []
        output_insert_transformations = []
        bias_correction_transformations = []
        qdq_node_removing_transformations = []
        model_extraction_transformation = None

        transformations = transformation_layout.transformations

        for transformation in transformations:
            if isinstance(transformation, ONNXQuantizerInsertionCommand):
                quantizer_insert_transformations.append(transformation)
            elif isinstance(transformation, ONNXOutputInsertionCommand):
                output_insert_transformations.append(transformation)
            elif isinstance(transformation, ONNXBiasCorrectionCommand):
                bias_correction_transformations.append(transformation)
            elif isinstance(transformation, ONNXModelExtractionCommand):
                model_extraction_transformation = transformation
            elif isinstance(transformation, ONNXQDQNodeRemovingCommand):
                qdq_node_removing_transformations.append(transformation)

        if quantizer_insert_transformations:
            self._apply_quantizer_insertion_transformations(quantizer_insert_transformations)
        if output_insert_transformations:
            self._apply_output_insertion_transformations(output_insert_transformations)
        if bias_correction_transformations:
            self._apply_bias_correction_transformations(bias_correction_transformations)
        if model_extraction_transformation:
            self._model = self._apply_model_extraction_transformation(model_extraction_transformation)
        if qdq_node_removing_transformations:
            self._apply_qdq_node_removing_transformation(qdq_node_removing_transformations)

        return self._model

    def _apply_output_insertion_transformations(self, transformations: List[ONNXOutputInsertionCommand]) -> None:
        """
        Applies incoming transformations to the model

        :param transformations: list of the ONNXOutputInsertionCommand transformations
        """
        onnx_graph = ONNXGraph(self._model)
        nncf_graph = NNCFGraphFactory.create(self._model)
        model_outputs = [output.name for output in onnx_graph.get_model_outputs()]
        extra_model_outputs = self._get_extra_model_outputs(nncf_graph,
                                                            onnx_graph,
                                                            transformations)

        model_with_intermediate_outputs = self._insert_outputs(self._model,
                                                               outputs=[*extra_model_outputs,
                                                                        *model_outputs])
        self._model = model_with_intermediate_outputs

    def _get_extra_model_outputs(self,
                                 nncf_graph: NNCFGraph,
                                 onnx_graph: ONNXGraph,
                                 transformations: List[ONNXOutputInsertionCommand]) -> Set[str]:
        """
        Collects extra model outputs based on transformations

        :param nncf_graph: NNCFGraph
        :param onnx_graph: ONNXGraph
        :param transformations: lisf of the ONNXOutputInsertionCommand
        :return: list of the output names
        """
        extra_model_outputs = set()
        input_edge_names = []

        for transformation in transformations:
            node_name = transformation.target_point.target_node_name
            if NNCFGraphNodeType.INPUT_NODE in node_name:
                nncf_node_name = nncf_graph.get_node_by_name(transformation.target_point.target_node_name)
                onnx_nodes_after_input_node = [edge.to_node for edge in nncf_graph.get_output_edges(nncf_node_name)]
                for onnx_node_name in onnx_nodes_after_input_node:
                    input_edge_names.append(onnx_graph.get_node_edge_names(onnx_node_name.node_name)['input'][0])
                extra_model_outputs.update(input_edge_names)
                input_edge_names = []
            else:
                if transformation.target_point.type == TargetType.POST_LAYER_OPERATION:
                    edge_name = onnx_graph.get_node_edge_names(node_name)['output'][
                        transformation.target_point.port_id]
                elif transformation.target_point.type == TargetType.PRE_LAYER_OPERATION:
                    edge_name = onnx_graph.get_node_edge_names(node_name)['input'][
                        transformation.target_point.port_id]
                else:
                    raise RuntimeError
                extra_model_outputs.add(edge_name)
            extra_model_outputs.update(input_edge_names)
        return extra_model_outputs

    def _insert_outputs(self, model: onnx.ModelProto, outputs: List[str] = None) -> onnx.ModelProto:
        """
        Takes a model and adds outputs based on the list of edge names to collect data from

        :param model: *ONNX* model
        :param outputs: edge names to collect data from
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
            value_info = onnx.helper.make_value_info(
                name=out, type_proto=type_proto)
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

    def _apply_quantizer_insertion_transformations(
            self,
            transformations: List[ONNXQuantizerInsertionCommand]) -> None:
        """
        Applies transformations on the model

        :param transformations: lisf of the TransformationCommand transformations
        """
        self._added_target_edges = Counter()
        onnx_graph = ONNXGraph(self._model)
        for transformation in transformations:
            self._insert_quantizer_dequantizer(transformation, onnx_graph)

    def _get_target_edge_name(self, transformation: ONNXQuantizerInsertionCommand, onnx_graph: ONNXGraph) -> \
            Optional[str]:
        target_edge_name = None
        if transformation.target_point.type == TargetType.OPERATION_WITH_WEIGHTS:
            target_edge_name = onnx_graph.get_node_edge_names(transformation.target_point.target_node_name)['input'][
                transformation.target_point.port_id]
        elif transformation.target_point.type == TargetType.PRE_LAYER_OPERATION:
            target_edge_name = onnx_graph.get_node_edge_names(transformation.target_point.target_node_name)['input'][
                transformation.target_point.port_id]
        elif transformation.target_point.type == TargetType.POST_LAYER_OPERATION:
            if NNCFGraphNodeType.INPUT_NODE in transformation.target_point.target_node_name:  # ADD INPUT NODE CASE
                nncf_node_name = self._nncf_graph.get_node_by_name(transformation.target_point.target_node_name)
                onnx_nodes_after_input_node = [edge.to_node for edge in
                                               self._nncf_graph.get_output_edges(nncf_node_name)]
                for onnx_node_name in onnx_nodes_after_input_node:
                    target_edge_name = onnx_graph.get_node_edge_names(onnx_node_name.node_name)['input'][
                        transformation.target_point.port_id]
                    break
            else:
                target_edge_name = onnx_graph.get_node_edge_names(transformation.target_point.target_node_name)[
                    'output'][transformation.target_point.port_id]
        else:
            raise RuntimeError(
                'Could not find the edge corresponding to node {}'.format(
                    transformation.target_point.target_node_name))
        self._added_target_edges[target_edge_name] += 1
        return target_edge_name

    def _get_quantize_dequantize_nodes(self, transformation: ONNXQuantizerInsertionCommand, target_edge_name: str) -> \
            Tuple[onnx.NodeProto, onnx.NodeProto]:
        axis = transformation.quantizer_parameters.axis

        cnt = self._added_target_edges[target_edge_name]

        input_target_edge = target_edge_name
        q_target_edge_name = target_edge_name + '_' + str(cnt)
        quantizer_name = ONNXModelTransformer.QUANTIZER_NAME_PREFIX + q_target_edge_name
        dequantizer_name = ONNXModelTransformer.DEQUANTIZER_NAME_PREFIX + q_target_edge_name
        scale_tensor_name = ONNXModelTransformer.SCALE_TENSOR_NAME_PREFIX + q_target_edge_name
        zero_point_tensor_name = ONNXModelTransformer.ZERO_POINT_NAME_PREFIX + q_target_edge_name

        quantizer = onnx.helper.make_node(
            name=quantizer_name,
            op_type='QuantizeLinear',
            inputs=[input_target_edge, scale_tensor_name, zero_point_tensor_name],
            outputs=['q_output_' + q_target_edge_name],
            axis=axis
        )

        dequantizer = onnx.helper.make_node(
            name=dequantizer_name,
            op_type='DequantizeLinear',
            inputs=['q_output_' + q_target_edge_name, scale_tensor_name, zero_point_tensor_name],
            outputs=['dq_output_' + q_target_edge_name],
            axis=axis,
        )

        return quantizer, dequantizer

    def _get_scale_zero_point_tensors(self, transformation: ONNXQuantizerInsertionCommand, quantizer: onnx.NodeProto,
                                      dequantizer: onnx.NodeProto) -> Tuple[onnx.TensorProto, onnx.TensorProto]:
        scale = transformation.quantizer_parameters.scale
        zero_point = transformation.quantizer_parameters.zero_point
        tensor_type = transformation.quantizer_parameters.tensor_type

        per_channel = scale.ndim > 0
        dims = scale.shape if per_channel else []
        onnx_scale = [scale.tolist()] if not per_channel else scale.tolist()
        onnx_zero_point = [zero_point.tolist()] if not per_channel else zero_point.tolist()
        if tensor_type == np.uint8:
            onnx_tensor_type = onnx.TensorProto.UINT8
        elif tensor_type == np.int8:
            onnx_tensor_type = onnx.TensorProto.INT8
        else:
            raise RuntimeError('Incorrect tensor type.')
        assert quantizer.input[1] == dequantizer.input[1] and quantizer.input[2] == dequantizer.input[2]
        scale_tensor_name = quantizer.input[1]
        zero_point_tensor_name = quantizer.input[2]
        onnx_scale_tensor = onnx.helper.make_tensor(scale_tensor_name, onnx.TensorProto.FLOAT, dims, onnx_scale)
        onnx_zero_point_tensor = onnx.helper.make_tensor(zero_point_tensor_name, onnx_tensor_type, dims,
                                                         onnx_zero_point)
        return onnx_scale_tensor, onnx_zero_point_tensor

    def _insert_quantizer_dequantizer(self, transformation: ONNXQuantizerInsertionCommand,
                                      onnx_graph: ONNXGraph) -> None:
        target_edge_name = self._get_target_edge_name(transformation, onnx_graph)
        quantizer, dequantizer = self._get_quantize_dequantize_nodes(transformation, target_edge_name)
        onnx_scale_tensor, onnx_zero_point_tensor = self._get_scale_zero_point_tensors(transformation, quantizer,
                                                                                       dequantizer)

        # If several nodes on one edge
        input_nodes = []
        input_nodes.extend(onnx_graph.get_nodes_by_input(target_edge_name))
        if not input_nodes:
            raise RuntimeError(
                f'Can not add the quantizer to the {target_edge_name} edge. This edge does not have end node.')

        if transformation.target_point.type == TargetType.PRE_LAYER_OPERATION:
            # If we need to change only target nodes input
            target_node = onnx_graph.get_node_by_name(transformation.target_point.target_node_name)
            for i, inp in enumerate(target_node.input):
                if inp == target_edge_name:
                    target_node.input[i] = dequantizer.output[0]
        else:
            for node in input_nodes:
                for i, inp in enumerate(node.input):
                    if inp == target_edge_name:
                        node.input[i] = dequantizer.output[0]

        onnx_scale_value_info = onnx.helper.make_tensor_value_info(onnx_scale_tensor.name, onnx_scale_tensor.data_type,
                                                                   onnx_scale_tensor.dims)
        onnx_zero_point_info = onnx.helper.make_tensor_value_info(onnx_zero_point_tensor.name,
                                                                  onnx_zero_point_tensor.data_type,
                                                                  onnx_zero_point_tensor.dims)
        self._model.graph.initializer.extend([onnx_scale_tensor, onnx_zero_point_tensor])
        self._model.graph.value_info.extend([onnx_scale_value_info, onnx_zero_point_info])
        insert_index = onnx_graph.get_node_index(input_nodes[0].name)
        self._model.graph.node.insert(insert_index, quantizer)
        self._model.graph.node.insert(insert_index + 1, dequantizer)

    def _apply_bias_correction_transformations(self, transformations: List[ONNXBiasCorrectionCommand]) -> None:
        """
        Applies bias correction transformations on the model

        :param transformations: lisf of the bias correction transformations
        """
        onnx_graph = ONNXGraph(self._model)
        for transformation in transformations:
            bias_tensor_position = transformation.target_point.port_id
            node_name = transformation.target_point.target_node_name
            onnx_node = onnx_graph.get_node_by_name(node_name)
            bias_initializer_name = onnx_node.input[bias_tensor_position]
            bias_initializer = onnx_graph.get_initializer(bias_initializer_name)

            new_bias_tensor = onnx.numpy_helper.from_array(transformation.bias_value,
                                                           bias_initializer_name)
            bias_initializer.CopyFrom(new_bias_tensor)

    def _apply_model_extraction_transformation(self, transformation: ONNXModelExtractionCommand) -> onnx.ModelProto:
        """
        Extracts sub-model from the original based on the inputs and outputs names

        :param transformation: model extraction transformation
        """
        onnx_model_exctactor = onnx.utils.Extractor(self._model)
        return onnx_model_exctactor.extract_model(transformation.inputs, transformation.outputs)

    def _apply_qdq_node_removing_transformation(self, transformations: List[ONNXQDQNodeRemovingCommand]) -> None:
        """
        Removes the layers from the model.

        :param transformations: lisf of the node removing transformations.
        """
        onnx_graph = ONNXGraph(self._model)
        for transformation in transformations:
            node = onnx_graph.get_node_by_name(transformation.target_point.target_node_name)

            node_children = onnx_graph.get_children(node)
            for node_child in node_children:
                for input_id, input_obj in enumerate(node_child.input):
                    if input_obj == node.output[0]:
                        node_child.input[input_id] = node.input[0]

            initializers = {i.name: i for i in self._model.graph.initializer}
            value_infos = {i.name: i for i in self._model.graph.value_info}
            for initializer_name in node.input:
                if initializer_name in initializers:
                    self._model.graph.initializer.remove(initializers[initializer_name])
                if initializer_name in value_infos:
                    self._model.graph.value_info.remove(value_infos[initializer_name])

            self._model.graph.node.remove(node)
