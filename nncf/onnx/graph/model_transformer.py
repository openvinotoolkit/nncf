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
from typing import Dict, List, Tuple

from copy import deepcopy
from collections import Counter
import onnx
import numpy as np

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.onnx.graph.onnx_graph import ONNXGraph
from nncf.onnx.graph.transformations.commands import ONNXBiasCorrectionCommand
from nncf.onnx.graph.transformations.commands import ONNXModelExtractionCommand
from nncf.onnx.graph.transformations.commands import ONNXOutputInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXQuantizerInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXQDQNodeRemovingCommand
from nncf.common.graph.model_transformer import ModelTransformer


class ONNXModelTransformer(ModelTransformer):
    """
    Applies transformations upon ONNX model.
    ModelTransformer should be created once for a particular model,
    and be used to apply transformations to the provided model.
    """

    QUANTIZER_NAME_PREFIX = 'QuantizeLinear_'
    DEQUANTIZER_NAME_PREFIX = 'DequantizeLinear_'
    SCALE_TENSOR_NAME_PREFIX = 'scale_'
    ZERO_POINT_NAME_PREFIX = 'zero_point_'

    def __init__(self, model: onnx.ModelProto):
        super().__init__(model)
        self.onnx_model_extractor = onnx.utils.Extractor(self._model)

    def _get_pre_post_target_edge(self, port_id: int, node_name: str, transform_type: TargetType,
                                  onnx_graph: ONNXGraph, nncf_input_node_next_nodes: Dict[str, str]) -> str:
        """
        Returns edge name corresponding to the node with a name equal to node_name, port_id and transform_type.

        :param port_id: Edge number of port.
        :param node_name: Node name.
        :param transform_type: Type of transformation.
        :param onnx_graph: ONNXGraph.
        :param nncf_input_node_next_nodes: Map between NNCF Input nodes and the following ONNX nodes.
        :return: Target edge name.
        """
        if transform_type == TargetType.PRE_LAYER_OPERATION:
            return onnx_graph.get_node_edge_names(node_name)['input'][port_id]
        if node_name in nncf_input_node_next_nodes:  # ADD INPUT NODE CASE
            node_names = nncf_input_node_next_nodes[node_name]
            input_edges = set(onnx_graph.get_node_edge_names(name)['input'][port_id] for name in node_names)
            assert len(input_edges) == 1
            return input_edges.pop()
        return onnx_graph.get_node_edge_names(node_name)['output'][port_id]

    def transform(self, transformation_layout: TransformationLayout) -> onnx.ModelProto:
        """
        Applies transformations to the model using an out-of-place approach.
        The transformations do not affect the original model, and a new model
        is returned with the transformations applied. If there are no transformations,
        returns a new instance of the original model.

        :param transformation_layout: Transformation commands.
        :return: The new instance of a model with applied transformations.
        """
        quantizer_insert_transformations = []
        output_insert_transformations = []
        bias_correction_transformations = []
        qdq_node_removing_transformations = []
        model_extraction_transformation = None
        transformations = transformation_layout.transformations
        # No transformation applied
        if not transformations:
            return deepcopy(self._model)
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
        # Inplace transformations, using deepcopy of model
        if quantizer_insert_transformations or bias_correction_transformations or qdq_node_removing_transformations:
            model = deepcopy(self._model)
            if quantizer_insert_transformations:
                model = self._apply_quantizer_insertion_transformations(model, quantizer_insert_transformations)
            if bias_correction_transformations:
                model = self._apply_bias_correction_transformations(model, bias_correction_transformations)
            if qdq_node_removing_transformations:
                model = self._apply_qdq_node_removing_transformations(model, qdq_node_removing_transformations)
        # Transformations that create new model
        if output_insert_transformations:
            model = self._apply_output_insertion_transformations(output_insert_transformations)
        if model_extraction_transformation:
            model = self._apply_model_extraction_transformation(model_extraction_transformation)
        return model

    def _apply_output_insertion_transformations(self,
                                                transformations: List[ONNXOutputInsertionCommand]) -> onnx.ModelProto:
        """
        Returns a new model with extra outputs provided by transformations.

        :param transformations: ONNXOutputInsertionCommand transformations.
        :return: New model with inserted outputs.
        """
        onnx_graph = ONNXGraph(self._model)
        model_outputs = [output.name for output in onnx_graph.get_model_outputs()]
        extra_model_outputs = []
        for transformation in transformations:
            port_id = transformation.target_point.port_id
            node_name = transformation.target_point.target_node_name
            transform_type = transformation.target_point.type
            nncf_input_node_next_onnx_nodes = transformation.nncf_input_node_next_onnx_nodes
            assert transform_type in [TargetType.PRE_LAYER_OPERATION, TargetType.POST_LAYER_OPERATION]
            target_edge_name = self._get_pre_post_target_edge(port_id, node_name, transform_type, onnx_graph,
                                                              nncf_input_node_next_onnx_nodes)
            extra_model_outputs.append(target_edge_name)
        return ONNXModelTransformer._insert_outputs(self._model, outputs=[*extra_model_outputs, *model_outputs])

    @staticmethod
    def _insert_outputs(model: onnx.ModelProto, outputs: List[str]) -> onnx.ModelProto:
        """
        Creates a new model as a copy of provided model with additional outputs.

        :param model: Model of which copy will be created.
        :param outputs: Edge names to use as outputs.
        :return: New model with inserted outputs.
        """
        onnx_graph = ONNXGraph(model)
        model_outputs = []
        for output in outputs:
            # shape should be None; if you place not None, some models will have inference problems (e.g. Mask RCNN)
            type_proto = onnx.helper.make_tensor_type_proto(onnx_graph.get_edge_dtype(output), shape=None)
            model_outputs.append(onnx.helper.make_value_info(name=output, type_proto=type_proto))

        graph = onnx.helper.make_graph(nodes=model.graph.node,
                                       name=model.graph.name,
                                       inputs=model.graph.input,
                                       outputs=model_outputs,
                                       initializer=model.graph.initializer,
                                       value_info=model.graph.value_info)
        new_model = onnx.helper.make_model(graph, ir_version=model.ir_version,
                                           producer_name=model.producer_name,
                                           producer_version=model.producer_version,
                                           domain=model.domain,
                                           model_version=model.model_version,
                                           doc_string=model.doc_string)
        if model.metadata_props:
            values = {p.key: p.value for p in model.metadata_props}
            onnx.helper.set_model_props(new_model, values)
        del new_model.opset_import[:]
        for oimp in model.opset_import:
            op_set = new_model.opset_import.add()
            op_set.domain = oimp.domain
            op_set.version = oimp.version
        return new_model

    def _apply_quantizer_insertion_transformations(self, model: onnx.ModelProto,
                                                   transformations: List[ONNXQuantizerInsertionCommand]) \
            -> onnx.ModelProto:
        """
        Creates a new model as a deepcopy of provided model and inserts QuantizeLinear-DequantizeLinear nodes pair.

        :param model: Model to apply transformations.
        :param transformations: QuantizeLinear-DequantizeLinear nodes pair insertion transformation commands.
        :return: New model with inserted QuantizeLinear-DequantizeLinear nodes pairs.
        """
        self._added_target_edges = Counter()
        for transformation in transformations:
            model = self._insert_quantizer_dequantizer(model, transformation)
        return model

    def _get_quantize_dequantize_nodes(self, transformation: ONNXQuantizerInsertionCommand,
                                       target_edge_name: str) -> Tuple[onnx.NodeProto, onnx.NodeProto]:
        """
        Returns QuantizeLinear-DequantizeLinear nodes pair, based on the transformation parameters and
        inserted onto edge with name target_edge_name.

        :param transformation: QuantizeLinear-DequantizeLinear insertion transformation,
        from which quantization axis is obtained.
        :param target_edge_name: Edge name on which QuantizeLinear-DequantizeLinear nodes pair should be placed.
        :return: QuantizeLinear-DequantizeLinear nodes pair.
        """
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

    @staticmethod
    def _get_scale_zero_point_tensors(transformation: ONNXQuantizerInsertionCommand, quantizer: onnx.NodeProto,
                                      dequantizer: onnx.NodeProto) -> Tuple[onnx.TensorProto, onnx.TensorProto]:
        """
        Returns scale and zero point of QuantizeLinear-DequantizeLinear nodes pair.

        :param transformation: QuantizeLinear-DequantizeLinear insertion transformation,
        from which scale and zero point values are obtained.
        :param quantizer: QuantizeLinear node.
        :param dequantizer: DequantizeLinear node.
        :return: Scale and zero point tensors.
        """
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

    def _get_quantizer_dequantizer_edge_name(self, transformation: ONNXQuantizerInsertionCommand,
                                             onnx_graph: ONNXGraph) -> str:
        """
        Returns an edge name on which QuantizeLinear-DequantizeLinear nodes pair has to be inserted.

        :param transformation: QuantizeLinear-DequantizeLinear insertion transformation.
        :param onnx_graph: ONNXGraph.
        :return: Edge name to insert QuantizeLinear-DequantizeLinear nodes pair.
        """
        port_id = transformation.target_point.port_id
        node_name = transformation.target_point.target_node_name
        transform_type = transformation.target_point.type
        nncf_input_node_next_onnx_nodes = transformation.nncf_input_node_next_onnx_nodes
        if transform_type == TargetType.OPERATION_WITH_WEIGHTS:
            target_edge_name = onnx_graph.get_node_edge_names(node_name)['input'][port_id]
        else:
            target_edge_name = self._get_pre_post_target_edge(port_id, node_name, transform_type, onnx_graph,
                                                              nncf_input_node_next_onnx_nodes)
        self._added_target_edges[target_edge_name] += 1
        return target_edge_name

    def _insert_quantizer_dequantizer(self, model: onnx.ModelProto,
                                      transformation: ONNXQuantizerInsertionCommand) -> onnx.ModelProto:
        """
        Inserts QuantizeLinear-DequantizeLinear nodes pair.

        :param model: Model to insert new nodes.
        :param transformation: QuantizeLinear-DequantizeLinear insertion transformation.
        :return: Updated model with inserted QuantizeLinear-DequantizeLinear pair.
        """
        onnx_graph = ONNXGraph(model)
        target_edge_name = self._get_quantizer_dequantizer_edge_name(transformation, onnx_graph)
        quantizer, dequantizer = self._get_quantize_dequantize_nodes(transformation, target_edge_name)
        onnx_scale_tensor, onnx_zero_point_tensor = ONNXModelTransformer._get_scale_zero_point_tensors(transformation,
                                                                                                       quantizer,
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
        model.graph.initializer.extend([onnx_scale_tensor, onnx_zero_point_tensor])
        model.graph.value_info.extend([onnx_scale_value_info, onnx_zero_point_info])
        insert_index = onnx_graph.get_node_index(input_nodes[0].name)
        model.graph.node.insert(insert_index, quantizer)
        model.graph.node.insert(insert_index + 1, dequantizer)
        return model

    def _apply_bias_correction_transformations(self,
                                               model: onnx.ModelProto,
                                               transformations: List[ONNXBiasCorrectionCommand]) -> onnx.ModelProto:
        """
        Creates a copy of original model and applies bias correction transformations on the model.

        :param model: Model to apply transformations.
        :param transformations: Bias correction transformations.
        :return: Copy of original model with updated biases.
        """
        onnx_graph = ONNXGraph(model)
        for transformation in transformations:
            bias_tensor_position = transformation.target_point.port_id
            node_name = transformation.target_point.target_node_name
            onnx_node = onnx_graph.get_node_by_name(node_name)
            bias_initializer_name = onnx_node.input[bias_tensor_position]
            bias_initializer = onnx_graph.get_initializer(bias_initializer_name)

            new_bias_tensor = onnx.numpy_helper.from_array(transformation.bias_value,
                                                           bias_initializer_name)
            bias_initializer.CopyFrom(new_bias_tensor)
        return model

    def _apply_model_extraction_transformation(self, transformation: ONNXModelExtractionCommand) -> onnx.ModelProto:
        """
        Returns a new model that is a sub-model from the original between provided inputs and outputs.

        :param transformation: Model extraction transformation.
        :return: Extracted sub-model.
        """
        onnx_graph = ONNXGraph(self._model)

        input_tensor_names = []
        for input_node_name in transformation.inputs:
            input_onnx_node = onnx_graph.get_node_by_name(input_node_name)
            input_tensor_names.append(input_onnx_node.input[0])

        output_tensor_names = []
        for output_node_name in transformation.outputs:
            output_onnx_node = onnx_graph.get_node_by_name(output_node_name)
            output_tensor_names.append(output_onnx_node.output[0])

        if not output_tensor_names:
            output_tensor_names = [n.name for n in onnx_graph.get_model_outputs()]

        return self.onnx_model_extractor.extract_model(input_tensor_names, output_tensor_names)

    def _apply_qdq_node_removing_transformations(self,
                                                 model: onnx.ModelProto,
                                                 transformations: List[ONNXQDQNodeRemovingCommand]) -> onnx.ModelProto:
        """
        Returns a copy of original model with removed nodes.

        :param model: Model to apply transformations.
        :param transformations: Nodes removing transformations.
        :return: Model with removed nodes.
        """
        onnx_graph = ONNXGraph(model)
        for transformation in transformations:
            node = onnx_graph.get_node_by_name(transformation.target_point.target_node_name)

            node_children = onnx_graph.get_children(node)
            for node_child in node_children:
                for input_id, input_obj in enumerate(node_child.input):
                    if input_obj == node.output[0]:
                        node_child.input[input_id] = node.input[0]

            initializers = {i.name: i for i in model.graph.initializer}
            value_infos = {i.name: i for i in model.graph.value_info}
            for initializer_name in node.input:
                if initializer_name in initializers:
                    model.graph.initializer.remove(initializers[initializer_name])
                if initializer_name in value_infos:
                    model.graph.value_info.remove(value_infos[initializer_name])

            model.graph.node.remove(node)
        return model
