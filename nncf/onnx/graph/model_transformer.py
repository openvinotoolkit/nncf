# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import onnx

from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.onnx.graph.node_utils import get_input_edge
from nncf.onnx.graph.onnx_helper import get_children
from nncf.onnx.graph.onnx_helper import get_children_node_mapping
from nncf.onnx.graph.onnx_helper import get_edge_dtype
from nncf.onnx.graph.onnx_helper import get_edge_info_mapping
from nncf.onnx.graph.onnx_helper import get_name_to_node_map
from nncf.onnx.graph.onnx_helper import get_node_index
from nncf.onnx.graph.onnx_helper import get_tensor
from nncf.onnx.graph.transformations.commands import ONNXBiasCorrectionCommand
from nncf.onnx.graph.transformations.commands import ONNXModelExtractionCommand
from nncf.onnx.graph.transformations.commands import ONNXOutputInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXQDQNodeRemovingCommand
from nncf.onnx.graph.transformations.commands import ONNXQuantizerInsertionCommand


class ONNXModelTransformer(ModelTransformer):
    """
    Applies transformations upon ONNX model.
    ModelTransformer should be created once for a particular model,
    and be used to apply transformations to the provided model.
    """

    QUANTIZER_NAME_PREFIX = "QuantizeLinear_"
    DEQUANTIZER_NAME_PREFIX = "DequantizeLinear_"
    SCALE_TENSOR_NAME_PREFIX = "scale_"
    ZERO_POINT_NAME_PREFIX = "zero_point_"

    def __init__(self, model: onnx.ModelProto):
        infered_model = onnx.shape_inference.infer_shapes(model)
        super().__init__(infered_model)
        self.onnx_model_extractor = onnx.utils.Extractor(self._model)

    def _get_target_edge(
        self,
        port_id: int,
        node_name: str,
        transform_type: TargetType,
        node_mapping: Dict[str, onnx.NodeProto],
        input_edges_mapping: Dict[str, str],
    ) -> str:
        """
        Returns edge name corresponding to the node with a name equal to node_name, port_id and transform_type.

        :param port_id: Edge number of port.
        :param node_name: Node name.
        :param transform_type: Type of transformation.
        :param node_mapping: Mapping from a node name to the node.
        :param input_edges_mapping: Mapping between NNCF Input nodes and
        the following ONNX nodes and corresponding input port id.
        :return: Target edge name.
        """
        if transform_type in [TargetType.PRE_LAYER_OPERATION, TargetType.OPERATION_WITH_WEIGHTS]:
            return node_mapping[node_name].input[port_id]
        if node_name in input_edges_mapping:  # ADD INPUT NODE CASE
            return get_input_edge(node_name, input_edges_mapping, node_mapping)
        return node_mapping[node_name].output[port_id]

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

    def _apply_output_insertion_transformations(
        self, transformations: List[ONNXOutputInsertionCommand]
    ) -> onnx.ModelProto:
        """
        Returns a new model with extra outputs provided by transformations.

        :param transformations: ONNXOutputInsertionCommand transformations.
        :return: New model with inserted outputs.
        """
        model_outputs = set(output.name for output in self._model.graph.output)
        node_mapping = get_name_to_node_map(self._model)
        for transformation in transformations:
            port_id = transformation.target_point.port_id
            node_name = transformation.target_point.target_node_name
            transform_type = transformation.target_point.type
            input_edges_mapping = transformation.input_edges_mapping
            target_edge_name = self._get_target_edge(
                port_id, node_name, transform_type, node_mapping, input_edges_mapping
            )
            model_outputs.add(target_edge_name)

        return ONNXModelTransformer._insert_outputs(self._model, outputs=model_outputs)

    @staticmethod
    def _insert_outputs(model: onnx.ModelProto, outputs: Union[List[str], Set[str]]) -> onnx.ModelProto:
        """
        Creates a new model as a copy of provided model with additional outputs.

        :param model: Model of which copy will be created.
        :param outputs: Edge names to use as outputs.
        :return: New model with inserted outputs.
        """
        model_outputs = []
        edge_info_mapping = get_edge_info_mapping(model)
        for output in outputs:
            edge = edge_info_mapping[output]
            onnx_dtype = get_edge_dtype(edge)
            type_proto = onnx.helper.make_tensor_type_proto(onnx_dtype, shape=None)
            model_outputs.append(onnx.helper.make_value_info(name=output, type_proto=type_proto))

        graph = onnx.helper.make_graph(
            nodes=model.graph.node,
            name=model.graph.name,
            inputs=model.graph.input,
            outputs=model_outputs,
            initializer=model.graph.initializer,
            value_info=model.graph.value_info,
        )
        new_model = onnx.helper.make_model(
            graph,
            ir_version=model.ir_version,
            producer_name=model.producer_name,
            producer_version=model.producer_version,
            domain=model.domain,
            model_version=model.model_version,
            doc_string=model.doc_string,
        )
        if model.metadata_props:
            values = {p.key: p.value for p in model.metadata_props}
            onnx.helper.set_model_props(new_model, values)
        del new_model.opset_import[:]
        for oimp in model.opset_import:
            op_set = new_model.opset_import.add()
            op_set.domain = oimp.domain
            op_set.version = oimp.version
        return new_model

    def _apply_quantizer_insertion_transformations(
        self, model: onnx.ModelProto, transformations: List[ONNXQuantizerInsertionCommand]
    ) -> onnx.ModelProto:
        """
        Creates a new model as a deepcopy of provided model and inserts QuantizeLinear-DequantizeLinear nodes pair.

        :param model: Model to apply transformations.
        :param transformations: QuantizeLinear-DequantizeLinear nodes pair insertion transformation commands.
        :return: New model with inserted QuantizeLinear-DequantizeLinear nodes pairs.
        """
        self._added_target_edges = Counter()
        node_mapping = get_name_to_node_map(model)
        children_node_mapping = get_children_node_mapping(model)
        for transformation in transformations:
            model = self._insert_quantizer_dequantizer(model, transformation, node_mapping, children_node_mapping)
        return model

    def _get_quantize_dequantize_nodes(
        self, transformation: ONNXQuantizerInsertionCommand, target_edge_name: str
    ) -> Tuple[onnx.NodeProto, onnx.NodeProto]:
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
        q_target_edge_name = target_edge_name + "_" + str(cnt)
        quantizer_name = ONNXModelTransformer.QUANTIZER_NAME_PREFIX + q_target_edge_name
        dequantizer_name = ONNXModelTransformer.DEQUANTIZER_NAME_PREFIX + q_target_edge_name
        scale_tensor_name = ONNXModelTransformer.SCALE_TENSOR_NAME_PREFIX + q_target_edge_name
        zero_point_tensor_name = ONNXModelTransformer.ZERO_POINT_NAME_PREFIX + q_target_edge_name

        quantizer = onnx.helper.make_node(
            name=quantizer_name,
            op_type="QuantizeLinear",
            inputs=[input_target_edge, scale_tensor_name, zero_point_tensor_name],
            outputs=["q_output_" + q_target_edge_name],
            axis=axis,
        )

        dequantizer = onnx.helper.make_node(
            name=dequantizer_name,
            op_type="DequantizeLinear",
            inputs=["q_output_" + q_target_edge_name, scale_tensor_name, zero_point_tensor_name],
            outputs=["dq_output_" + q_target_edge_name],
            axis=axis,
        )

        return quantizer, dequantizer

    @staticmethod
    def _get_scale_zero_point_tensors(
        transformation: ONNXQuantizerInsertionCommand, quantizer: onnx.NodeProto, dequantizer: onnx.NodeProto
    ) -> Tuple[onnx.TensorProto, onnx.TensorProto]:
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
        onnx_scale = [scale.tolist()] if not per_channel else scale
        onnx_zero_point = [zero_point.tolist()] if not per_channel else zero_point
        if tensor_type == np.uint8:
            onnx_tensor_type = onnx.TensorProto.UINT8
        elif tensor_type == np.int8:
            onnx_tensor_type = onnx.TensorProto.INT8
        else:
            raise RuntimeError(f"Incorrect tensor type - {tensor_type}.")
        assert quantizer.input[1] == dequantizer.input[1] and quantizer.input[2] == dequantizer.input[2]
        scale_tensor_name = quantizer.input[1]
        zero_point_tensor_name = quantizer.input[2]
        onnx_scale_tensor = onnx.helper.make_tensor(scale_tensor_name, onnx.TensorProto.FLOAT, dims, onnx_scale)
        onnx_zero_point_tensor = onnx.helper.make_tensor(
            zero_point_tensor_name, onnx_tensor_type, dims, onnx_zero_point
        )
        return onnx_scale_tensor, onnx_zero_point_tensor

    def _get_quantizer_dequantizer_edge_name(
        self, transformation: ONNXQuantizerInsertionCommand, node_mapping: Dict[str, onnx.NodeProto]
    ) -> str:
        """
        Returns an edge name on which QuantizeLinear-DequantizeLinear nodes pair has to be inserted.

        :param transformation: QuantizeLinear-DequantizeLinear insertion transformation.
        :param node_mapping: Mapping from a node name to the node.
        :return: Edge name to insert QuantizeLinear-DequantizeLinear nodes pair.
        """
        port_id = transformation.target_point.port_id
        node_name = transformation.target_point.target_node_name
        transform_type = transformation.target_point.type
        input_edges_mapping = transformation.input_edges_mapping
        target_edge_name = self._get_target_edge(port_id, node_name, transform_type, node_mapping, input_edges_mapping)
        self._added_target_edges[target_edge_name] += 1
        return target_edge_name

    def _insert_quantizer_dequantizer(
        self,
        model: onnx.ModelProto,
        transformation: ONNXQuantizerInsertionCommand,
        node_mapping: Dict[str, onnx.NodeProto],
        children_node_mapping: Dict[str, List[onnx.ValueInfoProto]],
    ) -> onnx.ModelProto:
        """
        Inserts QuantizeLinear-DequantizeLinear nodes pair.

        :param model: Model to insert new nodes.
        :param transformation: QuantizeLinear-DequantizeLinear insertion transformation.
        :param node_mapping: Mapping from node name to the node.
        :param children_node_mapping: Mapping from edge name to nodes which consume this edge as an input.
        :return: Updated model with inserted QuantizeLinear-DequantizeLinear pair.
        """
        target_edge_name = self._get_quantizer_dequantizer_edge_name(transformation, node_mapping)
        quantizer, dequantizer = self._get_quantize_dequantize_nodes(transformation, target_edge_name)
        onnx_scale_tensor, onnx_zero_point_tensor = ONNXModelTransformer._get_scale_zero_point_tensors(
            transformation, quantizer, dequantizer
        )

        # If several nodes on one edge
        input_nodes = []
        input_nodes.extend(children_node_mapping[target_edge_name])
        if not input_nodes:
            raise RuntimeError(
                f"Can not add the quantizer to the {target_edge_name} edge. This edge does not have end node."
            )

        if transformation.target_point.type == TargetType.PRE_LAYER_OPERATION:
            # If we need to change only target nodes input
            target_node = node_mapping[transformation.target_point.target_node_name]
            for i, inp in enumerate(target_node.input):
                if inp == target_edge_name:
                    target_node.input[i] = dequantizer.output[0]
        else:
            for node in input_nodes:
                for i, inp in enumerate(node.input):
                    if inp == target_edge_name:
                        node.input[i] = dequantizer.output[0]

        onnx_scale_value_info = onnx.helper.make_tensor_value_info(
            onnx_scale_tensor.name, onnx_scale_tensor.data_type, onnx_scale_tensor.dims
        )
        onnx_zero_point_info = onnx.helper.make_tensor_value_info(
            onnx_zero_point_tensor.name, onnx_zero_point_tensor.data_type, onnx_zero_point_tensor.dims
        )
        model.graph.initializer.extend([onnx_scale_tensor, onnx_zero_point_tensor])
        model.graph.value_info.extend([onnx_scale_value_info, onnx_zero_point_info])
        insert_index = get_node_index(model, input_nodes[0].name)
        model.graph.node.insert(insert_index, quantizer)
        model.graph.node.insert(insert_index + 1, dequantizer)
        return model

    def _apply_bias_correction_transformations(
        self, model: onnx.ModelProto, transformations: List[ONNXBiasCorrectionCommand]
    ) -> onnx.ModelProto:
        """
        Creates a copy of original model and applies bias correction transformations on the model.

        :param model: Model to apply transformations.
        :param transformations: Bias correction transformations.
        :return: Copy of original model with updated biases.
        """
        node_mapping = get_name_to_node_map(model)
        for transformation in transformations:
            bias_tensor_position = transformation.target_point.port_id
            node_name = transformation.target_point.target_node_name
            onnx_node = node_mapping[node_name]
            bias_initializer_name = onnx_node.input[bias_tensor_position]
            bias_initializer = get_tensor(model, bias_initializer_name)

            new_bias_tensor = onnx.numpy_helper.from_array(transformation.bias_value, bias_initializer_name)
            bias_initializer.CopyFrom(new_bias_tensor)
        return model

    def _apply_model_extraction_transformation(self, transformation: ONNXModelExtractionCommand) -> onnx.ModelProto:
        """
        Returns a new model that is a sub-model from the original between provided inputs and outputs.

        :param transformation: Model extraction transformation.
        :return: Extracted sub-model.
        """
        input_tensor_names = []
        node_mapping = get_name_to_node_map(self._model)
        for input_node_name in transformation.inputs:
            input_onnx_node = node_mapping[input_node_name]
            input_tensor_names.append(input_onnx_node.input[0])

        output_tensor_names = []
        for output_node_name in transformation.outputs:
            output_onnx_node = node_mapping[output_node_name]
            output_tensor_names.append(output_onnx_node.output[0])

        if not output_tensor_names:
            output_tensor_names = [n.name for n in self._model.graph.output]

        return self.onnx_model_extractor.extract_model(input_tensor_names, output_tensor_names)

    def _apply_qdq_node_removing_transformations(
        self, model: onnx.ModelProto, transformations: List[ONNXQDQNodeRemovingCommand]
    ) -> onnx.ModelProto:
        """
        Returns a copy of original model with removed nodes.

        :param model: Model to apply transformations.
        :param transformations: Nodes removing transformations.
        :return: Model with removed nodes.
        """
        for transformation in transformations:
            node_mapping = get_name_to_node_map(model)
            children_node_mapping = get_children_node_mapping(model)
            node = node_mapping[transformation.target_point.target_node_name]

            node_children = get_children(node, children_node_mapping)
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
