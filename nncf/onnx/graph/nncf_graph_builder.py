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
from typing import Union, List

from collections import Counter
import onnx
from onnx import ModelProto

from nncf.common.graph import NNCFGraph
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.layer_attributes import BaseLayerAttributes, Dtype
from nncf.common.graph.definitions import MODEL_INPUT_OP_NAME
from nncf.common.graph.definitions import MODEL_OUTPUT_OP_NAME
from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.logging import nncf_logger

from nncf.onnx.graph.onnx_graph import ONNXGraph
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNX_OPERATION_METATYPES
from nncf.onnx.graph.metatypes.onnx_metatypes import WEIGHT_LAYER_METATYPES


class GraphConverter:
    """
    Builds the NNCFGraph from an ONNX model.
    """

    DEFAULT_TENSOR_SHAPE = [1]

    @staticmethod
    def _replace_empty_node_name(model: onnx.ModelProto) -> onnx.ModelProto:
        """
        Sets a unique name to every node in 'model' with empty name field.
        NNCFGraph expects every node to have a unique name.

        :param model: ONNX model.
        :return: ONNX model with filled nodes.
        """
        for i, node in enumerate(model.graph.node):
            if node.name == '':
                node.name = node.op_type + '_nncf_' + str(i)

        name_counter = Counter([node.name for node in model.graph.node])

        if max(name_counter.values()) > 1:
            raise RuntimeError(
                f"Nodes {[(name, cnt) for name, cnt in name_counter.items() if cnt > 1]} "
                "(name, counts) occurred more than once. "
                "NNCF expects every node to have a unique name.")

        return model

    @staticmethod
    def _get_tensor_shape(onnx_graph: onnx.GraphProto, tensor: Union[str, onnx.ValueInfoProto]) -> List[int]:
        """
        Returns the shape of the 'tensor'.
        :param onnx_graph: Graph, in which 'tensor' is been seeking.
        :param tensor: Could be a name of tensor or ONNX internal tensor type.
        :return: the 'tensor' shape.
        """
        try:
            if isinstance(tensor, str):
                tensor_shape = onnx_graph.get_edge_shape(tensor)
            elif isinstance(tensor, onnx.ValueInfoProto):
                tensor_shape = ONNXGraph.get_tensor_shape(tensor)
        except RuntimeError as err:
            # This exception raised because ONNX format allows to not have shape field.
            # Model example - effecienet-v2, mobilenet_v2.
            # In fact, the quantization algorithm doesn't utilize tensor shape information.
            # So, if there is no shape, the DEFAULT_TENSOR_SHAPE is used.
            nncf_logger.debug(err)
            nncf_logger.debug('The default tensor shape will be set.')
            tensor_shape = GraphConverter.DEFAULT_TENSOR_SHAPE
        return tensor_shape

    @staticmethod
    def _add_nncf_input_nodes(onnx_graph: onnx.GraphProto, nncf_graph: NNCFGraph) -> None:
        """
        Adds special NNCF Input nodes to NNCFGraph.
        For all the ONNX model inputs, the special NNCF Input node is placed and then corresponding edges are added.
        :param onnx_graph: ONNXGraph, which helps to get information about the ONNX model.
        :param nncf_graph: NNCFGraph, in which the new nodes will be added.
        :return: None.
        """
        for i, _input in enumerate(onnx_graph.get_model_inputs()):
            input_name = _input.name
            layer_attributes = ONNXExtendedLayerAttributes([input_name], [input_name])
            input_node = nncf_graph.add_nncf_node(node_name=MODEL_INPUT_OP_NAME + '_' + str(i),
                                                  node_type=NNCFGraphNodeType.INPUT_NODE,
                                                  node_metatype=InputNoopMetatype,
                                                  layer_attributes=layer_attributes)
            to_nodes = onnx_graph.get_nodes_by_input(input_name)

            input_node_node_id = input_node.node_id
            input_shape = GraphConverter._get_tensor_shape(onnx_graph, input_name)
            onnx_dtype = onnx_graph.get_edge_dtype_name(input_name)
            nncf_dtype = GraphConverter.convert_onnx_dtype_to_nncf_dtype(onnx_dtype)
            output_port_id = 0
            for node in to_nodes:
                to_node_id = nncf_graph.get_node_by_name(node.name).node_id
                input_port_id = ONNXGraph.get_input_port_id_for_node_after_input(input_name, node)
                nncf_graph.add_edge_between_nncf_nodes(
                    from_node_id=input_node_node_id,
                    to_node_id=to_node_id,
                    tensor_shape=input_shape,
                    input_port_id=input_port_id,
                    output_port_id=output_port_id,
                    dtype=nncf_dtype
                )
                output_port_id += 1

    @staticmethod
    def _add_nncf_output_nodes(onnx_graph: onnx.GraphProto, nncf_graph: NNCFGraph) -> None:
        """
        Adds special NNCF Output nodes to NNCFGraph.
        For all the ONNX model outputs, the special NNCF Output node is placed and then corresponding edges are added.
        :param onnx_graph: ONNXGraph, which helps to get information about the ONNX model.
        :param nncf_graph: NNCFGraph, in which the new nodes will be added.
        :return: None.
        """
        for i, _output in enumerate(onnx_graph.get_model_outputs()):
            output_name = _output.name
            layer_attributes = ONNXExtendedLayerAttributes([output_name], [output_name])
            output_node = nncf_graph.add_nncf_node(node_name=MODEL_OUTPUT_OP_NAME + '_' + str(i),
                                                   node_type=NNCFGraphNodeType.OUTPUT_NODE,
                                                   node_metatype=OutputNoopMetatype,
                                                   layer_attributes=layer_attributes)
            from_nodes = onnx_graph.get_nodes_by_output(output_name)

            output_node_node_id = output_node.node_id
            output_shape = GraphConverter._get_tensor_shape(onnx_graph, output_name)
            onnx_dtype = onnx_graph.get_edge_dtype_name(output_name)
            nncf_dtype = GraphConverter.convert_onnx_dtype_to_nncf_dtype(onnx_dtype)
            input_port_id = 0
            for node in from_nodes:
                from_node_id = nncf_graph.get_node_by_name(node.name).node_id
                output_port_id = ONNXGraph.get_output_port_id_for_node_before_output(output_name, node)
                nncf_graph.add_edge_between_nncf_nodes(
                    from_node_id=from_node_id,
                    to_node_id=output_node_node_id,
                    tensor_shape=output_shape,
                    input_port_id=input_port_id,
                    output_port_id=output_port_id,
                    dtype=nncf_dtype
                )
                input_port_id += 1

    @staticmethod
    def convert_onnx_dtype_to_nncf_dtype(onnx_dtype: str) -> Dtype:
        """
        Converts the primitive types from the ONNX domain to the NNCF domain.
        :param onnx_dtype: ONNX primitive typename.
        :return: NNCF primitive type.
        """
        conversion_map = {
            "FLOAT": "float",
            "FLOAT16": "float",
            "BFLOAT16": "float",
            "DOUBLE": "float",
        }
        return Dtype(conversion_map.get(onnx_dtype, 'int'))

    @staticmethod
    def create_nncf_graph(onnx_model: ModelProto) -> NNCFGraph:
        """
        Creates NNCFGraph from 'onnx_model'.
        Initially, ONNXGraph is built. All nodes from onnx_model which have valid metatype are added to NNCFGraph.
        Then, corresponding edges are added to the NNCFGraph with shape, type, output and input port ids.
        In the last step, special NNCF Input and Output nodes are added.
        :param onnx_model: ONNX model.
        :return: NNCFGraph.
        """
        onnx_model = GraphConverter._replace_empty_node_name(onnx_model)
        nncf_graph = NNCFGraph()
        onnx_graph = ONNXGraph(onnx_model)
        for node in onnx_graph.get_all_nodes():
            metatype = ONNX_OPERATION_METATYPES.get_operator_metatype_by_op_name(node.op_type)
            if metatype is not UnknownMetatype:
                if metatype.get_subtypes():
                    subtype = metatype.determine_subtype(onnx_model, node)
                    if subtype is not None:
                        metatype = subtype
            layer_attributes = ONNXExtendedLayerAttributes(node.input, node.output)
            is_shared, layer_name = None, None
            if metatype in WEIGHT_LAYER_METATYPES:
                is_shared = onnx_graph.is_node_shared(node)
                layer_name = onnx_graph.get_node_layer_name(node)
            nncf_graph.add_nncf_node(node_name=node.name,
                                     node_type=node.op_type,
                                     node_metatype=metatype,
                                     layer_attributes=layer_attributes,
                                     layer_name=layer_name,
                                     is_shared=is_shared)
        for output_node in onnx_graph.get_all_nodes():
            output_edges = onnx_graph.get_node_edge_names(output_node.name)['output']
            for output_edge in output_edges:
                tensor_shape = GraphConverter._get_tensor_shape(onnx_graph, output_edge)

                output_node_id = nncf_graph.get_node_by_name(output_node.name).node_id
                try:
                    onnx_dtype = onnx_graph.get_edge_dtype_name(output_edge)
                except RuntimeError:
                    # If the edge was not added during inference of ONNX model,
                    # we do not add it to NNCFGraph.
                    # Particularly, BatchNorm exported in Training mode has unused outputs edges:
                    # mean, var, saved_mean, saved_var.
                    continue
                nncf_dtype = GraphConverter.convert_onnx_dtype_to_nncf_dtype(onnx_dtype)

                input_nodes = onnx_graph.get_nodes_by_input(output_edge)
                if not input_nodes:
                    # if this node is output
                    continue
                for input_node in input_nodes:
                    port_ids = ONNXGraph.get_port_ids_between_nodes(output_node, input_node)
                    input_port_id = port_ids['input_port_id']
                    output_port_id = port_ids['output_port_id']
                    in_node_id = nncf_graph.get_node_by_name(input_node.name).node_id
                    nncf_graph.add_edge_between_nncf_nodes(
                        from_node_id=output_node_id,
                        to_node_id=in_node_id,
                        tensor_shape=tensor_shape,
                        input_port_id=input_port_id,
                        output_port_id=output_port_id,
                        dtype=Dtype(nncf_dtype)
                    )
        GraphConverter._add_nncf_input_nodes(onnx_graph, nncf_graph)
        GraphConverter._add_nncf_output_nodes(onnx_graph, nncf_graph)
        return nncf_graph


class ONNXExtendedLayerAttributes(BaseLayerAttributes):
    """
    This class stores extended attributes of modules/layers for the algorithms.
    """

    def __init__(self, input_tensor_names, output_tensor_names):
        """
        :param input_tensor_names: List of the input tensor/edge names of the module/layer
        :param output_tensor_names: List of the output tensor/edge names of the module/layer
        """
        self.input_tensor_names = input_tensor_names
        self.output_tensor_names = output_tensor_names
