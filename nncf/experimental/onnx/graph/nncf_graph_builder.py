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
from typing import Union, List

import onnx
from onnx import ModelProto
from onnx import NodeProto  # pylint: disable=no-name-in-module

from nncf.common.graph import NNCFGraph
from nncf.common.graph import layer_attributes
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.graph.layer_attributes import BaseLayerAttributes, Dtype
from nncf.common.graph.definitions import MODEL_INPUT_OP_NAME
from nncf.common.graph.definitions import MODEL_OUTPUT_OP_NAME
from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from nncf.common.utils.logger import logger as nncf_logger

from nncf.experimental.onnx.graph.onnx_graph import ONNXGraph
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import LAYERS_WITH_BIAS_METATYPES, ONNX_OPERATION_METATYPES
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXConstantMetatype


class GraphConverter:
    """
    Builds the NNCFGraph from an ONNX model.
    """

    DEFAULT_TENSOR_SHAPE = [1]

    @staticmethod
    def _is_valid_onnx_metatype(node: NodeProto) -> bool:
        """
        Checks whether the node has the metatype which should be added to the NNCFGraph.
        :param node: Node to be checked.
        :return: True if the metatype is valid and False if not.
        """
        node_type = node.op_type
        metatype = ONNX_OPERATION_METATYPES.get_operator_metatype_by_op_name(node_type)
        if metatype == ONNXConstantMetatype:  # We don't need to quantize Constants
            nncf_logger.debug('The metatype is ONNXConstantMetatype, which in not quantizable. Skipping this node.')
            return False
        if metatype == UnknownMetatype:
            node_name = node.name
            nncf_logger.warning(
                'The node with name {} with type {} was mapped to UnknownMetatype,'
                ' which means that there was not registered such NNCF metatype. '
                'Please, Inform the NNCF developers about this message.'.format(
                    node_name, node_type))
            return True
        return True

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
                tensor_shape = onnx_graph.get_tensor_shape(tensor)
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
            layer_attributes = ExtendedLayerAttributes([input_name], [input_name])
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
            for node in filter(GraphConverter._is_valid_onnx_metatype, to_nodes):
                to_node_id = nncf_graph.get_node_by_name(node.name).node_id
                input_port_id = onnx_graph.get_input_port_id_for_node_after_input(input_name, node)
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
            layer_attributes = ExtendedLayerAttributes([output_name], [output_name])
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
            for node in filter(GraphConverter._is_valid_onnx_metatype, from_nodes):
                from_node_id = nncf_graph.get_node_by_name(node.name).node_id
                output_port_id = onnx_graph.get_output_port_id_for_node_before_output(output_name, node)
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
        nncf_graph = NNCFGraph()
        onnx_graph = ONNXGraph(onnx_model)
        for node in filter(GraphConverter._is_valid_onnx_metatype, onnx_graph.get_all_nodes()):
            metatype = ONNX_OPERATION_METATYPES.get_operator_metatype_by_op_name(node.op_type)
            layer_attributes = ExtendedLayerAttributes(node.input, node.output)
            nncf_graph.add_nncf_node(node_name=node.name,
                                     node_type=node.op_type,
                                     node_metatype=metatype,
                                     layer_attributes=layer_attributes)
        for output_node in filter(GraphConverter._is_valid_onnx_metatype, onnx_graph.get_all_nodes()):
            output_edges = onnx_graph.get_node_edges(output_node.name)['output']
            for output_edge in output_edges:
                tensor_shape = GraphConverter._get_tensor_shape(onnx_graph, output_edge)

                output_node_id = nncf_graph.get_node_by_name(output_node.name).node_id
                onnx_dtype = onnx_graph.get_edge_dtype_name(output_edge)
                nncf_dtype = GraphConverter.convert_onnx_dtype_to_nncf_dtype(onnx_dtype)

                input_nodes = onnx_graph.get_nodes_by_input(output_edge)
                if not input_nodes:
                    # if this node is output
                    continue
                for input_node in filter(GraphConverter._is_valid_onnx_metatype, input_nodes):
                    port_ids = onnx_graph.get_port_ids_between_nodes(output_node, input_node)
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

class ExtendedLayerAttributes(BaseLayerAttributes):
    def __init__(self, input_tensor_names, output_tensor_names):
        self.input_tensor_names = input_tensor_names
        self.output_tensor_names = output_tensor_names