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

from collections import defaultdict

from onnx import ModelProto  # pylint: disable=no-name-in-module

from nncf.common.graph import NNCFGraph
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.definitions import MODEL_INPUT_OP_NAME
from nncf.common.graph.definitions import MODEL_OUTPUT_OP_NAME
from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from nncf.common.utils.logger import logger as nncf_logger

from nncf.experimental.onnx.graph.onnx_graph import ONNXGraph
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNX_OPERATION_METATYPES
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXConstantMetatype


class GraphConverter:
    """
    Builds the NNCFGraph from an ONNX model
    """

    DEFAULT_TENSOR_SHAPE = [1]

    # pylint: disable=too-many-statements
    @staticmethod
    def create_nncf_graph(onnx_model: ModelProto) -> NNCFGraph:
        """
        Adds all ONNX nodes from 'onnx_model' and then adds thr special input_nodes and output_nodes.
        """

        def add_nncf_input_node(onnx_graph):
            for i, _input in enumerate(onnx_graph.get_model_inputs()):
                try:
                    input_shape = onnx_graph.get_tensor_shape(_input)
                except RuntimeError as err:
                    nncf_logger.error(err)
                    nncf_logger.error('The default tensor shape will be set.')
                    input_shape = GraphConverter.DEFAULT_TENSOR_SHAPE
                input_node = nncf_graph.add_nncf_node(node_name=MODEL_INPUT_OP_NAME + '_' + str(i),
                                                      node_type=NNCFGraphNodeType.INPUT_NODE,
                                                      node_metatype=InputNoopMetatype,
                                                      layer_attributes=None)
                input_name = _input.name
                to_nodes = onnx_graph.get_nodes_by_input(input_name)
                for node in to_nodes:
                    in_node_id = input_node.node_id
                    to_node_id = nncf_graph.get_node_by_name(node.name).node_id
                    input_counter[in_node_id] += 1
                    output_counter[to_node_id] += 1
                    onnx_dtype = onnx_graph.get_edge_dtype_name(input_name)
                    nncf_dtype = GraphConverter.convert_onnx_dtype_to_nncf_dtype(onnx_dtype)
                    nncf_graph.add_edge_between_nncf_nodes(
                        from_node_id=input_node.node_id,
                        to_node_id=to_node_id,
                        tensor_shape=input_shape,
                        input_port_id=input_counter[in_node_id],
                        output_port_id=output_counter[to_node_id],
                        dtype=nncf_dtype
                    )

        def add_nncf_output_nodes(onnx_graph):
            for i, _output in enumerate(onnx_graph.get_model_outputs()):
                try:
                    output_shape = onnx_graph.get_tensor_shape(_output)
                except RuntimeError as err:
                    nncf_logger.error(err)
                    nncf_logger.error('The default tensor shape will be set.')
                    output_shape = GraphConverter.DEFAULT_TENSOR_SHAPE
                output_node = nncf_graph.add_nncf_node(node_name=MODEL_OUTPUT_OP_NAME + '_' + str(i),
                                                       node_type=NNCFGraphNodeType.OUTPUT_NODE,
                                                       node_metatype=OutputNoopMetatype,
                                                       layer_attributes=None)

                output_name = _output.name
                to_nodes = onnx_graph.get_nodes_by_output(output_name)
                for node in to_nodes:
                    out_node_id = output_node.node_id
                    to_node_id = nncf_graph.get_node_by_name(node.name).node_id
                    input_counter[out_node_id] += 1
                    output_counter[to_node_id] += 1
                    onnx_dtype = onnx_graph.get_edge_dtype_name(output_name)
                    nncf_dtype = GraphConverter.convert_onnx_dtype_to_nncf_dtype(onnx_dtype)
                    nncf_graph.add_edge_between_nncf_nodes(
                        from_node_id=to_node_id,
                        to_node_id=output_node.node_id,
                        tensor_shape=output_shape,
                        input_port_id=input_counter[out_node_id],
                        output_port_id=output_counter[to_node_id],
                        dtype=nncf_dtype
                    )

        nncf_graph = NNCFGraph()
        onnx_graph = ONNXGraph(onnx_model)
        for _, node in enumerate(onnx_graph.get_all_nodes()):
            node_name = node.name
            node_type = node.op_type
            metatype = ONNX_OPERATION_METATYPES.get_operator_metatype_by_op_name(node_type)
            if metatype == ONNXConstantMetatype:  # We don't need to quantize Constants
                continue
            if metatype == UnknownMetatype:
                nncf_logger.warning(
                    'The node with name {} with type {} was mapped to UnknownMetatype,'
                    ' which means that there was not registered such NNCF metatype'.format(
                        node_name, node_type))
            nncf_graph.add_nncf_node(node_name=node_name,
                                     node_type=node_type,
                                     node_metatype=metatype,
                                     layer_attributes=None)
        input_counter = defaultdict(int)
        output_counter = defaultdict(int)
        for output_node in nncf_graph.get_all_nodes():
            output_node_id = output_node.node_id
            outputs = onnx_graph.get_node_edges(output_node.node_name)['output']
            for output in outputs:
                nodes = onnx_graph.get_nodes_by_input(output)
                if len(nodes) == 0:  # if this node is output
                    continue
                try:
                    shape = onnx_graph.get_edge_shape(output)
                # This exception raised because ONNX format allows to not have shape field.
                # Model example - effecienet-v2, mobilenet_v2.
                # In fact, the quantization algorithm doesn't utilize tensor shape information.
                # So, if there is no shape, the DEFAULT_TENSOR_SHAPE is used.
                except RuntimeError as err:
                    nncf_logger.error(err)
                    nncf_logger.error('The default tensor shape will be set.')
                    shape = GraphConverter.DEFAULT_TENSOR_SHAPE
                onnx_dtype = onnx_graph.get_edge_dtype_name(output)
                nncf_dtype = GraphConverter.convert_onnx_dtype_to_nncf_dtype(onnx_dtype)
                for in_node in nodes:
                    in_node_id = nncf_graph.get_node_by_name(in_node.name).node_id
                    input_counter[in_node_id] += 1
                    output_counter[output_node_id] += 1
                    nncf_graph.add_edge_between_nncf_nodes(
                        from_node_id=output_node_id,
                        to_node_id=in_node_id,
                        tensor_shape=shape,
                        input_port_id=input_counter[in_node_id],
                        output_port_id=output_counter[output_node_id],
                        dtype=Dtype(nncf_dtype)
                    )
        add_nncf_input_node(onnx_graph)
        add_nncf_output_nodes(onnx_graph)
        return nncf_graph

    @staticmethod
    def convert_onnx_dtype_to_nncf_dtype(onnx_dtype: str) -> Dtype:
        conversation_map = {
            "FLOAT": "float",
            "FLOAT16": "float",
            "BFLOAT16": "float",
            "DOUBLE": "float",
        }
        return Dtype(conversation_map.get(onnx_dtype, 'int'))
