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

from typing import Optional, Dict, Any, List, Tuple
from collections import deque

import tensorflow as tf

from nncf.common.graph import NNCFGraph
from nncf.common.graph.layer_attributes import Dtype
from nncf.tensorflow.graph.metatypes.matcher import get_op_metatype
from nncf.tensorflow.graph.metatypes.common import ALL_LAYER_METATYPES_WITH_WEIGHTS
from nncf.experimental.tensorflow.nncf_network import NNCFNetwork
from nncf.experimental.tensorflow.graph.node_attributes import TFNodeAttributes
from nncf.experimental.tensorflow.graph.node_attributes import TFWeightedNodeAttributes


class NodeDesc:
    """
    Contains description of a node in the TensorFlow `FuncGraph`.
    This information is required for `NNCFNode` creation.
    """

    def __init__(self,
                 op_name: str,
                 op_type_name: str,
                 metatype,
                 is_shared: bool,
                 resource_name: Optional[str] = None,
                 attrs: Optional[Any] = None):
        """
        Initializes description of a node.

        :param op_name: Name of a node in the `FuncGraph`.
        :param op_type_name: Type of operation.
        :param metatype: NNCF meta type which corresponds to operation.
        :param is_shared: `True` if the node is shared, `False` otherwise.
        :param resource_name: Name of a node that contains weights of operation.
        :param attrs: Attributes of a node.
        """
        self.op_name = op_name
        self.op_type_name = op_type_name
        self.metatype = metatype
        self.is_shared = is_shared
        self.resource_name = resource_name
        self.attrs = attrs


class EdgeDesc:
    """
    Contains description of an edge in the TensorFlow `FuncGraph`.
    This information is required for `NNCFGraphEdge` creation.
    """

    def __init__(self,
                 producer_op_name: str,
                 output_port_id: int,
                 consumer_op_name: str,
                 input_port_id: int,
                 tensor_shape: List[int],
                 tensor_dtype: Dtype):
        """
        Initializes description of an edge.

        :param producer_op_name: Name of the node where the edge comes out.
        :param output_port_id: Output port id.
        :param consumer_op_name: Name of the node where the edge comes in.
        :param input_port_id: Input port id.
        :param tensor_shape: Shape of the tensor corresponding to the edge.
        :param tensor_dtype: Dtype of tensor.
        """
        self.producer_op_name = producer_op_name
        self.output_port_id = output_port_id
        self.consumer_op_name = consumer_op_name
        self.input_port_id = input_port_id
        self.tensor_shape = tensor_shape
        self.tensor_dtype = tensor_dtype


class SubclassedConverter:
    """
    Converts the NNCF network to the NNCF graph.
    """

    def __init__(self,
                 nncf_network: NNCFNetwork,
                 training: bool = False):
        """
        Initializes the subclassed converter.

        :param nncf_network: The NNCF network.
        :param training: Mode of the model.
        """
        self._nncf_network = nncf_network
        self._training = training

    def convert(self) -> NNCFGraph:
        """
        Builds NNCF graph from provided NNCF network.

        :return: The NNCF graph.
        """
        concrete_function = tf.function(self._nncf_network).get_concrete_function(self._nncf_network.input_signature,
                                                                                  training=self._training)  # training?
        node_descs, edge_descs = SubclassedConverter._collect_tfgraph_description(concrete_function.graph)

        nncf_graph = NNCFGraph()
        op_name_to_node_id_map = {}
        for desc in node_descs:
            nncf_node = nncf_graph.add_nncf_node(
                node_name=desc.op_name,
                node_type=desc.op_type_name,
                node_metatype=desc.metatype,
                layer_attributes=desc.attrs,
                layer_name=desc.resource_name if desc.resource_name else desc.op_name,
                is_shared=desc.is_shared
            )
            op_name_to_node_id_map[desc.op_name] = nncf_node.node_id

        for desc in edge_descs:
            from_node_id = op_name_to_node_id_map[desc.producer_op_name]
            to_node_id = op_name_to_node_id_map[desc.consumer_op_name]

            nncf_graph.add_edge_between_nncf_nodes(
                from_node_id,
                to_node_id,
                tensor_shape=desc.tensor_shape,
                input_port_id=desc.input_port_id,
                output_port_id=desc.output_port_id,
                dtype=desc.tensor_dtype
            )

        return nncf_graph

    @staticmethod
    def _get_data_format(op: tf.Operation) -> str:
        """
        Returns data format for the TensorFlow operation in the Keras format.
        Returns default data format if the operation does not have it.

        :param op: TensorFlow operation.
        :return: String `channels_last` or `channels_first`.
        """
        try:
            data_format = op.get_attr('data_format')
        except ValueError:
            data_format = None

        if data_format:
            to_keras_data_format = {
                'NHWC': 'channels_last',
                'NCHW': 'channels_first',
                'NDHWC': 'channels_last',
                'NCDHW': 'channels_first',
            }
            return to_keras_data_format[data_format.decode('utf-8')]

        return tf.keras.backend.image_data_format()

    @staticmethod
    def _collect_tfgraph_description(graph) -> Tuple[List[NodeDesc], List[EdgeDesc]]:
        """
        Traverses the graph and collects information about nodes and edges
        which should be included in the NNCF graph.

        :param graph: TensorFlow `FuncGraph`.
        :return: Description of nodes and edges which should be included in the NNCF graph.
        """
        captured_input_ops = {}  # type: Dict[str, tf.Operation]
        for tensor in graph.internal_captures:
            captured_input_ops[tensor.op.name] = tensor.op

        regular_input_ops = {}  # type: Dict[str, tf.Operation]
        for tensor in graph.inputs:
            if tensor.op.name not in captured_input_ops:
                regular_input_ops[tensor.op.name] = tensor.op

        # Traverse the graph and mark all nodes reachable from `regular_input_ops`.
        # The NNCF graph contains only reachable nodes.
        # If `op_name` in `visited_nodes` the node with `op_name`
        # reachable from `regular_input_ops`.
        queue = deque(regular_input_ops.values())
        visited_nodes = dict(regular_input_ops)

        while len(queue) != 0:
            v = queue.popleft()
            # A successor of node `v` is a node `u` such that exists
            # a directed edge (v, u) from `v` to `u`.
            successors = (op for tensor in v.outputs for op in tensor.consumers())
            for u in successors:
                if u.name not in visited_nodes:
                    queue.append(u)
                    visited_nodes[u.name] = u

        resource_op_name_to_op_names_map =  SubclassedConverter._get_resource_op_name_to_op_names_map(
            captured_input_ops.values()
        )
        # Reverse map
        op_name_to_resource_op_name_map = {}
        for resource_op_name, op_names in resource_op_name_to_op_names_map.items():
            for op_name in op_names:
                op_name_to_resource_op_name_map[op_name] = resource_op_name

        # Collect descriptions for all visited nodes. The directed edge (v, u)
        # of `FuncGraph` is added only if nodes v, u were visited.
        node_descs = []
        edge_descs = []

        for op in visited_nodes.values():
            metatype = get_op_metatype(op.type)

            node_attributes = TFNodeAttributes(SubclassedConverter._get_data_format(op))

            is_shared = False
            resource_name = None
            if metatype in ALL_LAYER_METATYPES_WITH_WEIGHTS:
                # TODO(andrey-churkin): This code does not work for a quantized model.
                # Need to use a more advanced algorithm to find resource nodes.
                resource_name = op_name_to_resource_op_name_map.get(op.name)
                if resource_name:
                    is_shared = len(resource_op_name_to_op_names_map[resource_name]) > 1

                    assert len(metatype.weight_definitions) == 1
                    port_id = metatype.weight_definitions[0].port_id
                    weight_shape = op.inputs[port_id].shape.as_list()

                    node_attributes = TFWeightedNodeAttributes(
                        node_attributes.get_data_format(),
                        weight_shape
                    )

            node_descs.append(
                NodeDesc(
                    op_name=op.name,
                    op_type_name=op.type,
                    metatype=metatype,
                    is_shared=is_shared,
                    resource_name=resource_name,
                    attrs=node_attributes
                )
            )

            for input_port_id, tensor in enumerate(op.inputs):
                producer_op = tensor.op
                if producer_op.name not in visited_nodes:
                    continue

                edge_descs.append(
                    EdgeDesc(
                        producer_op_name=producer_op.name,
                        output_port_id=tensor.value_index,
                        consumer_op_name=op.name,
                        input_port_id=input_port_id,
                        tensor_shape=tensor.shape.as_list(),
                        tensor_dtype=SubclassedConverter._convert_dtype_to_nncf_format(tensor.dtype)
                    )
                )

        return node_descs, edge_descs

    @staticmethod
    def _convert_dtype_to_nncf_format(dtype: tf.dtypes.DType) -> Dtype:
        if dtype.is_floating:
            tensor_dtype = Dtype.FLOAT
        elif dtype.is_integer:
            tensor_dtype = Dtype.INTEGER
        else:
            raise RuntimeError(f'Unexpected dtype of tensor: {dtype}')

        return tensor_dtype

    @staticmethod
    def _get_resource_op_name_to_op_names_map(captured_input_ops) -> Dict[str, List[str]]:
        """
        Returns mapping from the name of resource node to the name of visited
        nodes that use this resource.

        :param captured_input_ops: Placeholders for weights.
        :return: The mapping from the name of resource node to the name of visited
            nodes that use this resource.
        """
        resource_op_name_to_op_names_map = {}
        for op in captured_input_ops:
            if len(op.outputs) != 1:
                raise RuntimeError(f'Unexpected number of outputs: {len(op.outputs)}')

            output_tensor = op.outputs[0]
            for consumer_op in output_tensor.consumers():
                if len(consumer_op.outputs) != 1:
                    raise RuntimeError(f'Unexpected number of outputs: {len(consumer_op.outputs)}')

                tensor = consumer_op.outputs[0]
                consumers = tensor.consumers()
                if len(consumers) != 1:
                    raise RuntimeError(f'Unexpected number of consumers: {len(consumers)}')

                op_names = resource_op_name_to_op_names_map.setdefault(op.name, [])
                op_names.append(consumers[0].name)
        return resource_op_name_to_op_names_map
