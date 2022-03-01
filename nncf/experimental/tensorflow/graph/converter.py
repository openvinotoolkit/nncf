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

from typing import Optional
from typing import Dict
from typing import Any
from typing import List
from typing import Tuple
from collections import deque

import tensorflow as tf
from tensorflow.python.keras.saving import saving_utils as _saving_utils
from tensorflow.lite.python.util import get_grappler_config as _get_grappler_config
from tensorflow.lite.python.util import run_graph_optimizations as _run_graph_optimizations
from tensorflow.python.framework import convert_to_constants as _convert_to_constants

from nncf.common.graph import NNCFGraph
from nncf.common.graph.layer_attributes import Dtype
from nncf.tensorflow.graph.converter import TFModelConverter
from nncf.tensorflow.graph.metatypes.matcher import get_op_metatype
from nncf.tensorflow.graph.metatypes.common import ALL_LAYER_METATYPES_WITH_WEIGHTS
from nncf.experimental.tensorflow.graph.node_attributes import TFNodeAttributes
from nncf.experimental.tensorflow.graph.node_attributes import TFWeightedNodeAttributes


class TensorFlowGraphBuilder:
    """
    Converts given Keras model to the tf.Graph.
    """

    def __init__(self, model: tf.keras.Model, input_signature):
        """
        Initializes the `TensorFlowGraphBuilder`.

        :param model: The instance of the `tf.keras.Model` class.
        :param input_signature: A structure of the `tf.TensorSpec` objects specifying the
            inputs to the model.
        """
        self._model = model
        self._input_signature = tf.nest.flatten(input_signature)

    def build(self, graph_optimizers: List[str]):
        """
        :param graph_optimizers: A list of the strings that represent
            the list of optimizers.
                - `constfold`
                - `layout`
                - `pruning`
                - `arithmetic`
                - `dependency`
                - `function`
            Look at the [guide](https://www.tensorflow.org/guide/graph_optimization) for more information.
        :return: The optimized tf.Graph.
        """
        # Step 1: Freeze Keras model to frozen graph

        func = _saving_utils.trace_model_call(self._model, self._input_signature)
        concrete_func = func.get_concrete_function()
        frozen_func, graph_def = \
            _convert_to_constants.convert_variables_to_constants_v2_as_graph(concrete_func, lower_control_flow=False)
        # List of input tensors
        input_tensors = [tensor for tensor in frozen_func.inputs if tensor.dtype != tf.dtypes.resource]
        # List of output tensors
        output_tensors = frozen_func.outputs

        # Step 2: Run a Grappler pass to oprimize the TensorFlow graph.

        # Creates a ConfigProto for configuring Grappler
        grappler_config = _get_grappler_config(graph_optimizers)
        # Skip running grappler when there are no optimizers to run. If not,
        # grappler will run with the default optimizer set and it will lead to
        # causing an unexpected behavior.
        if grappler_config.graph_options.rewrite_options.optimizers:
            graph_def = _run_graph_optimizations(
                graph_def,
                input_tensors,
                output_tensors,
                config=grappler_config,
                graph=frozen_func.graph
            )

        # Step 3: Convert the GraphDef to a tf.Graph
        with tf.Graph().as_default() as graph:  # pylint:disable=not-context-manager
            tf.graph_util.import_graph_def(graph_def, name='')

        return graph, input_tensors, output_tensors


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


class SubclassedConverter(TFModelConverter):
    """
    Converts the Keras model, which was created via `tf.keras.Model`
    subclass, to the `NNCFGraph`.

    Uses the frozen TF graph for the provided model with applied
    graph optimizers to create `NNCFGraph`
    """

    def __init__(self,
                 model: tf.keras.Model,
                 input_signature: List[tf.TensorSpec],
                 training: bool = False,
                 graph_optimizers: Optional[List[str]] = None):
        """
        Initializes the subclassed converter.

        :param model: Instance of the `tf.keras.Model` class.
        :param input_signature: A list of `tf.TensorSpec` objects specifying the
            inputs to the model.
        :param training: Mode of the model.
        :param graph_optimizers: A list of the strings that represent
            the list of optimizers.
                - `constfold`
                - `layout`
                - `pruning`
                - `arithmetic`
                - `dependency`
                - `function`
            Look at the [guide](https://www.tensorflow.org/guide/graph_optimization) for more information.
            If `graph_optimizers` is `None` `dependency` optimizer will be used.
        """
        self._model = model
        self._input_signature = input_signature
        self._training = training

        if graph_optimizers is None:
            graph_optimizers = ['dependency']
        self._graph_optimizers = graph_optimizers

        self._tfgraph_builder = TensorFlowGraphBuilder(self._model, self._input_signature)

    def convert(self) -> NNCFGraph:
        """
        Converts the Keras model to the `NNCFGraph` object.

        :return: The `NNCFGraph` object that represents the Keras model
            for compression algorithms.
        """
        tfgraph, input_tensors, _ = self._tfgraph_builder.build(self._graph_optimizers)

        input_op_names = [tensor.op.name for tensor in input_tensors]
        node_descs, edge_descs = SubclassedConverter._collect_tfgraph_descs(tfgraph, input_op_names)

        nncf_graph = SubclassedConverter._create_nncf_graph_from_descs(node_descs, edge_descs)

        return nncf_graph

    @staticmethod
    def _create_nncf_graph_from_descs(node_descs: List[NodeDesc],
                                      edge_descs: List[EdgeDesc]) -> NNCFGraph:
        """
        Creates the NNCF graph from the provided nodes and edges descriptions.

        :param node_descs: A list of `NodeDesc` objects.
        :param edge_descs: A list of `EdgeDesc` objects.
        :return: An instance of the `NNCFGraph` class.
        """
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
    def _collect_tfgraph_descs(graph: tf.Graph,
                               op_names: List[str]) -> Tuple[List[NodeDesc], List[EdgeDesc]]:
        """
        Traverses the TF graph and collects information about nodes and edges
        which should be included to the NNCF graph.

        :param graph: Frozen `tf.Graph` with applied `dependency` optimization.
        :param op_names: A list of names for the input operations ()
        :return: A description of nodes and edges which should be included to the NNCF graph.
        """

        # Traverse the `graph` and mark all ops reachable from the `input_ops`.
        # The op `u` is reachable from the `input_ops` if a directed path
        # from at least one op in `input_ops` to `u` exists in the `graph.`

        # The NNCF graph contains only reachable nodes (ops).

        # If `op_name` in `visited_ops` then operation `visited_ops[op_name]`
        # (i.e. operation with `op_name` name) is reachable from the `input_ops`.
        input_ops = [graph.get_operation_by_name(name) for name in op_names]  # type: List[tf.Operation]
        queue = deque(input_ops)
        visited_ops = {op.name: op for op in input_ops}  # type: Dict[str, tf.Operation]
        while len(queue) != 0:
            v = queue.popleft()
            # A successor of node `v` is a node `u` such that exists
            # a directed edge (v, u) from `v` to `u`.
            successors = (op for tensor in v.outputs for op in tensor.consumers())
            for u in successors:
                if u.name not in visited_ops:
                    queue.append(u)
                    visited_ops[u.name] = u

        op_name_to_const_op_names_map = SubclassedConverter._get_op_name_to_const_op_names_map(graph, visited_ops)

        # Collect descriptions for all visited ops. The directed edge (v, u)
        # of the `graph` is added only if ops v, u were visited.
        node_descs = []
        edge_descs = []
        for op in visited_ops.values():
            metatype = get_op_metatype(op.type)
            node_attributes = TFNodeAttributes(
                SubclassedConverter._get_data_format(op)
            )

            is_shared = False
            const_op_name = None
            if metatype in ALL_LAYER_METATYPES_WITH_WEIGHTS:
                const_op_names = op_name_to_const_op_names_map[op.name]
                if const_op_names:
                    # TODO(andrey-churkin): Currently, we don't have metatypes with
                    # multiple weights definitions.
                    if len(const_op_names) > 1 or len(metatype.weight_definitions) > 1:
                        raise NotImplementedError
                    const_op_name, is_shared = const_op_names[0]

                    # TODO(andrey-churkin): Seems like we can dynamically collect
                    # this information. Need to look at this.
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
                    resource_name=const_op_name,
                    attrs=node_attributes
                )
            )

            for input_port_id, tensor in enumerate(op.inputs):
                producer_op = tensor.op
                if producer_op.name not in visited_ops:
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
        """
        Converts `tf.dtypes.DType` to the NNCF representation.

        :param dtype: An instance of the `tf.dtypes.DType`.
        :return: An instance of the `Dtype`.
        """
        if dtype.is_floating:
            tensor_dtype = Dtype.FLOAT
        elif dtype.is_integer:
            tensor_dtype = Dtype.INTEGER
        else:
            raise RuntimeError(f'Unexpected dtype of tensor: {dtype}')

        return tensor_dtype

    @staticmethod
    def _get_op_name_to_const_op_names_map(graph,
                                           marked_ops: Dict[str, tf.Operation]) -> Dict[str, List[Tuple[str, bool]]]:
        """
        Returns information about constant operations for the `marked_ops`.

        :param graph: Frozen tf.Graph.
        :param marked_ops: A mapping from operation name to operation. The marked operations are
            the operations which reachable from the input operations (`input_ops`) in the `graph`.
        :return: A mapping from the operation name to the list of (const_op_name, is_shared) tuples where
            - `const_op_name` is the name of constant operation which is used by this operation
            - `is_shared` is the boolean flag. Takes one of the following values:
            `True` if the number of operations that use it is greater than 1, `False` otherwise.
        """
        const_ops = [
            op for op in graph.get_operations() if op.type == 'Const'
        ]

        const_op_name_to_op_names_map = {}  # type: Dict[str, List[str]]
        for const_op in const_ops:
            # Traverse the `graph` from the `const_op` and find the reachable ops
            # which are marked as visited (i.e. it's name in `marked_ops` dict)
            # from the `input_ops` in the previous traverse.
            queue = deque([const_op])
            visited_ops = {const_op.name: const_op}
            while len(queue) != 0:
                v = queue.popleft()
                if v.name in marked_ops:
                    const_op_name_to_op_names_map.setdefault(const_op.name, []).append(v.name)
                    continue
                # A successor of node `v` is a node `u` such that exists
                # a directed edge (v, u) from `v` to `u`.
                successors = (op for tensor in v.outputs for op in tensor.consumers())
                for u in successors:
                    if u.name not in visited_ops:
                        queue.append(u)
                        visited_ops[u.name] = u

        op_name_to_const_op_names_map = {}  # type: Dict[str, List[Tuple[str, bool]]]
        for const_op_name, op_names in const_op_name_to_op_names_map.items():
            is_shared = len(op_names) > 1
            for op_name in op_names:
                op_name_to_const_op_names_map.setdefault(op_name, []).append((const_op_name, is_shared))

        return op_name_to_const_op_names_map
