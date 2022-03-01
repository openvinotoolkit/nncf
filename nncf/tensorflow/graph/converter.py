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
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple
from typing import Type

import tensorflow as tf
from tensorflow.core.framework.node_def_pb2 import NodeDef
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNodeName
from nncf.common.graph import OperatorMetatype
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import MultipleInputLayerAttributes
from nncf.common.graph.layer_attributes import ReshapeLayerAttributes
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.utils import get_concat_axis
from nncf.common.utils.logger import logger as nncf_logger
from nncf.tensorflow.graph.metatypes.common import DECONV_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.common import DEPTHWISE_CONV_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.common import \
    LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_MULTIPLE_CONCAT_INPUTS
from nncf.tensorflow.graph.metatypes.common import GENERAL_CONV_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.common import LINEAR_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.common import RESHAPE_METATYPES
from nncf.tensorflow.graph.metatypes.matcher import get_keras_layer_metatype
from nncf.tensorflow.graph.metatypes.matcher import get_op_metatype
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from nncf.tensorflow.graph.metatypes.tf_ops import WEIGHTABLE_TF_OP_METATYPES
from nncf.tensorflow.graph.utils import get_shared_node_name
from nncf.tensorflow.graph.utils import reformat_inbound_nodes_for_oplambda
from nncf.tensorflow.graph.utils import is_builtin_layer
from nncf.tensorflow.graph.utils import is_functional_model
from nncf.tensorflow.graph.utils import is_sequential_model
from nncf.tensorflow.graph.utils import unwrap_layer
from nncf.tensorflow.layers.data_layout import get_input_channel_axis
from nncf.common.graph.definitions import NNCFGraphNodeType

PREFIX_AUXILIARY_OUTPUT_NODE = 'output'


class TFLayerInfo:
    def __init__(self, layer_name: str, instance_idx: int):
        self.layer_name = layer_name
        self.instance_idx = instance_idx


def convert_keras_model_to_nncf_graph(model: tf.keras.Model) -> NNCFGraph:
    """
    Convert Keras model graph to the NNCFGraph

    :param model: Keras model
    :return: NNCFGraph
    """
    converter = TFModelConverterFactory.create(model)
    return converter.convert()


class CustomLayerNodeInfo:
    """
    Describes a future NNCFGraph node corresponding to a TF operation inside a custom layer.
    """

    def __init__(self, graphdef_node_name: str,
                 custom_layer_name: str, target_node_name: NNCFNodeName,
                 node_type: str, node_metatype: Type[OperatorMetatype], weight_node_name: str,
                 dtype: Dtype):
        """
        :param graphdef_node_name: The name of the TF operation node in the GraphDef representation
          of the custom layer.
        :param custom_layer_name: The name of the custom layer in the model that hosts it.
        :param target_node_name: The name of the future NNCFNode that will be present in NNCFGraph
          to represent this TF operation in the corresponding custom layer.
        :param node_type: The TF-specific string describing the type of the TF operation.
        :param node_metatype: The metatype to be associated with this operation in the NNCFGraph
        :param weight_node_name: The name of the NNCFGraph node that corresponds to the weight of the operation.
        :param dtype: The dtype of the operation.
        """
        self.graphdef_node_name = graphdef_node_name
        self.custom_layer_name = custom_layer_name
        self.target_node_name = target_node_name
        self.node_type = node_type
        self.node_metatype = node_metatype
        self.weight_node_name = weight_node_name
        self.dtype = dtype


class CustomLayerEdgeInfo:
    """
    Describes an edge inside a custom layer graph representation in TF.
    """

    def __init__(self, tensor_shape: List[int],
                 input_port_id: int,
                 output_port_id: int,
                 dtype: Dtype):
        """
        :param tensor_shape: The shape of the associated tensor.
        :param input_port_id: The ID of the input tensor among the "to_node" inputs.
        :param output_port_id: The ID of the output tensor among the "from_node" outputs.
        :param dtype: The dtype of the tensor.
        """
        self.tensor_shape = tensor_shape
        self.input_port_id = input_port_id
        self.output_port_id = output_port_id
        self.dtype = dtype


class CustomLayerInfo:
    """
    Describes a custom layer in the TF model from the conversion standpoint.
    """

    def __init__(self):
        self.requires_inputs_from_nodes = set()  # type: Set[str]
        self.gives_outputs_from_nodes = set()  # type: Set[str]
        self.shared_weight_node_names_vs_weighted_op_node_names = defaultdict(set)  # type: Dict[str, Set[str]]
        self.node_infos = {}  # type: Dict[str, CustomLayerNodeInfo]
        self.edge_infos = {}  # type: Dict[Tuple[str, str], CustomLayerEdgeInfo]
        self.graphdef_node_name_to_pretty_node_name = {}  # type: Dict[str, str]


class TFModelConverter(ABC):
    """
    Abstract class which describes the interface needed to convert
    the Keras model to the `NNCFGraph` object.
    """

    @abstractmethod
    def convert(self) -> NNCFGraph:
        """
        Converts the Keras model to the `NNCFGraph` object.

        :return: The `NNCFGraph` object that represents the Keras model
            for compression algorithms.
        """


class BaseFunctionalSequentialConverter(TFModelConverter):
    """
    A base class for the `FunctionalConverter` and `SequentialConverter` classes.
    Contains the implementation of the common methods.
    """

    def __init__(self, model: tf.keras.Model):
        self._model = model
        self._node_info = {}  # type: Dict[str, Dict]
        self._custom_layer_infos = self._collect_custom_layer_infos(self._model)
        self._nncf_node_names_vs_custom_layer_name = {}  # type: Dict[NNCFNodeName, str]

    @staticmethod
    def _get_type_spec(tensor):
        if isinstance(tensor, KerasTensor):
            return tensor.type_spec
        return tf.TensorSpec.from_tensor(tensor)

    def _collect_custom_layer_infos(self, model: tf.keras.Model,
                                    use_graph_var_names: bool = False) -> Dict[str, CustomLayerInfo]:
        custom_layers = BaseFunctionalSequentialConverter.get_custom_layers(model)
        retval = {}
        for layer_name, layer in custom_layers.items():
            layer_input_spec = [self._get_type_spec(tensor)
                                for tensor in layer.input] if isinstance(layer.input, list) \
                else self._get_type_spec(layer.input)

            # TODO (vshampor) : Use the custom layer's inbound_nodes/outbound_nodes to determine what edges
            #  should connect it to the rest of the graph. Currently the custom layer
            #  subgraph will be present in the NNCFGraph after conversion, which is useful
            #  for purposes of weight modification target point creation and usage,
            #  but the subgraph won't be connected to the rest of the graph; in the main graph
            #  component, the custom layer will still be represented by a single node
            concr_fn = tf.function(layer).get_concrete_function(layer_input_spec, training=False)
            wrapped_function = convert_variables_to_constants_v2(concr_fn, lower_control_flow=False)

            graphdef_nodes = wrapped_function.graph.as_graph_def().node
            graphdef_name_to_layer_var_map = {} if use_graph_var_names else \
                BaseFunctionalSequentialConverter._get_graphdef_name_to_layer_var_map(concr_fn)
            nodes = {graphdef_name_to_layer_var_map.get(node.name, node.name): node for node in graphdef_nodes}
            graphdef_node_name_vs_node = {node.name: node for node in graphdef_nodes}

            custom_layer_info = CustomLayerInfo()
            for pretty_node_name, node in nodes.items():
                custom_layer_info.graphdef_node_name_to_pretty_node_name[node.name] = pretty_node_name

            for pretty_node_name, node in nodes.items():
                weight_node_name = None
                metatype = get_op_metatype(node.op)
                if metatype in WEIGHTABLE_TF_OP_METATYPES:
                    graphdef_weight_node_name = self._get_graphdef_node_name_for_custom_layer_node_weight(
                        node,
                        graphdef_node_name_vs_node)
                    if graphdef_weight_node_name in graphdef_name_to_layer_var_map:
                        weight_node_name = graphdef_name_to_layer_var_map[graphdef_weight_node_name]
                    else:
                        nncf_logger.warning('Could not associate a weighted custom layer node {} '
                                            'with a weight attribute of the custom layer - the corresponding weight '
                                            'will not be compressed! Make sure that the corresponding custom layer '
                                            'weight has a name.'.format(pretty_node_name))

                custom_layer_info.node_infos[pretty_node_name] = CustomLayerNodeInfo(
                    graphdef_node_name=node.name,
                    custom_layer_name=layer_name,
                    target_node_name=pretty_node_name,
                    node_type=node.op,
                    node_metatype=get_op_metatype(node.op),
                    weight_node_name=weight_node_name,
                    dtype=Dtype.FLOAT if node.attr['dtype'].type == 1 else Dtype.INTEGER)

                custom_layer_info.shared_weight_node_names_vs_weighted_op_node_names[weight_node_name].add(
                    pretty_node_name)

                for idx, input_graphdef_node_name_and_output_port_str in enumerate(node.input):
                    if '^' in input_graphdef_node_name_and_output_port_str:
                        continue  # Skip control_inputs
                    splits = input_graphdef_node_name_and_output_port_str.split(':')
                    if len(splits) == 1:
                        input_graphdef_node_name = splits[0]
                        output_port_id = 0
                    elif len(splits) == 2:
                        input_graphdef_node_name = splits[0]
                        output_port_id = int(splits[1])
                    else:
                        raise RuntimeError("Could not parse NodeDef's input field!")

                    pretty_input_node_name = \
                        custom_layer_info.graphdef_node_name_to_pretty_node_name[input_graphdef_node_name]

                    # TODO (vshampor): add proper tensor_shape, will probably involve
                    #                  running as_graph_def(add_shapes=True)
                    custom_layer_info.edge_infos[(pretty_input_node_name, pretty_node_name)] = \
                        CustomLayerEdgeInfo(tensor_shape=None,
                                            input_port_id=idx,
                                            output_port_id=output_port_id,
                                            dtype=custom_layer_info.node_infos[pretty_node_name].dtype)
                retval[layer_name] = custom_layer_info
        return retval

    def _get_layer_output_dtype(self, layer_config: Dict) -> Dtype:
        if layer_config['class_name'] in ['Functional', 'Sequential']:
            return self._get_layer_output_dtype(layer_config['config']['layers'][0])

        layer_name = layer_config['config']['name']
        if layer_config['class_name'] == 'TensorFlowOpLayer':
            layer_name = 'tf_op_layer_' + layer_name

        if layer_config['class_name'] == 'InputLayer':
            return Dtype.FLOAT

        keras_layer = self._get_layer(layer_name)
        if isinstance(keras_layer.output, (tf.Tensor, KerasTensor)):
            dtype = keras_layer.output.dtype
        else:
            # In case of multiple outputs, assume all outputs have the same type
            dtype = keras_layer.output[0].dtype

        if dtype == tf.int32:
            return Dtype.INTEGER
        return Dtype.FLOAT

    @staticmethod
    def _get_layer_type(layer_config: Dict) -> str:
        if layer_config['class_name'] == 'TensorFlowOpLayer':
            return layer_config['config']['node_def']['op']
        if layer_config['class_name'] in ['TFOpLambda', 'SlicingOpLambda']:
            return layer_config['config']['function']
        if layer_config['class_name'] == 'NNCFWrapper':
            # Return class_name of wrapped layer_config
            return layer_config['config']['layer']['class_name']
        return layer_config['class_name']

    @staticmethod
    def get_custom_layers(model: tf.keras.Model) -> Dict[str, tf.Module]:
        """
        Returns the mapping of custom layer names in the model vs associated custom layer
        module objects.

        :param model: The model in which to search for custom layers.
        :return: A dict of custom layer names vs custom layer modules.
        """
        custom_layers = {}
        for layer in model.submodules:
            if not is_builtin_layer(layer):
                custom_layers[layer.name] = layer
        return custom_layers

    @staticmethod
    def _get_graphdef_name_to_layer_var_map(concrete_fun) -> Dict[str, str]:
        names_map = {}
        inverse_map = defaultdict(set)
        for layer_var in concrete_fun.variables:
            for value_tensor, graphdef_name in concrete_fun.graph.captures:
                if layer_var.handle is value_tensor:
                    graphdef_name = graphdef_name.name.split(':')[0]
                    layer_var_name = layer_var.name.split(':')[0]
                    inverse_map[layer_var_name].add(graphdef_name)
                    names_map[graphdef_name] = layer_var_name

        for graphdef_names in inverse_map.values():
            if len(graphdef_names) > 1:
                # Name collision - remove all collided entries
                for graphdef_name in graphdef_names:
                    del names_map[graphdef_name]

        return names_map

    @staticmethod
    def _get_graphdef_node_name_for_custom_layer_node_weight(weighted_node: NodeDef,
                                                             graphdef_nodes: Dict[str, NodeDef]) -> str:
        def get_node_name(graphdef_node_name: str):
            return graphdef_node_name.split(':')[0]

        weight_node_name = None
        previous_node_names = [get_node_name(node_input) for node_input in weighted_node.input]
        while previous_node_names:
            weight_node_name = get_node_name(previous_node_names[-1])
            weight_node = graphdef_nodes[weight_node_name]  # TODO (vshampor): how correct is this actually?
            previous_node_names = [get_node_name(node_input) for node_input in weight_node.input]

            # Filter control inputs, whatever these are
            previous_node_names = list(filter(lambda x: '^' not in x, previous_node_names))
        if weight_node_name is None:
            raise RuntimeError("Could not find a weight node for a weighted node {}".format(weighted_node.name))
        return weight_node_name

    @staticmethod
    def _prepare_shape(shape) -> List:
        if not isinstance(shape, list):
            return [shape]
        return shape

    def get_layer_info_for_node(self, node_name: NNCFNodeName) -> Tuple[bool, TFLayerInfo]:
        """
        :param node_name: The node name in the converted NNCFGraph
        :return: A flag indicating whether the node corresponds to a custom layer,
          and the additional TF-specific information about the layer underlying the node.
        """
        if node_name in self._nncf_node_names_vs_custom_layer_name:
            is_custom = True
            custom_layer_name = self._nncf_node_names_vs_custom_layer_name[node_name]
            insertion_data = TFLayerInfo(custom_layer_name,
                                         instance_idx=0)
        else:
            is_custom = False
            node_tf_data = self._node_info[node_name]
            layer_name = node_tf_data['layer_name']
            instance_idx = node_tf_data['inbound_node_idx']
            if instance_idx is None:
                instance_idx = 0

            insertion_data = TFLayerInfo(layer_name,
                                         instance_idx=instance_idx)
        return is_custom, insertion_data

    def get_node_names_vs_custom_layer_names(self) -> Dict[NNCFNodeName, str]:
        """
        :return: A mapping of NNCFNode names corresponding to custom layers vs the corresponding
          custom layer name.
        """
        return self._nncf_node_names_vs_custom_layer_name

    def _get_layer(self, layer_name: str) -> tf.keras.layers.Layer:
        try:
            return self._model.get_layer(layer_name)
        except ValueError:
            for layer in self._model.submodules:
                if not isinstance(layer, tf.keras.layers.Layer):
                    continue
                if layer.name == layer_name:
                    return layer

        raise ValueError(f'No such layer: {layer_name}.')

    def _add_custom_layer_subgraph(self, nncf_graph: NNCFGraph, custom_layer_name: str) -> NNCFGraph:
        # TODO (vshampor): filter meaningless ops such as Identity, resource read etc.
        custom_layer_info = self._custom_layer_infos[custom_layer_name]
        node_name_vs_nncf_node_ids = {}  # type: Dict[NNCFNodeName, int]
        for node_info in custom_layer_info.node_infos.values():
            weight_node_name = node_info.weight_node_name
            is_shared = False
            if weight_node_name is not None:
                shared_node_dict = custom_layer_info.shared_weight_node_names_vs_weighted_op_node_names
                is_shared = len(shared_node_dict[weight_node_name]) > 1
            nncf_node = nncf_graph.add_nncf_node(node_name=node_info.target_node_name,
                                                 node_type=node_info.node_type,
                                                 node_metatype=node_info.node_metatype,
                                                 # TODO (vshampor): collect layer attributes for custom nodes
                                                 layer_attributes=None,
                                                 layer_name=node_info.weight_node_name,  # sic!
                                                 is_shared=is_shared,
                                                 ignored_algorithms=['quantization'])
            node_name_vs_nncf_node_ids[node_info.target_node_name] = nncf_node.node_id
            self._nncf_node_names_vs_custom_layer_name[node_info.target_node_name] = custom_layer_name
        for edge, edge_data in custom_layer_info.edge_infos.items():
            from_node_name, to_node_name = edge
            from_node_id = node_name_vs_nncf_node_ids[from_node_name]
            to_node_id = node_name_vs_nncf_node_ids[to_node_name]
            nncf_graph.add_edge_between_nncf_nodes(from_node_id, to_node_id,
                                                   tensor_shape=edge_data.tensor_shape,
                                                   input_port_id=edge_data.input_port_id,
                                                   output_port_id=edge_data.output_port_id,
                                                   dtype=edge_data.dtype)
        return nncf_graph


class TFModelConverterFactory:
    @staticmethod
    def create(model) -> TFModelConverter:
        func_model = is_functional_model(model)
        seq_model = is_sequential_model(model)

        if not func_model and not seq_model:
            RuntimeError('Only sequential or functional models are supported')

        if func_model:
            converter = FunctionalConverter(model)
        else:
            converter = SequentialConverter(model)
        return converter


class FunctionalConverter(BaseFunctionalSequentialConverter):
    """
    Converter for TF models that use the Functional API.
    """
    def __init__(self, model: tf.keras.Model):
        super().__init__(model)
        self._model_config = self._model.get_config()
        self._layer_info = {}  # type: Dict[str, Dict]
        self._collect_layer_information()
        self._layer_name_to_node_names = defaultdict(set)
        self._collect_node_information()
        self._edge_info = {}  # type: Dict[Tuple[str, str], Dict]
        self._collect_edge_information()

    def _collect_layer_information(self):
        for layer_config in self._model_config['layers']:
            layer_name = layer_config['name']
            layer_type = self._get_layer_type(layer_config)
            layer_output_dtype = self._get_layer_output_dtype(layer_config)
            data_format = layer_config['config'].get('data_format')
            model_layer = self._get_layer(layer_name)
            layer_metatype = get_keras_layer_metatype(model_layer)
            self._layer_info[layer_name] = {
                        'type': layer_type,
                        'metatype': layer_metatype,
                        'dtype': layer_output_dtype,
                        'data_format': data_format,
                        'inbound_nodes': layer_config.get('inbound_nodes')
                    }

    def _collect_node_information(self):
        for layer_config in self._model_config['layers']:
            layer_name = layer_config['name']
            if layer_name not in self._custom_layer_infos:
                self._add_regular_layer_nodes(layer_config)
            else:
                # TODO (vshampor): Instead of adding a single node for custom layer and an entire
                #  unconnected subgraph along with it, stitch the subgraph into the main graph
                #  properly
                self._add_regular_layer_nodes(layer_config)

    def _add_regular_layer_nodes(self, layer_config: Dict):
        layer_name = layer_config['name']
        layer = self._get_layer(layer_name)
        if layer_config['inbound_nodes']:
            instances_count = len(layer_config['inbound_nodes'])
            is_shared = instances_count > 1
            for i in range(instances_count):
                node_name = get_shared_node_name(layer_name, i) if is_shared else layer_name
                input_shapes = [tuple(tensor.shape) for tensor in layer.inbound_nodes[i].keras_inputs]
                output_shapes = self._prepare_shape(layer.inbound_nodes[i].output_shapes)
                self._node_info[node_name] = {
                    'layer_name': layer_name,
                    'target_node_name': layer_name,
                    'inbound_node_idx': i,
                    'input_shapes': input_shapes,
                    'output_shapes': output_shapes,
                }
                self._layer_name_to_node_names[layer_name].add(node_name)
        else:
            node_name = layer_name
            input_shapes = self._prepare_shape(layer.input_shape)
            output_shapes = self._prepare_shape(layer.output_shape)
            self._node_info[node_name] = {
                'layer_name': layer_name,
                'target_node_name': layer_name,
                'inbound_node_idx': None,
                'input_shapes': input_shapes,
                'output_shapes': output_shapes,
            }

    def _is_layer_shared(self, layer_name: str):
        # Only gives valid results if called after collect_node_information()
        return len(self._layer_name_to_node_names[layer_name]) > 1

    def _collect_edge_information(self):
        for layer_config in self._model_config['layers']:
            layer_name = layer_config['name']

            inbound_nodes = layer_config['inbound_nodes']
            if layer_config['class_name'] in ['TFOpLambda', 'SlicingOpLambda']:
                inbound_nodes = reformat_inbound_nodes_for_oplambda(inbound_nodes)

            for layer_instance_idx, inbound_nodes in enumerate(inbound_nodes):
                if self._is_layer_shared(layer_name):
                    node_name = get_shared_node_name(layer_name, layer_instance_idx)
                else:
                    node_name = layer_name
                input_shapes = self._node_info[node_name]['input_shapes']

                layer_instance_input_port_id = 0
                for inbound_node in inbound_nodes:
                    producer_layer_name, producer_layer_instance, \
                    producer_layer_instance_output_port, _ = inbound_node

                    if self._is_layer_shared(producer_layer_name):
                        producer_node_name = get_shared_node_name(producer_layer_name, producer_layer_instance)
                    else:
                        producer_node_name = producer_layer_name

                    producer_layer_info = self._layer_info[producer_layer_name]
                    dtype = producer_layer_info['dtype']
                    tensor_shape = input_shapes[layer_instance_input_port_id]

                    edge = (producer_node_name, node_name)
                    self._edge_info[edge] = {
                        'tensor_shape': tensor_shape,
                        'dtype': dtype,
                        'to_node_input_port_id': layer_instance_input_port_id,
                        'from_node_output_port_id': producer_layer_instance_output_port
                    }
                    layer_instance_input_port_id += 1

    def convert(self) -> NNCFGraph:
        nncf_graph = NNCFGraph()
        node_name_vs_nncf_node_ids = {}  # type: Dict[str, int]
        output_node_id_vs_model_output_idx = {}  # type: Dict[int, int]

        # Regular nodes
        for node_name, node_info in self._node_info.items():
            layer_name = node_info['layer_name']
            node_name_vs_nncf_node_ids[layer_name] = []
            layer_info = self._layer_info[layer_name]
            metatype = layer_info['metatype']
            layer = self._get_layer(layer_name)
            if metatype in DEPTHWISE_CONV_LAYER_METATYPES:
                layer_attributes = _get_conv_layer_attributes(layer, is_depthwise=True)
            elif metatype in GENERAL_CONV_LAYER_METATYPES:
                layer_attributes = _get_conv_layer_attributes(layer, is_depthwise=False)
            elif metatype in LINEAR_LAYER_METATYPES:
                layer_attributes = _get_linear_layer_attributes(layer)
            elif metatype in LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_MULTIPLE_CONCAT_INPUTS:
                layer_attributes = _get_multiple_input_layer_attributes(layer)
            elif metatype in RESHAPE_METATYPES:
                layer_attributes = _get_reshape_layer_attributes(layer)
            else:
                layer_attributes = None
            is_shared = len(self._layer_name_to_node_names[layer_name]) > 1
            nncf_node = nncf_graph.add_nncf_node(node_name=node_name,
                                                 node_type=layer_info['type'],
                                                 node_metatype=metatype,
                                                 layer_attributes=layer_attributes,
                                                 layer_name=layer_name,
                                                 is_shared=is_shared)
            node_name_vs_nncf_node_ids[node_name] = nncf_node.node_id

            #pylint:disable=protected-access
            if layer in self._model._output_layers:
                output_idx = self._model._output_layers.index(layer)
                output_node_id_vs_model_output_idx[nncf_node.node_id] = output_idx

        # Regular edges
        for edge, edge_info in self._edge_info.items():
            from_node_name, to_node_name = edge
            from_node_id = node_name_vs_nncf_node_ids[from_node_name]
            to_node_id = node_name_vs_nncf_node_ids[to_node_name]
            nncf_graph.add_edge_between_nncf_nodes(from_node_id,
                                                   to_node_id,
                                                   tensor_shape=edge_info['tensor_shape'],
                                                   input_port_id=edge_info['to_node_input_port_id'],
                                                   output_port_id=edge_info['from_node_output_port_id'],
                                                   dtype=edge_info['dtype'])

        # Custom nodes and edges
        for custom_layer_name in self._custom_layer_infos:
            nncf_graph = self._add_custom_layer_subgraph(nncf_graph, custom_layer_name)
            # TODO (vshampor): connect the subgraphs with the rest of the graph

        for output_node_id, model_output_idx in output_node_id_vs_model_output_idx.items():
            # Ticket: 56853
            # We won't add an NNCF output auxiliary node for all of the NNCF nodes corresponding to real
            # model output, but only for the nodes that do not serve as a tensor source for any other node.
            # The reason is that current TF capabilities do not allow to insert post-hooks after TF functional model
            # output nodes without changing the name of the corresponding output, which won't be obvious to the user.
            nncf_node = nncf_graph.get_node_by_id(output_node_id)
            if not nncf_graph.get_next_nodes(nncf_node):
                output_aux_node_name = f'{nncf_node.node_name}_{PREFIX_AUXILIARY_OUTPUT_NODE}_{model_output_idx}'
                output_node = nncf_graph.add_nncf_node(
                    node_name=output_aux_node_name,
                    node_type=NNCFGraphNodeType.OUTPUT_NODE,
                    node_metatype=OutputNoopMetatype)
                node_info = self._node_info[nncf_node.node_name]  # works if _node_info keys are identical to node_names
                nncf_graph.add_edge_between_nncf_nodes(nncf_node.node_id,
                                                       output_node.node_id,
                                                       tensor_shape=node_info['output_shapes'][0],
                                                       input_port_id=0,
                                                       output_port_id=0,
                                                       dtype=Dtype.FLOAT)

        return nncf_graph


class SequentialConverter(BaseFunctionalSequentialConverter):
    """
    Converter for the TF models using the Sequential API.
    """
    def convert(self) -> NNCFGraph:
        nncf_graph = NNCFGraph()
        producer_layer_id = None
        model_config = self._model.get_config()

        layer_name = None
        for layer_config in model_config['layers']:
            layer_name = layer_config['config']['name']
            if layer_name in self._custom_layer_infos:
                nncf_graph = self._add_custom_layer_subgraph(nncf_graph, layer_name)
                continue
            layer_type = self._get_layer_type(layer_config)
            layer_output_dtype = self._get_layer_output_dtype(layer_config)
            data_format = layer_config['config'].get('data_format')
            model_layer = self._get_layer(layer_name)
            layer_metatype = get_keras_layer_metatype(model_layer)

            attrs = dict(type=layer_type,
                         dtype=layer_output_dtype,
                         metatype=layer_metatype,
                         data_format=data_format,
                         in_ports=[0],
                         out_ports=[0],
                         is_shared=False)

            layer_attributes = None
            if layer_metatype in DEPTHWISE_CONV_LAYER_METATYPES:
                layer_attributes = _get_conv_layer_attributes(model_layer, is_depthwise=True)
            elif layer_metatype in GENERAL_CONV_LAYER_METATYPES:
                layer_attributes = _get_conv_layer_attributes(model_layer, is_depthwise=False)
            elif layer_metatype in LINEAR_LAYER_METATYPES:
                layer_attributes = _get_linear_layer_attributes(model_layer)
            elif layer_metatype in LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_MULTIPLE_CONCAT_INPUTS:
                layer_attributes = _get_multiple_input_layer_attributes(model_layer)
            elif layer_metatype in RESHAPE_METATYPES:
                layer_attributes = _get_reshape_layer_attributes(model_layer)

            if layer_attributes is not None:
                attrs.update({NNCFGraph.LAYER_ATTRIBUTES: layer_attributes})

            node_name = layer_name
            nncf_node = nncf_graph.add_nncf_node(node_name,
                                                 node_type=layer_type,
                                                 node_metatype=layer_metatype,
                                                 layer_attributes=layer_attributes,
                                                 layer_name=layer_name,
                                                 is_shared=False)

            input_shapes = self._prepare_shape(model_layer.input_shape)
            output_shapes = self._prepare_shape(model_layer.output_shape)
            self._node_info[node_name] = {
                'layer_name': layer_name,
                'target_node_name': layer_name,
                'inbound_node_idx': None,
                'input_shapes': input_shapes,
                'output_shapes': output_shapes,
            }

            if producer_layer_id is not None:
                input_shapes = self._prepare_shape(self._model.get_layer(layer_name).input_shape)
                nncf_graph.add_edge_between_nncf_nodes(producer_layer_id,
                                                       nncf_node.node_id,
                                                       tensor_shape=input_shapes[0],
                                                       input_port_id=0,
                                                       output_port_id=0,
                                                       dtype=Dtype.FLOAT)  # TODO(vshampor): determine from keras layers
            producer_layer_id = nncf_node.node_id

        if layer_name is not None:
            last_producer_layer_name = layer_name
            last_producer_layer_id = producer_layer_id
            output_model_layer = self._model.get_layer(last_producer_layer_name)
            output_aux_node_name = PREFIX_AUXILIARY_OUTPUT_NODE + '_0'
            output_node = nncf_graph.add_nncf_node(node_name=output_aux_node_name,
                                                   node_type=NNCFGraphNodeType.OUTPUT_NODE,
                                                   node_metatype=OutputNoopMetatype)
            nncf_graph.add_edge_between_nncf_nodes(last_producer_layer_id, output_node.node_id,
                                                   tensor_shape=self._prepare_shape(output_model_layer.output_shape),
                                                   input_port_id=0, output_port_id=0, dtype=Dtype.FLOAT)

        return nncf_graph


def _get_multiple_input_layer_attributes(layer: tf.keras.layers.Layer) -> MultipleInputLayerAttributes:
    if hasattr(layer, 'axis'):
        axis = layer.axis
    else:
        input_shape = layer.input_shape
        output_shape = layer.output_shape
        if not isinstance(output_shape, list):
            output_shape = [output_shape]
        axis = get_concat_axis(input_shape, output_shape)
    return MultipleInputLayerAttributes(axis)


def _get_conv_layer_attributes(layer: tf.keras.layers.Layer, is_depthwise: bool = False) -> ConvolutionLayerAttributes:
    channel_axis = get_input_channel_axis(layer)
    layer_ = unwrap_layer(layer)
    layer_metatype = get_keras_layer_metatype(layer_, determine_subtype=False)
    strides = layer_.strides[0]
    in_channels = layer.get_input_shape_at(0)[channel_axis]
    out_channels = layer.get_output_shape_at(0)[channel_axis]

    # TF does not deign to properly set the groups attribute on a depthwise layer, and for compatibility
    # with common code the groups attribute of the returned ConvolutionLayerAttribute must be set equal
    # to the in_channels attribute in order for the layer to be detected as depthwise
    groups = layer_.groups if not is_depthwise else in_channels
    kernel_size = layer_.kernel_size

    transpose = layer_metatype in DECONV_LAYER_METATYPES

    return ConvolutionLayerAttributes(layer.trainable,
                                      in_channels,
                                      out_channels,
                                      kernel_size,
                                      strides,
                                      groups, transpose=transpose,
                                      padding_values=([0, 0, 0, 0]))


def _get_linear_layer_attributes(layer: tf.keras.layers.Layer) -> LinearLayerAttributes:
    channel_axis = get_input_channel_axis(layer)
    in_features = layer.get_input_shape_at(0)[channel_axis]
    out_features = layer.get_output_shape_at(0)[channel_axis]
    return LinearLayerAttributes(layer.trainable,
                                 in_features,
                                 out_features)


def _get_reshape_layer_attributes(layer: tf.keras.layers.Layer) -> ReshapeLayerAttributes:
    input_shape = layer.input_shape
    output_shape = layer.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    return ReshapeLayerAttributes(input_shape, output_shape)
