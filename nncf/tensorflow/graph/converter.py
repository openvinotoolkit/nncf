"""
 Copyright (c) 2020 Intel Corporation
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
from typing import Tuple

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from nncf.tensorflow.graph.metatypes.common import DECONV_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.common import GENERAL_CONV_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.matcher import get_keras_layer_metatype
from nncf.tensorflow.graph.metatypes.matcher import get_op_metatype
from nncf.tensorflow.graph.metatypes.nncf_op import OutputNoopMetatype
from nncf.tensorflow.graph.utils import get_shared_node_name
from nncf.tensorflow.graph.utils import is_functional_model
from nncf.tensorflow.graph.utils import is_sequential_model
from nncf.tensorflow.graph.utils import unwrap_layer
from nncf.tensorflow.layers.data_layout import get_input_channel_axis
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFGraphNodeType
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import Dtype

PREFIX_AUXILIARY_OUTPUT_NODE = 'output'


def convert_layer_graph_to_nncf_graph(layer, use_graph_var_names=False) -> NNCFGraph:
    def get_graph_to_layer_var_names_map(concrete_fun):
        names_map = {}
        for layer_var in concrete_fun.variables:
            for value_tensor, graph_name in concrete_fun.graph.captures:
                if layer_var.handle is value_tensor:
                    names_map[graph_name.name.split(':')[0]] = layer_var.name.split(':')[0]
        return names_map

    layer_input_spec = [tf.TensorSpec(input_shape)
                        for input_shape in layer.input_shape] if isinstance(layer.input_shape, list) \
        else tf.TensorSpec(layer.input_shape)
    concr_fn = tf.function(layer).get_concrete_function(layer_input_spec, training=False)
    wrapped_function = convert_variables_to_constants_v2(concr_fn, lower_control_flow=False)

    nodes = wrapped_function.graph.as_graph_def().node
    graph_to_layer_names_map = {} if use_graph_var_names else get_graph_to_layer_var_names_map(concr_fn)

    nncf_graph = NNCFGraph()

    graphdef_node_name_to_id = {}  # type: Dict[str, int]

    for node in nodes:
        nncf_node = nncf_graph.add_nncf_node(node_name=graph_to_layer_names_map.get(node.name, node.name),
                                             node_type=node.op,
                                             node_metatype=get_op_metatype(node.op))
        graphdef_node_name_to_id[node.name] = nncf_node.node_id

    for node in nodes:
        dtype = Dtype.FLOAT if node.attr['dtype'].type == 1 else Dtype.INTEGER
        for idx, input_node in enumerate(node.input):
            nncf_node_id = graphdef_node_name_to_id[node.name]
            if input_node in graphdef_node_name_to_id:
                input_nncf_node_id = graphdef_node_name_to_id[input_node]
                nncf_graph.add_edge_between_nncf_nodes(input_nncf_node_id,
                                                       nncf_node_id,
                                                       tensor_shape=None,  # TODO(vshampor): fix
                                                       input_port_id=idx, dtype=dtype)

    return nncf_graph


def convert_keras_model_to_nncf_graph(model: tf.keras.Model) -> NNCFGraph:
    """
    Convert Keras model graph to the NNCFGraph

    :param model: Keras model
    :return: NNCFGraph
    """
    func_model = is_functional_model(model)
    seq_model = is_sequential_model(model)

    if not func_model and not seq_model:
        RuntimeError('convert_keras_model_to_nncf_graph function supports '
                     'only sequential or functional models')

    if func_model:
        converter = FunctionalConverter(model)
    else:
        converter = SequentialConverter(model)

    return converter.convert()


class TFModelConverter(ABC):
    def __init__(self, model: tf.keras.Model):
        self._model = model

    @staticmethod
    def _get_layer_type(layer_config: Dict) -> str:
        if layer_config['class_name'] == 'TensorFlowOpLayer':
            return layer_config['config']['node_def']['op']
        if layer_config['class_name'] == 'NNCFWrapper':
            # Return class_name of wrapped layer_config
            return layer_config['config']['layer']['class_name']
        return layer_config['class_name']

    @staticmethod
    def _get_layer_dtype(layer_config: Dict) -> str:
        dtype = layer_config['config']['dtype']
        if layer_config['class_name'] == 'TensorFlowOpLayer':
            dtype = layer_config['config']['node_def'].get('attr', {}).get('T', {}).get('type') or dtype
        return dtype

    @staticmethod
    def _prepare_shape(shape) -> List:
        if not isinstance(shape, list):
            return [shape]
        return shape

    @abstractmethod
    def convert(self) -> NNCFGraph:
        pass

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


class FunctionalConverter(TFModelConverter):
    def __init__(self, model: tf.keras.Model):
        super().__init__(model)
        self._raw_nodes = {}  # type: Dict[str, List[Dict]]
        self._layer_info = {}  # type: Dict[str, Dict]
        self._collect_layer_information()
        self._node_info = {}  # type: Dict[str, Dict]
        self._layer_name_to_node_names = defaultdict(set)
        self._collect_node_information()
        self._edge_info = {}  # type: Dict[Tuple[str, str], Dict]
        self._collect_edge_information()

    def _collect_layer_information(self):
        model_config = self._model.get_config()
        for layer_config in model_config['layers']:
            layer_name = layer_config['name']
            layer_type = self._get_layer_type(layer_config)
            layer_dtype = self._get_layer_dtype(layer_config)
            data_format = layer_config['config'].get('data_format')
            model_layer = self._get_layer(layer_name)
            layer_metatype = get_keras_layer_metatype(model_layer)
            self._layer_info[layer_name] = {
                        'type': layer_type,
                        'metatype': layer_metatype,
                        'dtype': layer_dtype,
                        'data_format': data_format,
                        'inbound_nodes': layer_config.get('inbound_nodes')
                    }

    def _collect_node_information(self):
        model_config = self._model.get_config()
        for layer_config in model_config['layers']:
            layer_name = layer_config['name']
            layer = self._get_layer(layer_name)
            if layer_config['inbound_nodes']:
                instances_count = len(layer_config['inbound_nodes'])
                is_shared = instances_count > 1
                for i in range(instances_count):
                    node_name = get_shared_node_name(layer_name, i) if is_shared else layer_name
                    input_shapes = self._prepare_shape(layer.inbound_nodes[i].input_shapes)
                    output_shapes = self._prepare_shape(layer.inbound_nodes[i].output_shapes)

                    self._node_info[node_name] = {
                        'layer_name': layer_name,
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
                    'inbound_node_idx': None,
                    'input_shapes': input_shapes,
                    'output_shapes': output_shapes,
                }


    def _is_layer_shared(self, layer_name: str):
        # Only gives valid results if called after _collect_node_information()
        return len(self._layer_name_to_node_names[layer_name]) > 1

    def _collect_edge_information(self):
        model_config = self._model.get_config()
        for layer_config in model_config['layers']:
            layer_name = layer_config['name']
            for layer_instance_idx, instance_input_info in enumerate(layer_config['inbound_nodes']):
                if self._is_layer_shared(layer_name):
                    node_name = get_shared_node_name(layer_name, layer_instance_idx)
                else:
                    node_name = layer_name
                input_shapes = self._node_info[node_name]['input_shapes']
                for layer_instance_input_port_id, inbound_node in enumerate(instance_input_info):
                    producer_layer_name, producer_layer_instance, *_ = inbound_node
                    if self._is_layer_shared(producer_layer_name):
                        producer_node_name = get_shared_node_name(producer_layer_name, producer_layer_instance)
                    else:
                        producer_node_name = producer_layer_name

                    producer_layer_info = self._layer_info[producer_layer_name]
                    dtype = Dtype.FLOAT if 'float' in producer_layer_info['dtype'].lower() else Dtype.INTEGER
                    edge = (producer_node_name, node_name)
                    tensor_shape = input_shapes[layer_instance_input_port_id]
                    self._edge_info[edge] = {
                        'tensor_shape': tensor_shape,
                        'dtype': dtype,
                        'associated_node_input_port_id': layer_instance_input_port_id
                    }

    def convert(self) -> NNCFGraph:
        nncf_graph = NNCFGraph()

        node_name_vs_nncf_node_ids = {}  # type: Dict[str, int]
        output_node_id_vs_model_output_idx = {}  # type: Dict[int, int]

        for node_name, node_info in self._node_info.items():
            layer_name = node_info['layer_name']
            node_name_vs_nncf_node_ids[layer_name] = []
            layer_info = self._layer_info[layer_name]
            metatype = layer_info['metatype']
            layer = self._get_layer(layer_name)
            if metatype in GENERAL_CONV_LAYER_METATYPES:
                layer_attributes = _get_conv_layer_attributes(self._get_layer(layer_name))
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

        for edge, edge_info in self._edge_info.items():
            from_node_name, to_node_name = edge
            from_node_id = node_name_vs_nncf_node_ids[from_node_name]
            to_node_id = node_name_vs_nncf_node_ids[to_node_name]
            nncf_graph.add_edge_between_nncf_nodes(from_node_id,
                                                   to_node_id,
                                                   tensor_shape=edge_info['tensor_shape'],
                                                   input_port_id=edge_info['associated_node_input_port_id'],
                                                   dtype=edge_info['dtype'])

        for output_node_id, model_output_idx in output_node_id_vs_model_output_idx.items():
            # Ticket: 56853
            # We won't add an NNCF output auxiliary node for all of the NNCF nodes corresponding to real
            # model output, but only for the nodes that do not serve as a tensor source for any other node.
            # The reason is that current TF capabilities do not allow to insert post-hooks after TF functional model
            # output nodes without changing the name of the corresponding output, which won't be obvious to the user.
            nncf_node = nncf_graph.get_node_by_id(output_node_id)
            if not nncf_graph.get_next_nodes(nncf_node):
                output_aux_node_name = PREFIX_AUXILIARY_OUTPUT_NODE + '_{}'.format(model_output_idx)
                output_node = nncf_graph.add_nncf_node(
                    node_name=output_aux_node_name,
                    node_type=NNCFGraphNodeType.OUTPUT_NODE,
                    node_metatype=OutputNoopMetatype)
                node_info = self._node_info[nncf_node.node_name]  # works if _node_info keys are identical to node_names
                nncf_graph.add_edge_between_nncf_nodes(nncf_node.node_id,
                                                       output_node.node_id,
                                                       tensor_shape=node_info['output_shapes'][0],
                                                       input_port_id=0,
                                                       dtype=Dtype.FLOAT)

        return nncf_graph


class SequentialConverter(TFModelConverter):
    def convert(self) -> NNCFGraph:
        nncf_graph = NNCFGraph()
        producer_layer_id = None
        model_config = self._model.get_config()
        layer_name = None
        for layer_config in model_config['layers']:
            layer_name = layer_config['config']['name']
            layer_type = self._get_layer_type(layer_config)
            layer_dtype = self._get_layer_dtype(layer_config)
            data_format = layer_config['config'].get('data_format')
            model_layer = self._get_layer(layer_name)
            layer_metatype = get_keras_layer_metatype(model_layer)

            attrs = dict(type=layer_type,
                         dtype=layer_dtype,
                         metatype=layer_metatype,
                         data_format=data_format,
                         in_ports=[0],
                         out_ports=[0],
                         is_shared=False)

            layer_attributes = None
            if layer_metatype in GENERAL_CONV_LAYER_METATYPES:
                layer_attributes = _get_conv_layer_attributes(self._model.get_layer(layer_name))
                attrs.update({NNCFGraph.LAYER_ATTRIBUTES: layer_attributes})

            nncf_node = nncf_graph.add_nncf_node(layer_name,
                                                 node_type=layer_type,
                                                 node_metatype=layer_metatype,
                                                 layer_attributes=layer_attributes,
                                                 layer_name=layer_name,
                                                 is_shared=False)

            if producer_layer_id is not None:
                input_shapes = self._prepare_shape(self._model.get_layer(layer_name).input_shape)
                nncf_graph.add_edge_between_nncf_nodes(producer_layer_id,
                                                       nncf_node.node_id,
                                                       tensor_shape=input_shapes[0],
                                                       input_port_id=0,
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
                                                   input_port_id=0, dtype=Dtype.FLOAT)

        return nncf_graph


def _get_conv_layer_attributes(layer: tf.keras.layers.Layer) -> ConvolutionLayerAttributes:
    channel_axis = get_input_channel_axis(layer)
    layer_ = unwrap_layer(layer)
    layer_metatype = get_keras_layer_metatype(layer_, determine_subtype=False)
    strides = layer_.strides[0]
    groups = layer_.groups
    kernel_size = layer_.kernel_size

    transpose = layer_metatype in DECONV_LAYER_METATYPES

    return ConvolutionLayerAttributes(layer.trainable,
                                      layer.get_input_shape_at(0)[channel_axis],
                                      layer.get_output_shape_at(0)[channel_axis],
                                      kernel_size,
                                      strides,
                                      groups, transpose=transpose,
                                      padding_values=([0, 0, 0, 0]))
