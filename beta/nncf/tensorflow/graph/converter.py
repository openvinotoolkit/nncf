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
from typing import List, Dict

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from beta.nncf.tensorflow.graph.metatypes.common import GENERAL_CONV_LAYER_METATYPES
from beta.nncf.tensorflow.graph.metatypes.matcher import get_keras_layer_metatype
from beta.nncf.tensorflow.graph.metatypes.matcher import get_op_metatype
from beta.nncf.tensorflow.graph.metatypes.nncf_op import OutputNoopMetatype
from beta.nncf.tensorflow.graph.graph import TFNNCFGraph
from beta.nncf.tensorflow.graph.utils import get_expanded_node_name
from beta.nncf.tensorflow.graph.utils import is_functional_model
from beta.nncf.tensorflow.graph.utils import is_sequential_model
from beta.nncf.tensorflow.graph.utils import unwrap_layer
from beta.nncf.tensorflow.layers.data_layout import get_input_channel_axis
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFGraphNodeType
from nncf.common.graph.module_attributes import ConvolutionLayerAttributes
from nncf.common.graph.module_attributes import Dtype

PREFIX_AUXILIARY_OUTPUT_NODE = 'output'

def _get_layer_type(layer):
    if layer['class_name'] == 'TensorFlowOpLayer':
        return layer['config']['node_def']['op']
    if layer['class_name'] == 'NNCFWrapper':
        # Return class_name of wrapped layer
        return layer['config']['layer']['class_name']
    return layer['class_name']


def _get_layer_dtype(layer):
    dtype = layer['config']['dtype']
    if layer['class_name'] == 'TensorFlowOpLayer':
        dtype = layer['config']['node_def'].get('attr', {}).get('T', {}).get('type') or dtype
    return dtype


def _get_layer(model: tf.keras.Model, layer_name: str) -> tf.keras.layers.Layer:
    try:
        return model.get_layer(layer_name)
    except ValueError:
        for layer in model.submodules:
            if not isinstance(layer, tf.keras.layers.Layer):
                continue
            if layer.name == layer_name:
                return layer

    raise ValueError(f'No such layer: {layer_name}.')


def _process_outputs(outputs, raw_nodes):
    if isinstance(outputs, list):
        for layer_name, instance, out_port in outputs:
            raw_nodes[layer_name][instance]['out_ports'].add(out_port)
    elif isinstance(outputs, dict):
        for output in outputs.values():
            if isinstance(output, list):
                raw_nodes[output[0]][output[1]]['out_ports'].add(output[2])
            elif isinstance(output, dict):
                for layer_name, instance, out_port in output.values():
                    raw_nodes[layer_name][instance]['out_ports'].add(out_port)

    return raw_nodes


def convert_layer_graph_to_nncf_graph(layer, use_graph_var_names=False):
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
    for node in nodes:
        nncf_graph.add_node(
            graph_to_layer_names_map.get(node.name, node.name),
            type=node.op,
            dtype=node.attr['dtype'],
            metatype=get_op_metatype(node.op))

    for node in nodes:
        for input_node in node.input:
            node_name = graph_to_layer_names_map.get(node.name, node.name)
            input_node_name = graph_to_layer_names_map.get(input_node, input_node)
            if input_node_name in nncf_graph.get_all_node_keys():
                nncf_graph.add_edge(input_node_name, node_name)

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
        nncf_graph = _get_nncf_graph_from_functional(model)
    else:
        nncf_graph = _get_nncf_graph_from_sequential(model)

    return nncf_graph


def _get_nncf_graph_from_functional(model: tf.keras.Model) -> NNCFGraph:
    #pylint:disable=too-many-statements
    #pylint:disable=too-many-branches
    nncf_graph = TFNNCFGraph()
    model_config = model.get_config()
    raw_nodes = {}  # type: Dict[str, List[Dict]]
    for layer in model_config['layers']:
        layer_name = layer['name']
        layer_type = _get_layer_type(layer)
        layer_dtype = _get_layer_dtype(layer)
        data_format = layer['config'].get('data_format')
        model_layer = _get_layer(model, layer_name)
        layer_metatype = get_keras_layer_metatype(model_layer)
        raw_nodes[layer_name] = []

        is_output = False
        # pylint: disable=protected-access
        if model_layer in model._output_layers:
            is_output = True
        if layer['inbound_nodes']:
            is_shared = len(layer['inbound_nodes']) > 1
            for i, inbound_node in enumerate(layer['inbound_nodes']):
                input_shape = _prepare_shape(model_layer.inbound_nodes[i].input_shapes)
                instance = {
                    'type': layer_type,
                    'metatype': layer_metatype,
                    'dtype': layer_dtype,
                    'data_format': data_format,
                    'is_shared': is_shared,
                    'input_shapes': input_shape,
                    'in_ports': list(range(len(inbound_node))),
                    'out_ports': set(),
                    'is_output': is_output,
                    'output_shape': _prepare_shape(model_layer.inbound_nodes[i].output_shapes)
                }
                if layer_metatype in GENERAL_CONV_LAYER_METATYPES:
                    module_attributes = _get_module_attributes(model_layer)
                    instance.update({NNCFGraph.MODULE_ATTRIBUTES: module_attributes})
                for parent_name, parent_instance_index, parent_out_ports, _ in inbound_node:
                    parent_instance = raw_nodes[parent_name][parent_instance_index]
                    if parent_instance['out_ports']:
                        parent_instance['out_ports'].add(parent_out_ports)
                    else:
                        parent_instance['out_ports'] = {parent_out_ports}
                raw_nodes[layer_name].append(instance)
        else:
            input_shapes = _prepare_shape(model_layer.input_shape)
            instance = {
                'type': layer_type,
                'metatype': layer_metatype,
                'dtype': layer_dtype,
                'data_format': data_format,
                'is_shared': False,
                'input_shapes': input_shapes,
                'in_ports': [],
                'out_ports': set(),
                'is_output': is_output,
                'output_shape': _prepare_shape(model_layer.output_shape)
            }
            if layer_metatype in GENERAL_CONV_LAYER_METATYPES:
                module_attributes = _get_module_attributes(model_layer)
                instance.update({NNCFGraph.MODULE_ATTRIBUTES: module_attributes})
            raw_nodes[layer_name].append(instance)

    outputs = model_config['output_layers']

    if isinstance(outputs, list):
        for layer_name, instance, out_port in outputs:
            raw_nodes[layer_name][instance]['out_ports'].add(out_port)
    elif isinstance(outputs, dict):
        for output in outputs.values():
            if isinstance(output, list):
                raw_nodes[output[0]][output[1]]['out_ports'].add(output[2])
            elif isinstance(output, dict):
                for layer_name, instance, out_port in output.values():
                    raw_nodes[layer_name][instance]['out_ports'].add(out_port)

    for instance_dicts in raw_nodes.values():
        for instance in instance_dicts:
            instance['out_ports'] = sorted(list(instance['out_ports']))

    layer_name_vs_nncf_node_ids = {}  # type: Dict[str, List[int]]

    for layer_name, instances in raw_nodes.items():
        layer_name_vs_nncf_node_ids[layer_name] = []
        for i, attributes in enumerate(instances):
            nncf_node = nncf_graph.add_nncf_node(get_expanded_node_name(layer_name, i,
                                                                        attributes['is_shared']),
                                                 node_type=attributes['type'],
                                                 node_metatype=attributes['metatype'],
                                                 module_attributes=attributes.get('module_attributes'),
                                                 containing_layer_name=layer_name)
            layer_name_vs_nncf_node_ids[layer_name].append(nncf_node.node_id)

            if attributes['is_output']:
                # Aligning the structure of auxiliary output nodes is only necessary for NNCFGraph
                output_aux_node_name = PREFIX_AUXILIARY_OUTPUT_NODE + '_{}'.format(i)
                output_node = nncf_graph.add_nncf_node(
                    node_name=output_aux_node_name,
                    node_type=NNCFGraphNodeType.OUTPUT_NODE,
                    node_metatype=OutputNoopMetatype)
                nncf_graph.add_edge_between_nncf_nodes(nncf_node.node_id,
                                                       output_node.node_id,
                                                       tensor_shape=attributes['output_shape'],
                                                       input_port_id=0,
                                                       dtype=Dtype.FLOAT)

    for layer in model_config['layers']:
        layer_name = layer['name']
        for i, inbound_nodes in enumerate(layer['inbound_nodes']):
            node_id = layer_name_vs_nncf_node_ids[layer_name][i]
            for k, inbound_node in enumerate(inbound_nodes):
                producer_name, producer_instance, *_ = inbound_node
                producer_node_id = layer_name_vs_nncf_node_ids[producer_name][producer_instance]
                input_shape = raw_nodes[layer_name][i]['input_shapes'][k]
                producer_node_attrs = raw_nodes[producer_name][producer_instance]
                dtype = Dtype.FLOAT if 'float' in producer_node_attrs['dtype'].lower() else Dtype.INTEGER
                nncf_graph.add_edge_between_nncf_nodes(producer_node_id,
                                                       node_id,
                                                       tensor_shape=input_shape,
                                                       input_port_id=k,
                                                       dtype=dtype)
    return nncf_graph


def _prepare_shape(shape) -> List:
    if not isinstance(shape, list):
        return [shape]
    return shape

def _get_nncf_graph_from_sequential(model: tf.keras.Model) -> NNCFGraph:
    nncf_graph = TFNNCFGraph()
    producer_layer_id = None
    model_config = model.get_config()
    layer_name = None
    for layer in model_config['layers']:
        layer_name = layer['config']['name']
        layer_type = _get_layer_type(layer)
        layer_dtype = _get_layer_dtype(layer)
        data_format = layer['config'].get('data_format')
        model_layer = _get_layer(model, layer_name)
        layer_metatype = get_keras_layer_metatype(model_layer)

        attrs = dict(type=layer_type,
                     dtype=layer_dtype,
                     metatype=layer_metatype,
                     data_format=data_format,
                     in_ports=[0],
                     out_ports=[0],
                     is_shared=False)

        module_attributes = None
        if layer_metatype in GENERAL_CONV_LAYER_METATYPES:
            module_attributes = _get_module_attributes(model.get_layer(layer_name))
            attrs.update({NNCFGraph.MODULE_ATTRIBUTES: module_attributes})

        nncf_node = nncf_graph.add_nncf_node(layer_name,
                                             node_type=layer_type,
                                             node_metatype=layer_metatype,
                                             module_attributes=module_attributes,
                                             containing_layer_name=layer_name)

        if producer_layer_id is not None:
            input_shape = _prepare_shape(model.get_layer(layer_name).input_shape)
            nncf_graph.add_edge_between_nncf_nodes(producer_layer_id,
                                                   nncf_node.node_id,
                                                   tensor_shape=input_shape[0],
                                                   input_port_id=0,
                                                   dtype=Dtype.FLOAT)  # TODO: determine from keras layers
        producer_layer_id = nncf_node.node_id

    if layer_name is not None:
        last_producer_layer_name = layer_name
        last_producer_layer_id = producer_layer_id
        output_model_layer = model.get_layer(last_producer_layer_name)
        output_aux_node_name = PREFIX_AUXILIARY_OUTPUT_NODE + '_0'
        output_node = nncf_graph.add_nncf_node(node_name=output_aux_node_name,
                                               node_type=NNCFGraphNodeType.OUTPUT_NODE,
                                               node_metatype=OutputNoopMetatype)
        nncf_graph.add_edge_between_nncf_nodes(last_producer_layer_id, output_node.node_id,
                                               tensor_shape=_prepare_shape(output_model_layer.output_shape),
                                               input_port_id=0, dtype=Dtype.FLOAT)

    return nncf_graph


def _get_module_attributes(layer: tf.keras.layers.Layer) -> ConvolutionLayerAttributes:
    channel_axis = get_input_channel_axis(layer)
    layer_ = unwrap_layer(layer)
    strides = layer_.strides[0]
    groups = layer_.groups
    kernel_size = layer_.kernel_size

    return ConvolutionLayerAttributes(layer.trainable,
                                      layer.get_input_shape_at(0)[channel_axis],
                                      layer.get_output_shape_at(0)[channel_axis],
                                      kernel_size,
                                      strides,
                                      groups, transpose=False)  # TODO: how to determine transpose here?
