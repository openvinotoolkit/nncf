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

from addict import Dict
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from beta.nncf.tensorflow.graph.metatypes.common import GENERAL_CONV_LAYER_METATYPES
from beta.nncf.tensorflow.graph.metatypes.matcher import get_keras_layer_metatype
from beta.nncf.tensorflow.graph.metatypes.matcher import get_op_metatype
from beta.nncf.tensorflow.graph.metatypes.nncf_op import OutputNoopMetatype
from beta.nncf.tensorflow.graph.utils import get_expanded_node_name
from beta.nncf.tensorflow.graph.utils import is_functional_model
from beta.nncf.tensorflow.graph.utils import is_sequential_model
from beta.nncf.tensorflow.graph.utils import unwrap_layer
from beta.nncf.tensorflow.layers.data_layout import get_input_channel_axis
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFGraphNodeType
from nncf.common.graph.module_attributes import ConvolutionModuleAttributes

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
    model_config = model.get_config()
    raw_nodes = _prepare_raw_nodes(model)
    return _get_nncf_graph_from_raw_nodes(model_config, raw_nodes)


def _prepare_shape(shape):
    if not isinstance(shape, list):
        return [shape]
    return shape


def _prepare_raw_nodes(model: tf.keras.Model) -> Dict:
    model_config = model.get_config()
    raw_nodes = Dict()
    for layer in model_config['layers']:
        layer_name = layer['name']
        layer_type = _get_layer_type(layer)
        layer_dtype = _get_layer_dtype(layer)
        data_format = layer['config'].get('data_format')
        model_layer = _get_layer(model, layer_name)
        layer_metatype = get_keras_layer_metatype(model_layer)

        is_output = False
        # pylint: disable=protected-access
        if model_layer in model._output_layers:
            is_output = True
        if layer['inbound_nodes']:
            is_shared = len(layer['inbound_nodes']) > 1
            for i, inbound_node in enumerate(layer['inbound_nodes']):
                input_shape = _prepare_shape(model_layer.inbound_nodes[i].input_shapes)
                instance = raw_nodes[layer_name][i]
                instance['type'] = layer_type
                instance['metatype'] = layer_metatype
                instance['dtype'] = layer_dtype
                instance['data_format'] = data_format
                instance['is_shared'] = is_shared
                instance['input_shape'] = input_shape
                instance['in_ports'] = list(range(len(inbound_node)))
                if not instance['out_ports']:
                    instance['out_ports'] = set()
                if layer_metatype in GENERAL_CONV_LAYER_METATYPES:
                    module_attributes = _get_module_attributes(model_layer)
                    instance.update({NNCFGraph.MODULE_ATTRIBUTES: module_attributes})
                for parent_name, parent_instance_index, parent_out_ports, _ in inbound_node:
                    parent_instance = raw_nodes[parent_name][parent_instance_index]
                    if parent_instance['out_ports']:
                        parent_instance['out_ports'].add(parent_out_ports)
                    else:
                        parent_instance['out_ports'] = {parent_out_ports}
                instance['is_output'] = is_output
                instance['output_shape'] = _prepare_shape(model_layer.inbound_nodes[i].output_shapes)
        else:
            instance = raw_nodes[layer_name][0]
            instance['type'] = layer_type
            instance['dtype'] = layer_dtype
            instance['metatype'] = layer_metatype
            instance['data_format'] = data_format
            instance['is_shared'] = False
            instance['in_ports'] = []
            instance['input_shape'] = _prepare_shape(model_layer.input_shape)
            if layer_metatype in GENERAL_CONV_LAYER_METATYPES:
                module_attributes = _get_module_attributes(model_layer)
                instance.update({NNCFGraph.MODULE_ATTRIBUTES: module_attributes})
            instance['is_output'] = is_output
            instance['output_shape'] = _prepare_shape(model_layer.output_shape)

    outputs = model_config['output_layers']
    raw_nodes = _process_outputs(outputs, raw_nodes)

    for instance_dict in raw_nodes.values():
        for instance in instance_dict.values():
            instance['out_ports'] = sorted(list(instance['out_ports']))

    return raw_nodes


def _get_nncf_graph_from_raw_nodes(model_config: dict, raw_nodes: Dict) -> NNCFGraph:
    nncf_graph = NNCFGraph()
    nncf_graph = _update_graph_with_raw_nodes(nncf_graph, raw_nodes, model_config)
    return nncf_graph


def _update_graph_with_raw_nodes(graph: NNCFGraph,
                                 raw_nodes: Dict,
                                 model_config: dict) -> NNCFGraph:
    for original_name, instances in raw_nodes.items():
        for i, attributes in instances.items():
            node_name = get_expanded_node_name(original_name, i, attributes['is_shared'])
            graph.add_node(node_name, original_name=original_name, **attributes)

            if attributes['is_output']:
                # Aligning the structure of auxiliary output nodes is only necessary for NNCFGraph
                output_aux_node_name = PREFIX_AUXILIARY_OUTPUT_NODE + '_{}'.format(i)
                node_attrs = {
                    NNCFGraph.NODE_TYPE_ATTR: NNCFGraphNodeType.OUTPUT_NODE,
                    NNCFGraph.METATYPE_ATTR: OutputNoopMetatype,
                    'original_name': output_aux_node_name
                }
                edge_attrs = {
                    NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR: attributes['output_shape'],
                    NNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0,
                }
                graph.add_node(output_aux_node_name, **node_attrs)
                graph.add_edge(node_name, output_aux_node_name, **edge_attrs)

    for layer in model_config['layers']:
        layer_name = layer['name']
        for i, inbound_nodes in enumerate(layer['inbound_nodes']):
            is_shared = raw_nodes[layer_name][i]['is_shared']
            node_full_name = get_expanded_node_name(layer_name, i, is_shared=is_shared)
            for k, inbound_node in enumerate(inbound_nodes):
                producer_name, producer_instance, *_ = inbound_node
                is_shared = raw_nodes[producer_name][producer_instance]['is_shared']
                producer_full_name = get_expanded_node_name(producer_name, producer_instance, is_shared=is_shared)
                attr = {
                    NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR: raw_nodes[layer_name][i]['input_shape'][k],
                    NNCFGraph.IN_PORT_NAME_EDGE_ATTR: k
                }
                graph.add_edge(producer_full_name, node_full_name, **attr)
    return graph


def _get_nncf_graph_from_sequential(model: tf.keras.Model) -> NNCFGraph:
    nncf_graph = NNCFGraph()
    producer_layer = None
    model_config = model.get_config()
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
        if layer_metatype in GENERAL_CONV_LAYER_METATYPES:
            module_attributes = _get_module_attributes(model_layer)
            attrs.update({NNCFGraph.MODULE_ATTRIBUTES: module_attributes})

        nncf_graph.add_node(layer_name, **attrs)
        if producer_layer is not None:
            input_shape = _prepare_shape(_get_layer(model, layer_name).input_shape)
            attr = {
                NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR: input_shape[0],
                NNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0
            }
            nncf_graph.add_edge(producer_layer, layer_name, **attr)
        producer_layer = layer_name

    output_model_layer = model.get_layer(producer_layer)
    output_aux_node_name = PREFIX_AUXILIARY_OUTPUT_NODE + '_0'
    node_attrs = {
        NNCFGraph.NODE_TYPE_ATTR: NNCFGraphNodeType.OUTPUT_NODE,
        NNCFGraph.METATYPE_ATTR: OutputNoopMetatype,
        'original_name': output_aux_node_name
    }
    edge_attrs = {
        NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR: _prepare_shape(output_model_layer.output_shape),
        NNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0,
    }
    nncf_graph.add_node(output_aux_node_name, **node_attrs)
    nncf_graph.add_edge(producer_layer, output_aux_node_name, **edge_attrs)

    return nncf_graph


def _get_module_attributes(layer: tf.keras.layers.Layer) -> ConvolutionModuleAttributes:
    channel_axis = get_input_channel_axis(layer)
    layer_ = unwrap_layer(layer)
    strides = layer_.strides[0]
    groups = layer_.groups
    kernel_size = layer_.kernel_size

    return ConvolutionModuleAttributes(layer.trainable,
                                       layer.get_input_shape_at(0)[channel_axis],
                                       layer.get_output_shape_at(0)[channel_axis],
                                       kernel_size,
                                       strides,
                                       groups)
