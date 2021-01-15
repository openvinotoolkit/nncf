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
import networkx as nx
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from nncf.tensorflow.graph.utils import get_expanded_node_name
from nncf.tensorflow.graph.utils import is_functional_model
from nncf.tensorflow.graph.utils import is_sequential_model


def convert_keras_model_to_nxmodel(model):
    """
    Convert Keras model graph to the NetworkX directed graph

    :param model: Keras model
    :return: NetworkX directed graph
    """
    func_model = is_functional_model(model)
    seq_model = is_sequential_model(model)

    if not func_model and not seq_model:
        RuntimeError('convert_keras_model_to_nxmodel function supports '
                     'only sequential or functional models')

    if func_model:
        nxmodel = _get_nxmodel_from_functional(model)
    else:
        nxmodel = _get_nxmodel_from_sequential(model)

    #nx.drawing.nx_pydot.write_dot(nxmodel, str("nxmodel_graph.dot"))

    return nxmodel


def _get_layer_type(layer):
    if layer['class_name'] == 'TensorFlowOpLayer':
        return layer['config']['node_def']['op']
    return layer['class_name']


def _get_layer_dtype(layer):
    dtype = layer['config']['dtype']
    if layer['class_name'] == 'TensorFlowOpLayer':
        dtype = layer['config']['node_def'].get('attr', {}).get('T', {}).get('type') or dtype
    return dtype


def _get_nxmodel_from_functional(model):
    raw_nodes = Dict()
    model_config = model.get_config()
    for layer in model_config['layers']:
        layer_name = layer['name']
        layer_type = _get_layer_type(layer)
        layer_dtype = _get_layer_dtype(layer)
        data_format = layer['config'].get('data_format')
        if layer['inbound_nodes']:
            is_shared = len(layer['inbound_nodes']) > 1
            for i, inbound_node in enumerate(layer['inbound_nodes']):
                instance = raw_nodes[layer_name][i]
                instance['type'] = layer_type
                instance['dtype'] = layer_dtype
                instance['data_format'] = data_format
                instance['is_shared'] = is_shared
                instance['in_ports'] = list(range(len(inbound_node)))
                if not instance['out_ports']:
                    instance['out_ports'] = set()
                for parent_name, parent_instance_index, parent_out_ports, _ in inbound_node:
                    parent_instance = raw_nodes[parent_name][parent_instance_index]
                    if parent_instance['out_ports']:
                        parent_instance['out_ports'].add(parent_out_ports)
                    else:
                        parent_instance['out_ports'] = {parent_out_ports}
        else:
            instance = raw_nodes[layer_name][0]
            instance['type'] = layer_type
            instance['dtype'] = layer_dtype
            instance['data_format'] = data_format
            instance['is_shared'] = False
            instance['in_ports'] = []

    outputs = model_config['output_layers']
    raw_nodes = _process_outputs(outputs, raw_nodes)

    for instance_dict in raw_nodes.values():
        for instance in instance_dict.values():
            instance['out_ports'] = sorted(list(instance['out_ports']))

    return _get_nxmodel_from_raw_nodes(model_config, raw_nodes)


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


def _get_nxmodel_from_raw_nodes(model_config, raw_nodes):
    nxmodel = nx.DiGraph()
    for original_name, instances in raw_nodes.items():
        for i, attributes in instances.items():
            node_name = get_expanded_node_name(original_name, i, attributes['is_shared'])
            nxmodel.add_node(node_name, **attributes)

    for layer in model_config['layers']:
        layer_name = layer['name']
        for i, inbound_nodes in enumerate(layer['inbound_nodes']):
            is_shared = raw_nodes[layer_name][i]['is_shared']
            node_full_name = get_expanded_node_name(layer_name, i, is_shared=is_shared)
            for producer_name, producer_instance, *_ in inbound_nodes:
                is_shared = raw_nodes[producer_name][producer_instance]['is_shared']
                producer_full_name = get_expanded_node_name(producer_name, producer_instance, is_shared=is_shared)
                nxmodel.add_edge(producer_full_name, node_full_name)

    return nxmodel


def _get_nxmodel_from_sequential(model):
    nxmodel = nx.DiGraph()
    producer_layer = None
    model_config = model.get_config()
    for layer in model_config['layers']:
        layer_name = layer['config']['name']
        layer_type = _get_layer_type(layer)
        layer_dtype = _get_layer_dtype(layer)
        data_format = layer['config'].get('data_format')
        nxmodel.add_node(layer_name,
                         type=layer_type,
                         dtype=layer_dtype,
                         data_format=data_format,
                         in_ports=[0],
                         out_ports=[0],
                         is_shared=False)
        if producer_layer is not None:
            nxmodel.add_edge(producer_layer, layer_name)
        producer_layer = layer_name

    return nxmodel


def convert_layer_graph_to_nxmodel(layer, use_graph_var_names=False):
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

    nxmodel = nx.DiGraph()
    for node in nodes:
        nxmodel.add_node(graph_to_layer_names_map.get(node.name, node.name),
                         type=node.op, dtype=node.attr['dtype'])

    for node in nodes:
        for input_node in node.input:
            node_name = graph_to_layer_names_map.get(node.name, node.name)
            input_node_name = graph_to_layer_names_map.get(input_node, input_node)
            if input_node_name in nxmodel.nodes:
                nxmodel.add_edge(input_node_name, node_name)

    return nxmodel
