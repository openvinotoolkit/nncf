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

from typing import List, Tuple
import sys
import inspect

import tensorflow as tf

from nncf.tensorflow.graph.metatypes.keras_layers import TFNNCFWrapperLayerMetatype
from nncf.tensorflow.graph.metatypes.matcher import get_keras_layer_metatype
from nncf.tensorflow.layers.wrapper import NNCFWrapper
from nncf.tensorflow.layers.operation import NNCFOperation
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName

SHARED_OPERATION_MARK = '^'


def is_sequential_or_functional_model(model):
    return is_sequential_model(model) or is_functional_model(model)


def is_sequential_model(model):
    return isinstance(model, tf.keras.Sequential)


def is_functional_model(model):
    return isinstance(model, tf.keras.Model) \
           and not isinstance(model, tf.keras.Sequential) \
           and getattr(model, '_is_graph_network', False)


def get_keras_layers_class_names():
    keras_layers = [class_name for class_name, _ in
                    inspect.getmembers(sys.modules[tf.keras.layers.__name__], inspect.isclass)]
    keras_layers += ['TensorFlowOpLayer']
    return keras_layers


def get_keras_activation_names():
    keras_activations = [activation_name for activation_name, _ in
                    inspect.getmembers(sys.modules[tf.keras.activations.__name__], inspect.isfunction)]
    return keras_activations


def get_custom_objects(model):
    keras_layers = get_keras_layers_class_names()
    keras_activations = get_keras_activation_names()
    custom_objects = {}
    for layer in model.submodules:
        if layer.__class__.__name__ == 'NNCFWrapper':
            layer = layer.layer
        if layer.__class__.__name__ not in keras_layers:
            custom_objects[layer.__class__.__name__] = layer.__class__
        if layer.__class__.__name__ == 'Activation':
            if layer.activation.__name__ not in keras_activations:
                custom_objects[layer.activation.__name__] = layer.activation
    return custom_objects


def get_custom_layers(model):
    keras_layers = get_keras_layers_class_names()
    custom_layers = []
    for layer in model.submodules:
        if layer.__class__.__name__ not in keras_layers:
            custom_layers.append(layer)
    return custom_layers


def get_weight_name(name, layer_name=None):
    if layer_name and layer_name in name:
        return name.split(layer_name + '/')[-1]
    return name


def get_weight_by_name(layer, weight_name):
    for w in layer.weights:
        if w.name.split(":")[0] == weight_name:
            return w
    return None


def collect_wrapped_layers(model):
    wrapped_layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            wrapped_layers += collect_wrapped_layers(layer)
        if isinstance(layer, NNCFWrapper):
            wrapped_layers.append(layer)
    return wrapped_layers


def get_shared_node_name(layer_name: str, instance_index: int):
    return '{}{}{}'.format(layer_name, SHARED_OPERATION_MARK, instance_index)


def get_original_name_and_instance_index(node_name):
    result = node_name.split(SHARED_OPERATION_MARK)
    original_name = result[0]
    instance_index = 0 if len(result) == 1 else int(result[1])
    return original_name, instance_index


def get_original_name(node_name):
    return get_original_name_and_instance_index(node_name)[0]


def get_layer_to_graph_nodes_map(model, node_names):
    layer_to_nodes_map = {layer.name: {'type': layer.__class__.__name__,
                                       'nodes': []}
                          for layer in model.layers}
    for node in node_names:
        parent_layer_name = node.split('/')[1]  # model_name/layer_name/layer_op_name/...
        if parent_layer_name not in layer_to_nodes_map:
            raise RuntimeError('Could not find {} layer in Model'.format(parent_layer_name))
        layer_to_nodes_map[parent_layer_name]['nodes'].append(node)
    return layer_to_nodes_map


def get_weight_node_name(graph: NNCFGraph, node_name: NNCFNodeName) -> NNCFNodeName:
    node = graph.get_node_by_name(node_name)
    while list(graph.get_previous_nodes(node)):
        node = list(graph.get_previous_nodes(node))[-1]
    return node.node_name


def get_layer_identifier(node: NNCFNode):
    layer_name, _ = get_original_name_and_instance_index(node.node_name)
    return layer_name


def unwrap_layer(layer):
    layer_metatype = get_keras_layer_metatype(layer, determine_subtype=False)
    if layer_metatype == TFNNCFWrapperLayerMetatype:
        return layer.layer
    return layer


def get_nncf_operations(model: tf.keras.Model, operation_names: List[str]) -> Tuple[NNCFWrapper, str, NNCFOperation]:
    """
    Yields the operations from the model which names in `operation_names`.

    :param model: Wrapped model.
    :param operation_names: List of operation names.
    :return: A tuple (wrapped_layer, weight_attr, op) where
        - wrapped_layer: A wrapped layer, which contains operation weights.
        - weight_attr: A name of the attribute of the wrapped layer to which
            the operation is applied.
        - op: NNCF operation.
    """
    for wrapped_layer in collect_wrapped_layers(model):
        for weight_attr, ops in wrapped_layer.weights_attr_ops.items():
            for op in ops.values():
                if op.name in operation_names:
                    yield wrapped_layer, weight_attr, op
