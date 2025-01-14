# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import sys
from typing import List, Tuple

import tensorflow as tf

import nncf
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.tensorflow.graph.metatypes.keras_layers import TFNNCFWrapperLayerMetatype
from nncf.tensorflow.graph.metatypes.matcher import get_keras_layer_metatype
from nncf.tensorflow.layers.operation import NNCFOperation
from nncf.tensorflow.layers.wrapper import NNCFWrapper

SHARED_OPERATION_MARK = "^"


def is_sequential_or_functional_model(model):
    return is_sequential_model(model) or is_functional_model(model)


def is_sequential_model(model):
    return isinstance(model, tf.keras.Sequential)


def is_functional_model(model):
    return (
        isinstance(model, tf.keras.Model)
        and not isinstance(model, tf.keras.Sequential)
        and getattr(model, "_is_graph_network", False)
    )


def is_keras_layer_model(model: tf.keras.Model) -> bool:
    """
    Checks if there is `tensorflow_hub.KerasLayer` layer in the model or not.

    :param model: Keras model to check.
    :return: `True` if there is `hub.KerasLayer` in the model and
        `False` otherwise.
    """
    for layer in model.submodules:
        if layer.__class__.__name__ == "KerasLayer":
            return True
    return False


def get_keras_layers_class_names():
    keras_layers = [
        class_name for class_name, _ in inspect.getmembers(sys.modules[tf.keras.layers.__name__], inspect.isclass)
    ]
    keras_layers += ["TensorFlowOpLayer", "TFOpLambda", "SlicingOpLambda"]
    return keras_layers


def get_keras_activation_names():
    keras_activations = [
        activation_name
        for activation_name, _ in inspect.getmembers(sys.modules[tf.keras.activations.__name__], inspect.isfunction)
    ]
    return keras_activations


def get_custom_objects(model):
    keras_layers = get_keras_layers_class_names()
    keras_activations = get_keras_activation_names()
    custom_objects = {}
    for layer in model.submodules:
        if layer.__class__.__name__ == "NNCFWrapper":
            layer = layer.layer
        if layer.__class__.__name__ not in keras_layers:
            custom_objects[layer.__class__.__name__] = layer.__class__
        if layer.__class__.__name__ == "Activation" and layer.activation.__name__ not in keras_activations:
            custom_objects[layer.activation.__name__] = layer.activation
    return custom_objects


def get_weight_name(name, layer_name=None):
    if layer_name and layer_name in name:
        return name.split(layer_name + "/")[-1]
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


def get_shared_node_name(layer_name: str, instance_idx: int):
    return "{}{}{}".format(layer_name, SHARED_OPERATION_MARK, instance_idx)


def get_original_name_and_instance_idx(node_name: NNCFNodeName):
    result = node_name.split(SHARED_OPERATION_MARK)
    original_name = result[0]
    instance_idx = 0 if len(result) == 1 else int(result[1])
    return original_name, instance_idx


def get_original_name(node_name):
    return get_original_name_and_instance_idx(node_name)[0]


def get_layer_to_graph_nodes_map(model, node_names):
    layer_to_nodes_map = {layer.name: {"type": layer.__class__.__name__, "nodes": []} for layer in model.layers}
    for node in node_names:
        parent_layer_name = node.split("/")[1]  # model_name/layer_name/layer_op_name/...
        if parent_layer_name not in layer_to_nodes_map:
            raise nncf.ValidationError("Could not find {} layer in Model".format(parent_layer_name))
        layer_to_nodes_map[parent_layer_name]["nodes"].append(node)
    return layer_to_nodes_map


def get_weight_node_name(graph: NNCFGraph, node_name: NNCFNodeName) -> NNCFNodeName:
    node = graph.get_node_by_name(node_name)
    while list(graph.get_previous_nodes(node)):
        node = list(graph.get_previous_nodes(node))[-1]
    return node.node_name


def get_layer_identifier(node: NNCFNode):
    layer_name, _ = get_original_name_and_instance_idx(node.node_name)
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


def _was_specially_wrapped_with_keras_export(layer, attr_name) -> bool:
    return hasattr(layer, attr_name) and getattr(layer, attr_name) != ("keras.layers.Layer",)


def is_builtin_layer(layer) -> bool:
    # A similar logic is actually what gets used in TF as
    # tensorflow.python.keras.utils.layer_utils.is_builtin_layer.
    return (
        layer.__class__.__name__ in ["SlicingOpLambda", "TFOpLambda", "TensorFlowOpLayer"]
        or _was_specially_wrapped_with_keras_export(layer, "_keras_api_names")
        or _was_specially_wrapped_with_keras_export(layer, "_keras_api_names_v1")
    )


def get_list_level(lst: List) -> int:
    return isinstance(lst, list) and list(map(get_list_level, lst or [0]))[0] + 1


def check_oplambda_input_data(x: List) -> bool:
    input_stracture = [type(item) for item in x]
    return input_stracture in ([str, int, int], [str, int, int, dict])


def reformat_inbound_nodes_for_oplambda(inbound_nodes: List) -> List[List[List]]:
    """
    Default format examples for inbound_nodes of the layer are as follows:
    1- [[['name', id, id, kwargs]]] - Single input, not shared layer.
    2- [[['name_a', id, id, kwargs], ['name_b', id, id, kwargs]]] - Multiple (two) inputs, not shared layer.
    3- [[['name_a', id, id, kwargs]], [['name_b', id, id, kwargs]]] - Single input, shared layer.

    This function assumes there are no shared layers (aka shared convolutions) among
    'TFOpLambda' or 'SlicingOpLambda' layers.

    'TFOpLambda' or 'SlicingOpLambda' layers, apart from default format templates (1 and 2 above),
    could have the following formats:
    4 - [['name', id, id, kwargs]] - Single input.
    5 - [[[['_CONSTANT_VALUE', -1, serialized_kt, {'y': ['name', id, id]}]]]] -
    Multiple (two) inputs with one constant input.

    :param inbound_nodes: Inbound nodes of the 'TFOpLambda' or 'SlicingOpLambda' layer.
    :return: Reformatted inbound_nodes.
    """
    if get_list_level(inbound_nodes) == 2:  # [[ ]] -> [[[ ]]]
        inbound_nodes = [inbound_nodes]
    if get_list_level(inbound_nodes) == 4:  # [[[[ ]]]] -> [[[ ]]]
        inbound_nodes = inbound_nodes[0]

    inbound_nodes_oplambda = []
    for inbound_node in inbound_nodes[0]:
        if inbound_node[0] != "_CONSTANT_VALUE":
            # If an element in the first call argument did not originate as a keras tensor
            # and is a constant value, it is saved using '_CONSTANT_VALUE'.
            inbound_nodes_oplambda.append(inbound_node)

        # Check for nested inbound nodes in kwargs
        kwargs = inbound_node[3]
        for item in kwargs.values():
            if isinstance(item, list) and check_oplambda_input_data(item):
                if len(item) == 3:
                    item.append({})
                inbound_nodes_oplambda.append(item)

    return [inbound_nodes_oplambda]  # convert to [[[ ]]] format, which is default for inbound nodes
