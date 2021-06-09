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

from collections import OrderedDict
from collections import namedtuple

import tensorflow as tf

from nncf.tensorflow.graph.transformations.layout import TFTransformationLayout
from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationType
from nncf.tensorflow.graph.utils import get_custom_objects
from nncf.tensorflow.graph.utils import get_weight_name
from nncf.tensorflow.graph.utils import is_functional_model
from nncf.tensorflow.graph.utils import is_sequential_or_functional_model
from nncf.tensorflow.layers.custom_objects import get_nncf_custom_objects
from nncf.tensorflow.layers.wrapper import NNCFWrapper

WeightOperations = namedtuple('WeightOperations',
                              ('weights_attr_name', 'operations'))


class TFModelTransformer(ModelTransformer):
    """
    Applies transformations to a Keras model graph.
    """
    def __init__(self, model):
        """
        Initializes Model Transformer

        :param model: Keras model to be transformed
        """
        if not is_sequential_or_functional_model(model):
            raise ValueError(
                'Only tf.keras sequential or functional models can be transformed.')

        super().__init__(model)
        self._model_config = model.get_config()
        self._custom_objects = dict(
            list(get_custom_objects(model).items()) + list(get_nncf_custom_objects().items())
        )
        self._name_mapping = {}

    def transform(self, transformation_layout: TFTransformationLayout):
        """ Applies transformations to the Keras model.

        :param transformation_layout: List of transformations
        :return: The transformed Keras model
        """
        layer_weights_map = {layer.name: self._get_layer_weights(layer) for layer in self._model.layers}

        for transform in transformation_layout.transformations:
            self._apply_transformation(transform)

        if is_functional_model(self._model):
            transformed_model = tf.keras.Model.from_config(self._model_config, self._custom_objects)
        else:
            transformed_model = tf.keras.Sequential.from_config(self._model_config, self._custom_objects)

        for layer in transformed_model.layers:
            weights = layer_weights_map.get(layer.name)
            if weights:
                self._set_layer_weights(layer, weights)

        return transformed_model

    def _get_layer_weights(self, layer):
        weights_map = OrderedDict()
        for weight_tensor, weight_numpy in \
                zip(layer.weights, layer.get_weights()):
            weights_map[get_weight_name(weight_tensor.name, layer.name)] = weight_numpy

        return weights_map

    def _set_layer_weights(self, layer, weights_map):
        weight_value_tuples = []
        for weight_tensor in layer.weights:
            weight_name = get_weight_name(weight_tensor.name, layer.name)
            if weight_name in weights_map:
                weight_value_tuples.append(
                    (weight_tensor, weights_map[weight_name]))

        tf.keras.backend.batch_set_value(weight_value_tuples)

    def _get_layer(self, layer_name):
        for layer in self._model.layers:
            if layer.name == layer_name:
                return layer

        _, layer_config = self._find_layer_config(layer_name)
        if layer_config:
            return tf.keras.utils.deserialize_keras_object(
                layer_config, custom_objects=self._custom_objects)

        return None

    def _find_layer_config(self, layer_name):
        for idx, layer in enumerate(self._model_config['layers']):
            layer_name_ = layer['name'] if is_functional_model(self._model) \
                else layer['config']['name']
            if layer_name_ == layer_name:
                return idx, layer
        return None, None

    def _update_layer_mapping(self, src_layer_name, dst_layer_name):
        if src_layer_name in self._name_mapping.values():
            for orig_layer_name in self._name_mapping:
                if self._name_mapping[orig_layer_name] == src_layer_name:
                    self._name_mapping[orig_layer_name] = dst_layer_name
        else:
            self._name_mapping[src_layer_name] = dst_layer_name

    def _apply_transformation(self, transformation):
        if transformation.type == TransformationType.INSERT:
            self._insert(transformation.target_point, transformation.insertion_objects)
        elif transformation.type == TransformationType.MULTI_INSERT:
            self._multi_insertion(transformation.target_point, transformation.commands)
        elif transformation.type == TransformationType.REMOVE:
            self._remove(transformation.target_point)
        else:
            raise TypeError('Transformation type {} does not support'
                            .format(transformation.type))

    def _get_target_layer_name(self, target_point):
        target_layer_name = target_point.layer_name
        if target_layer_name in self._name_mapping:
            return self._name_mapping[target_layer_name]
        return target_layer_name

    def _insert(self, target_point, insertion_objects):
        target_layer_name = self._get_target_layer_name(target_point)

        if target_point.type == TargetType.OPERATION_WITH_WEIGHTS:
            weight_operations = [
                WeightOperations(target_point.weights_attr_name, insertion_objects)]
            self._insert_weight_operations(target_layer_name, weight_operations)
        elif target_point.type == TargetType.AFTER_LAYER:
            self._insert_layers_after(target_layer_name, target_point.instance_index,
                                      target_point.out_port, insertion_objects)
        else:
            raise TypeError('Insertion transform does not support {} '
                            'target point type'.format(target_point.type))

    def _multi_insertion(self, target_point, commands):
        if target_point.type != TargetType.LAYER:
            raise TypeError('Multiple insertion transform does not support '
                            '{} target point type'.format(target_point.type))

        target_layer_name = self._get_target_layer_name(target_point)

        weight_operations = []
        for cmd in commands:
            if cmd.type != TransformationType.INSERT or \
                    cmd.target_point.type != TargetType.OPERATION_WITH_WEIGHTS:
                raise TypeError('Multiple insertion transform does not support command: '
                                'command type - {}; target point type - {}'
                                .format(cmd.type, cmd.target_point.type))
            weight_operations.append(
                WeightOperations(
                    cmd.target_point.weights_attr_name,
                    cmd.insertion_objects
                ))

        self._insert_weight_operations(target_layer_name, weight_operations)

    def _remove(self, target_point):
        target_layer_name = self._get_target_layer_name(target_point)

        if target_point.type == TargetType.OPERATION_WITH_WEIGHTS:
            self._remove_weight_operation(
                target_layer_name,
                target_point.weights_attr_name,
                target_point.operation_name)
        else:
            raise TypeError('{} removal does not support'.format(target_point.type))

    def _remove_weight_operation(self, layer_name, weights_attr_name, operation_name):
        _, layer_config = self._find_layer_config(layer_name)
        weights_operations = layer_config['config']['weights_attr_operations'].pop(weights_attr_name)

        def find_weights_operation(operations, name):
            for op in operations:
                if op['config']['name'] == name:
                    return op
            return None

        found = find_weights_operation(weights_operations, operation_name)
        weights_operations.remove(found)

        if weights_operations:
            layer_config['config']['weights_attr_operations'][weights_attr_name] = weights_operations
        elif not layer_config['config']['weights_attr_operations']:
            self._replace_config(layer_name, layer_config['config']['layer'])

    def _insert_weight_operations(self, layer_name, weight_operations):
        layer = self._get_layer(layer_name)
        wrapper = layer if isinstance(layer, NNCFWrapper) else NNCFWrapper(layer)

        for weights_attr_name, operations in weight_operations:
            for op in operations:
                wrapper.registry_weight_operation(weights_attr_name, op)

        self._replace(layer_name, wrapper)

    def _replace(self, layer_name, replace_layer):
        replace_layer_config = tf.keras.utils.serialize_keras_object(replace_layer)
        self._replace_config(layer_name, replace_layer_config)

    def _replace_config(self, layer_name, replace_layer_config):
        replace_layer_name = replace_layer_config['config']['name']
        if is_functional_model(self._model):
            if 'name' not in replace_layer_config:
                replace_layer_config['name'] = replace_layer_name
            self._replace_functional(layer_name, replace_layer_config)
        else:
            self._replace_sequential(layer_name, replace_layer_config)

        self._update_layer_mapping(layer_name, replace_layer_name)

    def _replace_functional(self, layer_name, replace_layer_config):
        replace_layer_name = replace_layer_config['name']
        for layer in self._model_config['layers']:
            for inbound_node in layer['inbound_nodes']:
                self._process_replacement(inbound_node, layer_name, replace_layer_name)

        self._replace_in_model_outputs(layer_name, replace_layer_name)

        idx, layer_config = self._find_layer_config(layer_name)
        replace_layer_config['inbound_nodes'] = layer_config['inbound_nodes']
        self._model_config['layers'][idx] = replace_layer_config

    def _replace_sequential(self, layer_name, replace_layer_config):
        idx, _ = self._find_layer_config(layer_name)
        self._model_config['layers'][idx] = replace_layer_config

    def _insert_layers_after(self, layer_name, instance_index, out_port, layers):
        functional_model = is_functional_model(self._model)

        layer_configs = []
        for layer in layers:
            config = tf.keras.utils.serialize_keras_object(layer)
            if functional_model:
                config['name'] = config['config']['name']
                config['inbound_nodes'] = [[[layer_name, instance_index, out_port, {}]]]
            layer_configs.append(config)

        for config in layer_configs:
            if functional_model:
                self._insert_layer_after_functional(layer_name, instance_index, config)
            else:
                self._insert_layer_after_sequential(layer_name, config)

    def _insert_layer_after_functional(self, layer_name, instance_index, layer_config):
        layer_out_ports = set()
        replace_layer_name = layer_config['name']
        for layer in self._model_config['layers']:
            for inbound_node in layer['inbound_nodes']:
                self._process_insertion_after(inbound_node, layer_name, instance_index,
                                              layer_out_ports, replace_layer_name)

        self._insert_after_model_outputs(layer_name, instance_index, layer_out_ports, replace_layer_name)
        if len(layer_out_ports) > 1:
            raise RuntimeError('Insertion after layer ({}) with multiple ports '
                               'is not supported'.format(layer_name))

        self._insert_layer_after_sequential(layer_name, layer_config)

    def _insert_layer_after_sequential(self, layer_name, layer_configs):
        idx, _ = self._find_layer_config(layer_name)
        self._model_config['layers'].insert(idx + 1, layer_configs)

    @staticmethod
    def _process_insertion_after(connection_infos, layer_name, instance_index, layer_out_ports, replace_layer_name):
        for connection_info in connection_infos:
            if connection_info[0] == layer_name:
                layer_out_ports.add(connection_info[2])
                if connection_info[1] == instance_index:
                    connection_info[0] = replace_layer_name
                    connection_info[1] = 0

    def _insert_after_model_outputs(self, layer_name, instance_index, layer_out_ports, replace_layer_name):
        output_layers = self._model_config['output_layers']
        if isinstance(output_layers, list):
            self._process_insertion_after(output_layers, layer_name, instance_index,
                                          layer_out_ports, replace_layer_name)
        elif isinstance(output_layers, dict):
            for out_layers in output_layers.values():
                if isinstance(out_layers, list):
                    self._process_insertion_after([out_layers], layer_name, instance_index,
                                                  layer_out_ports, replace_layer_name)
                elif isinstance(out_layers, dict):
                    self._process_insertion_after(out_layers.values(), layer_name, instance_index,
                                                  layer_out_ports, replace_layer_name)

    @staticmethod
    def _process_replacement(connection_infos, layer_name, replace_layer_name):
        for connection_info in connection_infos:
            if connection_info[0] == layer_name:
                connection_info[0] = replace_layer_name

    def _replace_in_model_outputs(self, layer_name, replace_layer_name):
        output_layers = self._model_config['output_layers']
        if isinstance(output_layers, list):
            self._process_replacement(output_layers, layer_name, replace_layer_name)
        elif isinstance(output_layers, dict):
            for out_layers in output_layers.values():
                if isinstance(out_layers, list):
                    self._process_replacement([out_layers], layer_name, replace_layer_name)
                elif isinstance(out_layers, dict):
                    self._process_replacement(out_layers.values(), layer_name, replace_layer_name)
