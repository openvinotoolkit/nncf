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
import copy
from collections import OrderedDict
from collections import namedtuple
from typing import Callable, Dict, List, Set, TypeVar, Union

import tensorflow as tf

import nncf
from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationType
from nncf.tensorflow.graph.transformations.commands import TFAfterLayer
from nncf.tensorflow.graph.transformations.commands import TFBeforeLayer
from nncf.tensorflow.graph.transformations.commands import TFLayer
from nncf.tensorflow.graph.transformations.commands import TFLayerWeight
from nncf.tensorflow.graph.transformations.commands import TFMultiLayerPoint
from nncf.tensorflow.graph.transformations.commands import TFOperationWithWeights
from nncf.tensorflow.graph.transformations.layout import TFTransformationLayout
from nncf.tensorflow.graph.utils import get_custom_objects
from nncf.tensorflow.graph.utils import get_weight_name
from nncf.tensorflow.graph.utils import is_functional_model
from nncf.tensorflow.graph.utils import is_sequential_or_functional_model
from nncf.tensorflow.graph.utils import reformat_inbound_nodes_for_oplambda
from nncf.tensorflow.layers.custom_objects import get_nncf_custom_objects
from nncf.tensorflow.layers.wrapper import NNCFWrapper

WeightOperations = namedtuple("WeightOperations", ("weights_attr_name", "operations"))
TModel = TypeVar("TModel")


class TFModelTransformer(ModelTransformer):
    """
    Applies transformations to a Keras model graph.
    """

    def __init__(self, model: TModel) -> None:
        """
        Initializes Model Transformer

        :param model: Keras model to be transformed
        """
        if not is_sequential_or_functional_model(model):
            raise ValueError("Only tf.keras sequential or functional models can be transformed.")

        super().__init__(model)
        self._model_config = model.get_config()
        self._custom_objects = dict(list(get_custom_objects(model).items()) + list(get_nncf_custom_objects().items()))
        self._name_mapping = {}

    def transform(self, transformation_layout: TFTransformationLayout):
        """Applies transformations to the Keras model.

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
        for weight_tensor, weight_numpy in zip(layer.weights, layer.get_weights()):
            weights_map[get_weight_name(weight_tensor.name, layer.name)] = weight_numpy

        return weights_map

    @staticmethod
    def _set_layer_weights(layer, weights_map: Dict):
        weight_value_tuples = []
        for weight_tensor in layer.weights:
            weight_name = get_weight_name(weight_tensor.name, layer.name)
            if weight_name in weights_map:
                weight_value_tuples.append((weight_tensor, weights_map[weight_name]))

        tf.keras.backend.batch_set_value(weight_value_tuples)

    def _get_layer(self, layer_name: str):
        # For purposes of performance, will try to get the original unmodified layer
        # first. This function could have been made to return the modified layer state by
        # first trying to deserialize the corresponding name from the config (which is potentially
        # compression-modified at this stage), but then the time required to process e.g. all of the weight
        # insertions in complex custom layers becomes prohibitive since for each weight adjustment in the same layer
        # we would have to deserialize the layer object.
        for layer in self._model.layers:
            if layer.name == layer_name:
                return layer

        _, layer_config = self._find_layer_config(layer_name)
        if layer_config:
            return tf.keras.utils.deserialize_keras_object(layer_config, custom_objects=self._custom_objects)

        return None

    def _find_layer_config(self, layer_name: str):
        for idx, layer in enumerate(self._model_config["layers"]):
            layer_name_ = layer["name"] if is_functional_model(self._model) else layer["config"]["name"]
            if layer_name_ == layer_name:
                return idx, layer
        return None, None

    def _update_layer_mapping(self, src_layer_name: str, dst_layer_name: str):
        if src_layer_name in self._name_mapping.values():
            for orig_layer_name, orig_layer in self._name_mapping.items():
                if orig_layer == src_layer_name:
                    self._name_mapping[orig_layer_name] = dst_layer_name
        else:
            self._name_mapping[src_layer_name] = dst_layer_name

    def _apply_transformation(self, transformation: TransformationCommand):
        if transformation.type == TransformationType.INSERT:
            self._insert(transformation.target_point, transformation.insertion_objects)
        elif transformation.type == TransformationType.MULTI_INSERT:
            self._multi_insertion(transformation.target_point, transformation.commands)
        elif transformation.type == TransformationType.REMOVE:
            self._remove(transformation.target_point)
        else:
            raise TypeError("Transformation type {} does not support".format(transformation.type))

    def _insert(self, target_point: Union[TargetPoint, TFMultiLayerPoint], insertion_objects: List[Callable]):
        if isinstance(target_point, TFMultiLayerPoint):
            self._shared_insert_layers(target_point.target_points, insertion_objects)
        elif isinstance(target_point, TFLayerWeight):
            weight_operations = [WeightOperations(target_point.weights_attr_name, insertion_objects)]
            self._insert_weight_operations(target_point.layer_name, weight_operations)
        elif isinstance(target_point, TFBeforeLayer):
            self._insert_layers_before(
                target_point.layer_name, target_point.instance_idx, target_point.input_port_id, insertion_objects
            )
        elif isinstance(target_point, TFAfterLayer):
            self._insert_layers_after(
                target_point.layer_name, target_point.instance_idx, target_point.output_port_id, insertion_objects
            )
        else:
            raise TypeError("Insertion transform does not support {} target point type".format(target_point.type))

    def _shared_insert_layers(self, target_points: List[TargetPoint], layers_to_insert: List[Callable]):
        functional_model = is_functional_model(self._model)
        if functional_model:
            for layer in self._model_config["input_layers"]:
                for tp in target_points:
                    if isinstance(tp, TFBeforeLayer) and tp.layer_name == layer[0]:
                        raise nncf.InternalError(f"Insertion before input layer: {tp.layer_name} is not supported")

        layer_configs = []
        for layer in layers_to_insert:
            config = tf.keras.utils.serialize_keras_object(layer)
            if functional_model:
                config["name"] = config["config"]["name"]
                config["inbound_nodes"] = []
                for i, tp in enumerate(target_points):
                    if isinstance(tp, TFAfterLayer):
                        config["inbound_nodes"].append([[tp.layer_name, tp.instance_idx, tp.output_port_id, {}]])
                    elif isinstance(tp, TFBeforeLayer):
                        idx, input_layer_cfg = self._find_layer_config(tp.layer_name)
                        inbound = input_layer_cfg["inbound_nodes"][tp.instance_idx][tp.input_port_id]
                        config["inbound_nodes"].append([[inbound[0], inbound[1], inbound[2], {}]])
                        self._model_config["layers"][idx]["inbound_nodes"][tp.instance_idx][tp.input_port_id] = [
                            config["name"],
                            i,
                            0,
                            inbound[3],
                        ]
                    else:
                        raise TypeError(
                            f"Insertion transform does not support {target_points[0].type} target point type"
                        )

            layer_configs.append(config)

        for config in layer_configs:
            for i, tp in enumerate(target_points):
                if functional_model and isinstance(tp, TFAfterLayer):
                    layer_out_ports = set()
                    replace_layer_name = config["name"]
                    for layer in self._model_config["layers"]:
                        inbound_nodes = layer["inbound_nodes"]
                        if layer["class_name"] in ["TFOpLambda", "SlicingOpLambda"]:
                            inbound_nodes = reformat_inbound_nodes_for_oplambda(inbound_nodes)

                        for inbound_node in inbound_nodes:
                            self._process_insertion_after(
                                inbound_node, tp.layer_name, tp.instance_idx, layer_out_ports, replace_layer_name, i
                            )

                    self._insert_after_model_outputs(
                        tp.layer_name, tp.instance_idx, layer_out_ports, replace_layer_name, i
                    )
                    if len(layer_out_ports) > 1:
                        raise nncf.InternalError(
                            "Insertion after layer ({}) with multiple ports is not supported".format(tp.layer_name)
                        )

            layer_name = target_points[0].layer_name
            self._insert_layer_after_sequential(layer_name, config)

    def _multi_insertion(self, target_point: TargetPoint, commands: List[TransformationCommand]):
        if not isinstance(target_point, TFLayer):
            raise TypeError(
                "Multiple insertion transform does not support {} target point type".format(target_point.type)
            )

        weight_operations = []
        for cmd in commands:
            if cmd.type != TransformationType.INSERT or cmd.target_point.type != TargetType.OPERATION_WITH_WEIGHTS:
                raise TypeError(
                    "Multiple insertion transform does not support command: "
                    "command type - {}; target point type - {}".format(cmd.type, cmd.target_point.type)
                )
            weight_operations.append(WeightOperations(cmd.target_point.weights_attr_name, cmd.insertion_objects))

        self._insert_weight_operations(target_point.layer_name, weight_operations)

    def _remove(self, target_point: TargetPoint):
        if isinstance(target_point, TFOperationWithWeights):
            target_layer_name = target_point.layer_name
            self._remove_weight_operation(
                target_layer_name, target_point.weights_attr_name, target_point.operation_name
            )
        else:
            raise TypeError("{} removal does not support".format(target_point.type))

    def _remove_weight_operation(self, layer_name: str, weights_attr_name: str, operation_name: str):
        _, layer_config = self._find_layer_config(layer_name)
        weights_operations = layer_config["config"]["weights_attr_operations"].pop(weights_attr_name)

        def find_weights_operation(operations, name):
            for op in operations:
                if op["config"]["name"] == name:
                    return op
            return None

        found = find_weights_operation(weights_operations, operation_name)
        weights_operations.remove(found)

        if weights_operations:
            layer_config["config"]["weights_attr_operations"][weights_attr_name] = weights_operations
        elif not layer_config["config"]["weights_attr_operations"]:
            self._replace_config(layer_name, layer_config["config"]["layer"])

    def _insert_weight_operations(self, layer_name: str, weight_operations: List[WeightOperations]):
        """
        Assigns the operations to be executed with a layer weight while accessing it.
        Any operation that have already been associated with the weight will be replaced with the
        new operation(s).
        :param layer_name: The name of the weighted layer.
        :param weight_operations: The operations to be executed with the weight
        """
        layer = self._get_layer(layer_name)
        wrapper = layer if isinstance(layer, NNCFWrapper) else NNCFWrapper(layer)

        for weights_attr_name, operations in weight_operations:
            for op in operations:
                wrapper.registry_weight_operation(weights_attr_name, op)

        self._replace(layer_name, wrapper)

    def _replace(self, layer_name: str, replace_layer):
        # NOTE: the change to the layer will only get propagated to self._model.layers
        # once the model is deserialized from config, i.e. not immediately.
        replace_layer_config = tf.keras.utils.serialize_keras_object(replace_layer)
        self._replace_config(layer_name, replace_layer_config)

    def _replace_config(self, layer_name: str, replace_layer_config: Dict):
        replace_layer_name = replace_layer_config["config"]["name"]
        if is_functional_model(self._model):
            if "name" not in replace_layer_config:
                replace_layer_config["name"] = replace_layer_name
            self._replace_functional(layer_name, replace_layer_config)
        else:
            self._replace_sequential(layer_name, replace_layer_config)

        self._update_layer_mapping(layer_name, replace_layer_name)

    def _replace_functional(self, layer_name: str, replace_layer_config: Dict):
        replace_layer_name = replace_layer_config["name"]
        for layer in self._model_config["layers"]:
            for inbound_node in layer["inbound_nodes"]:
                self._process_replacement(inbound_node, layer_name, replace_layer_name)

        self._replace_in_model_outputs(layer_name, replace_layer_name)

        idx, layer_config = self._find_layer_config(layer_name)
        replace_layer_config["inbound_nodes"] = layer_config["inbound_nodes"]
        self._model_config["layers"][idx] = replace_layer_config

    def _replace_sequential(self, layer_name: str, replace_layer_config: Dict):
        idx, _ = self._find_layer_config(layer_name)
        self._model_config["layers"][idx] = replace_layer_config

    def _insert_layers_before(self, layer_name: str, instance_idx: int, input_port_id: int, layers_to_insert: List):
        """
        Performs insertion before the (downstream) layer.

        :param layer_name: Name of the layer, before which to insert (downstream).
        :param instance_idx: Instance ID of the layer, before which to insert (downstream).
        :param input_port_id: Input port ID of the layer, before which to insert (downstream).
        :param layers_to_insert: List of the layers, which will be inserted into the graph.
        """
        functional_model = is_functional_model(self._model)

        if functional_model:
            for layer in self._model_config["input_layers"]:
                if layer_name == layer[0]:
                    raise nncf.InternalError("Insertion before input layer: {} is not supported".format(layer_name))

        layer_configs = []
        idx, downstream_layer_cfg = self._find_layer_config(layer_name)

        for layer in layers_to_insert:
            config = tf.keras.utils.serialize_keras_object(layer)
            if functional_model:
                config["name"] = config["config"]["name"]

                downstream_layer_inbound_nodes = downstream_layer_cfg["inbound_nodes"]
                if downstream_layer_cfg["class_name"] in ["TFOpLambda", "SlicingOpLambda"]:
                    downstream_layer_inbound_nodes = reformat_inbound_nodes_for_oplambda(downstream_layer_inbound_nodes)

                # Update config of the layer to insert
                inbound_node_info = copy.deepcopy(downstream_layer_inbound_nodes)[instance_idx][input_port_id]
                config["inbound_nodes"] = [[inbound_node_info[0], inbound_node_info[1], inbound_node_info[2], {}]]

                # Downstream layer config update
                if downstream_layer_cfg["class_name"] in ["TFOpLambda", "SlicingOpLambda"]:
                    downstream_layer_inbound_nodes[instance_idx][input_port_id][0] = config["name"]
                    downstream_layer_inbound_nodes[instance_idx][input_port_id][1] = 0
                    downstream_layer_inbound_nodes[instance_idx][input_port_id][2] = 0
                else:
                    self._model_config["layers"][idx]["inbound_nodes"][instance_idx][input_port_id] = [
                        config["name"],
                        0,
                        0,
                        {},
                    ]

            layer_configs.append(config)

        for config in layer_configs:
            self._model_config["layers"].insert(idx, config)

    def _insert_layers_after(self, layer_name: str, instance_idx: int, output_port_id: int, layers_to_insert: List):
        """
        Performs insertion after the (upstream) layer.

        :param layer_name: Name of the layer, after which to insert (upstream).
        :param instance_idx: Instance ID of the layer, after which to insert (upstream).
        :param input_port_id: Input port ID of the layer, after which to insert (upstream).
        :param layers_to_insert: List of the layers, which will be inserted into the graph.
        """
        functional_model = is_functional_model(self._model)

        layer_configs = []
        for layer in layers_to_insert:
            config = tf.keras.utils.serialize_keras_object(layer)
            if functional_model:
                config["name"] = config["config"]["name"]
                config["inbound_nodes"] = [[[layer_name, instance_idx, output_port_id, {}]]]
            layer_configs.append(config)

        for config in layer_configs:
            if functional_model:
                self._insert_layer_after_functional(layer_name, instance_idx, config)
            else:
                self._insert_layer_after_sequential(layer_name, config)

    def _insert_layer_after_functional(self, layer_name: str, instance_idx: int, layer_to_insert_config: Dict):
        """
        Performs insertion after the (upstream) layer (functional model).

        :param layer_name: Name of the layer, after which to insert (upstream).
        :param instance_idx: Instance ID of the Layer, after which to insert (upstream).
        :param layer_to_insert_config: Config of the layer, which is supposed to be inserted into the graph.
        """
        layer_out_ports = set()
        replace_layer_name = layer_to_insert_config["name"]  # new inbound_node name in the downstream layer

        for layer in self._model_config["layers"]:
            inbound_nodes = layer["inbound_nodes"]

            if layer["class_name"] in ["TFOpLambda", "SlicingOpLambda"]:
                inbound_nodes = reformat_inbound_nodes_for_oplambda(inbound_nodes)

            for inbound_node in inbound_nodes:
                self._process_insertion_after(
                    inbound_node, layer_name, instance_idx, layer_out_ports, replace_layer_name
                )

        self._insert_after_model_outputs(layer_name, instance_idx, layer_out_ports, replace_layer_name)
        if len(layer_out_ports) > 1:
            raise nncf.InternalError(
                "Insertion after layer ({}) with multiple ports is not supported".format(layer_name)
            )
        self._insert_layer_after_sequential(layer_name, layer_to_insert_config)

    def _insert_layer_after_sequential(self, layer_name: str, layer_configs):
        idx, _ = self._find_layer_config(layer_name)
        if idx is None:
            raise nncf.InternalError("Layer is not found: {}".format(layer_name))
        self._model_config["layers"].insert(idx + 1, layer_configs)

    @staticmethod
    def _process_insertion_after(
        connection_infos,
        layer_name: str,
        instance_idx: int,
        layer_out_ports: Set,
        replace_layer_name: str,
        insert_with_instance_idx: int = 0,
    ):
        if not isinstance(connection_infos[0], list):
            connection_infos = [connection_infos[0]]

        for connection_info in connection_infos:
            if connection_info[0] == layer_name:
                layer_out_ports.add(connection_info[2])
                if connection_info[1] == instance_idx:
                    connection_info[0] = replace_layer_name
                    connection_info[1] = insert_with_instance_idx

    def _insert_after_model_outputs(
        self,
        layer_name: str,
        instance_idx: int,
        layer_out_ports: Set,
        replace_layer_name: str,
        insert_with_instance_idx: int = 0,
    ):
        output_layers = self._model_config["output_layers"]
        if isinstance(output_layers, list):
            self._process_insertion_after(output_layers, layer_name, instance_idx, layer_out_ports, replace_layer_name)
        elif isinstance(output_layers, dict):
            for out_layers in output_layers.values():
                if isinstance(out_layers, list):
                    self._process_insertion_after(
                        [out_layers],
                        layer_name,
                        instance_idx,
                        layer_out_ports,
                        replace_layer_name,
                        insert_with_instance_idx,
                    )
                elif isinstance(out_layers, dict):
                    self._process_insertion_after(
                        list(out_layers.values()),
                        layer_name,
                        instance_idx,
                        layer_out_ports,
                        replace_layer_name,
                        insert_with_instance_idx,
                    )

    @staticmethod
    def _process_replacement(connection_infos, layer_name: str, replace_layer_name: str):
        if not isinstance(connection_infos[0], list):
            connection_infos = [connection_infos[0]]

        for connection_info in connection_infos:
            if connection_info[0] == layer_name:
                connection_info[0] = replace_layer_name

    def _replace_in_model_outputs(self, layer_name: str, replace_layer_name: str):
        output_layers = self._model_config["output_layers"]
        if isinstance(output_layers, list):
            self._process_replacement(output_layers, layer_name, replace_layer_name)
        elif isinstance(output_layers, dict):
            for out_layers in output_layers.values():
                if isinstance(out_layers, list):
                    self._process_replacement([out_layers], layer_name, replace_layer_name)
                elif isinstance(out_layers, dict):
                    self._process_replacement(list(out_layers.values()), layer_name, replace_layer_name)
