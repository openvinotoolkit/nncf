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

from collections import OrderedDict
from inspect import getfullargspec
from typing import Dict

import tensorflow as tf

import nncf
from nncf.tensorflow.layers.custom_objects import NNCF_CUSTOM_OBJECTS
from nncf.tensorflow.layers.custom_objects import get_nncf_custom_objects
from nncf.tensorflow.layers.operation import InputType
from nncf.tensorflow.layers.operation import NNCFOperation


@NNCF_CUSTOM_OBJECTS.register()
class NNCFWrapper(tf.keras.layers.Wrapper):
    """
    This wrapper augments a keras layer so the NNCF Operations may be applied to weights,
    callable attributes (like activations), input and output of the wrapped layer.
    """

    def __init__(self, layer, **kwargs):
        """
        Create a pruning wrapper for a keras layer.

        :param layer: the keras layer to be wrapped
        :param kwargs: additional keyword arguments to be passed to the keras layer.
        """
        if layer is None:
            raise ValueError("`layer` cannot be None.")

        if not isinstance(layer, tf.keras.layers.Layer) or isinstance(layer, tf.keras.Model):
            raise ValueError(
                "`layer` can only be a `tf.keras.layers.Layer` instance. "
                "You passed an instance of type: {input}.".format(input=layer.__class__.__name__)
            )

        if "name" not in kwargs:
            kwargs["name"] = layer.name

        super().__init__(layer, **kwargs)
        self._track_trackable(layer, name="layer")

        self.weights_attr_ops: Dict[str, Dict[str, NNCFOperation]] = {}

        self._init_layer_call_fn_args()
        self._trainable_weights = []
        self._non_trainable_weights = []
        self._ops_weights = {}
        self._op_build = False
        self._layer_weights = {}

    @property
    def trainable(self):
        return self.layer.trainable

    @trainable.setter
    def trainable(self, value):
        """
        By default, trainable property of operations follows the trainable property of the layer:
        - To freeze both wrapped layer weights and op weights -> layer.trainable = False
        - To unfreeze both wrapped layer weights and op weights -> layer.trainable = True
        """
        self.layer.trainable = value
        self.set_ops_trainable(value)

    def set_ops_trainable(self, value):
        """
        Introduction of trainable property to the operations gives the following additional options:
        - Frozen wrapped layer weights, unfrozen op weights -> layer.trainable = False; layer.set_ops_trainable(True)
        - Unfrozen wrapped layer weights, frozen op weights -> layer.trainable = True; layer.set_ops_trainable(False)
        """
        for ops in self.weights_attr_ops.values():
            for op in ops.values():
                op.trainable = value

    @property
    def trainable_weights(self):
        trainable_weights_by_definition = self._trainable_weights + self.layer.trainable_weights

        if self._op_build:
            trainable_ops_weights_on, trainable_ops_weights_off = self.classify_trainable_ops_weights()
        else:
            trainable_ops_weights_on, trainable_ops_weights_off = [], []

        if self.trainable:
            weights = trainable_weights_by_definition
            for trainable_weight_off in trainable_ops_weights_off:
                weights.remove(trainable_weight_off)
            outputs = weights
        else:
            outputs = trainable_ops_weights_on
        return outputs

    @property
    def non_trainable_weights(self):
        trainable_weights_by_definition = self._trainable_weights + self.layer.trainable_weights
        non_trainable_weights_by_definition = self._non_trainable_weights + self.layer.non_trainable_weights

        if self._op_build:
            trainable_ops_weights_on, trainable_ops_weights_off = self.classify_trainable_ops_weights()
        else:
            trainable_ops_weights_on, trainable_ops_weights_off = [], []

        if self.trainable:
            outputs = non_trainable_weights_by_definition + trainable_ops_weights_off
        else:
            weights = trainable_weights_by_definition
            for trainable_weight_on in trainable_ops_weights_on:
                weights.remove(trainable_weight_on)
            outputs = non_trainable_weights_by_definition + weights
        return outputs

    def classify_trainable_ops_weights(self):
        """
        Classifies operation trainable weights by trainable property of the corresponding operation.
        Note: Operation with trainable weights can be switched off from training.
              Operation trainable property can be adjusted
              to be different from NNCFWrapper global trainable property.

        :return trainable_ops_weights_on: List of trainable operation weights
                which is getting updated during training.
        :return trainable_ops_weights_off: List of trainable operation weights
                which is NOT getting updated during training.
        """
        trainable_ops_weights_on = []
        trainable_ops_weights_off = []
        for ops in self.weights_attr_ops.values():
            for op_name, op in ops.items():
                op_weights = self._ops_weights[op_name]
                for weight_val in op_weights.values():
                    if not weight_val.trainable:
                        continue

                    if op.trainable:
                        trainable_ops_weights_on.append(weight_val)
                    else:
                        trainable_ops_weights_off.append(weight_val)
        return trainable_ops_weights_on, trainable_ops_weights_off

    @property
    def updates(self):
        return self.layer.updates + self._updates

    @property
    def losses(self):
        return self.layer.losses + self._losses

    @property
    def data_format(self):
        return getattr(self.layer, "data_format", "channels_last")

    @property
    def ops_weights(self):
        return self._ops_weights

    @property
    def layer_weights(self):
        return self._layer_weights

    def build(self, input_shape=None):
        super().build(input_shape)
        for weight_attr, ops in self.weights_attr_ops.items():
            weight = self.get_layer_weight(weight_attr)
            for op_name, op in ops.items():
                self._ops_weights[op_name] = op.build(weight.shape, InputType.WEIGHTS, weight_attr, self)
            self._layer_weights[weight_attr] = weight
            self._trainable_weights.append(weight)
        self._op_build = True

    def call(self, inputs, training=None):
        training = self._get_training_value(training)

        self._apply_ops(training)

        if self._layer_expects_training_arg:
            outputs = self.layer.call(inputs, training=training)
        else:
            outputs = self.layer.call(inputs)

        return outputs

    def _apply_ops(self, training):
        for weight_attr, ops in self.weights_attr_ops.items():
            layer_weight = self._layer_weights[weight_attr]
            for op_name, op in ops.items():
                layer_weight = op(layer_weight, self._ops_weights[op_name], training)
            self.set_layer_weight(weight_attr, layer_weight)

    def registry_weight_operation(self, weights_attr: str, op: NNCFOperation):
        if weights_attr not in self.weights_attr_ops:
            self.weights_attr_ops[weights_attr] = OrderedDict()

        if op.name in self.weights_attr_ops[weights_attr]:
            raise nncf.InternalError(
                f"Attempt to apply an operation with the same name {op.name} on layer weight twice"
            )

        self.weights_attr_ops[weights_attr][op.name] = op

    def get_operation_weights(self, operation_name):
        return self._ops_weights[operation_name]

    def get_layer_weight(self, weight_attr):
        weight = getattr(self.layer, weight_attr, None)
        if weight is not None:
            return weight
        for w in self.layer.weights:
            if w.name.split(":")[0] == weight_attr:
                return w
        return None

    def set_layer_weight(self, weight_attr, weights):
        if hasattr(self.layer, weight_attr):
            setattr(self.layer, weight_attr, weights)
        else:
            self._layer_weights[weight_attr].assign(weights)

    def _init_layer_call_fn_args(self):
        call_full_argspec = getfullargspec(self.layer.call)
        call_fn_args = self._get_call_fn_args(call_full_argspec)
        self._layer_expects_training_arg = "training" in call_fn_args

    @staticmethod
    def _get_call_fn_args(call_full_argspec):
        all_args = call_full_argspec.args + call_full_argspec.kwonlyargs
        if all_args and all_args[0] == "self":
            return all_args[1:]
        return all_args

    @staticmethod
    def _get_training_value(training):
        if training is None:
            training = tf.keras.backend.learning_phase()
            if tf.is_tensor(training):
                training = tf.cast(training, tf.bool)
            else:
                training = bool(training)
        return training

    def get_config(self):
        config = super().get_config()

        weights_attr_ops = {}
        for weights_attr, ops in self.weights_attr_ops.items():
            weights_attr_ops[weights_attr] = []
            for op_name in ops:
                op_config = tf.keras.utils.serialize_keras_object(ops[op_name])
                weights_attr_ops[weights_attr].append(op_config)
        config["weights_attr_operations"] = weights_attr_ops
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()

        weights_attr_ops_config = config.pop("weights_attr_operations")

        layer = tf.keras.layers.deserialize(config.pop("layer"), custom_objects=custom_objects)
        wrapper = cls(layer=layer, **config)

        for weights_attr, operations in weights_attr_ops_config.items():
            for op_config in operations:
                wrapper.registry_weight_operation(
                    weights_attr, tf.keras.layers.deserialize(op_config, custom_objects=get_nncf_custom_objects())
                )

        return wrapper
