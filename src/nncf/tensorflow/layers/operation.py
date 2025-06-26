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

from nncf.common.hook_handle import add_op_to_registry


class InputType:
    INPUTS = "inputs"
    WEIGHTS = "weights"


class NNCFOperation:
    """
    The abstract class represents main building block for adding compression
    extensions to a model.
    """

    def __init__(self, name, trainable=True):
        """
        Initializes internal NNCF operation state

        :param name: unique operation name in algorithm scope.
        """
        self._call_pre_hooks = OrderedDict()
        self._name = name
        self._trainable = trainable

    @property
    def name(self):
        return self._name

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value

    def build(self, input_shape, input_type, name, layer):
        """
        This method can be used to create weights that depend on the shape(s)
        of the input(s) and register them in the NNCF Wrapper `layer`. The method
        will be automatically called when NNCF Wrapper `layer` is built.

        :param input_shape: shape of the input
        :param input_type: type of the input identifies that inputs are layer weights
                           or inputs of the layer
        :param name: operation name
        :param layer: NNCF Wrapper layer, where the operation is registered
        :return: weights dictionary {weight name: weight value}
        """

    def call(self, inputs, weights, training):
        """
        The method performs the logic of applying the operation to the input tensors
        (which should be passed in as argument).

        :param inputs: input tensors
        :param weights: operation weights
        :param training: identifying that the model is training or evaluating
        :return: output tensors
        """
        raise NotImplementedError

    def register_hook_pre_call(self, hook):
        """
        Registers a hook which will be called before `call` function.
        The NNCF operation does not support serialization of the registered hooks.

        :param hook: callable object with following signatures:
                     hook(inputs) -> None or modified input
        :return: a handle that can be used to remove the hook form
                 the NNCF operation by calling handle.remove()
        """
        return add_op_to_registry(self._call_pre_hooks, hook)

    def __call__(self, *args, **kwargs):
        inputs = args[0]
        for hook in self._call_pre_hooks.values():
            result = hook(inputs)
            if result is not None:
                inputs = result
        return self.call(inputs, *args[1:], **kwargs)

    def get_config(self):
        return {"name": self._name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
