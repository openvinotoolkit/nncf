"""
 Copyright (c) 2021 Intel Corporation
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
from collections import OrderedDict

from nncf.tensorflow.utils.hook_handle import HookHandle
from nncf.experimental.tensorflow.nncf_network import NNCFNetwork


class NNCFOperation(ABC):
    """
    Represents the main building block for adding compression
    extensions to the NNCF network.
    """

    def __init__(self, name: str, trainable: bool = True):
        """
        Initializes the internal state of an NNCF operation.

        :param name: Name of operation. Unique identifier inside
            the NNCF network.
        :param trainable: Boolean flag that specified whether the
            operation's variables should be trainable.
        """
        self._name = name
        self._trainable = trainable
        self._call_pre_hooks = OrderedDict()

    @property
    def name(self) -> str:
        return self._name

    @property
    def trainable(self) -> bool:
        return self._trainable

    @trainable.setter
    def trainable(self, value: bool):
        self._trainable = value

    @abstractmethod
    def build(self, nncf_network: NNCFNetwork) -> None:
        """
        Creates operation's weights and adds them to the NNCF network.

        Adding operation weights to the NNCF network is required
        because this is the only way to save it to a checkpoint.

        :param nncf_network: NNCF network, where this operation was added.
        """

    @abstractmethod
    def call(self, inputs, *args, **kwargs):
        """
        Performs the logic of applying the operation to the input tensor.

        :param inputs: Input tensor.
        :return: Result of operation.
        """

    def register_hook_pre_call(self, hook) -> HookHandle:
        """
        Registers a hook which will be called before the `call` method.
        The NNCF operation does not support serialization of the registered hooks.

        :param hook: Callable object with the following signature:
            hook(inputs) -> None or modified input.
        :return: A handle that can be used to remove the hook from
            the NNCF operation by calling the `HookHandle.remove()` method.
        """
        handle = HookHandle(self._call_pre_hooks)
        self._call_pre_hooks[handle.hook_id] = hook
        return handle

    def __call__(self, *args, **kwargs):
        inputs = args[0]
        for hook in self._call_pre_hooks.values():
            result = hook(inputs)
            if result is not None:
                inputs = result
        return self.call(inputs, *args[1:], **kwargs)
