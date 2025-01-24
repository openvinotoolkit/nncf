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

import json
from typing import Any, Dict

import tensorflow as tf

from nncf.common.compression import BaseCompressionAlgorithmController

# TODO(achurkin): remove pylint ignore after 120296 ticked is fixed


class TFCompressionState(tf.train.experimental.PythonState):
    """
    A wrapper for `BaseCompressionAlgorithmController` that allows saving
    the compression state to the checkpoint.
    """

    def __init__(self, controller: BaseCompressionAlgorithmController):
        """
        Initializes the wrapper for the controller.

        :param controller: The controller which gives the compressions state.
        """
        self._ctrl = controller

    def serialize(self) -> str:
        """
        Callback to serialize the compression state.

        :return: A serialized compression state.
        """
        compression_state = self._ctrl.get_compression_state()
        return json.dumps(compression_state)

    def deserialize(self, string_value: str) -> None:
        """
        Callback to deserialize the compression state.

        :param string_value: A serialized compression state.
        """
        compression_state = json.loads(string_value)
        ctrl_state = compression_state[BaseCompressionAlgorithmController.CONTROLLER_STATE]
        self._ctrl.load_state(ctrl_state)


class TFCompressionStateLoader(tf.train.experimental.PythonState):
    """
    This is a class that allows extracting of the compression state from a checkpoint.
    The extracted compression state is not applied.
    """

    def __init__(self):
        """
        Initializes the compression state loader.
        """
        self._state = None

    @property
    def state(self) -> Dict[str, Any]:
        """
        Returns the compression state which was extracted from the checkpoint.

        :return: The compression state.
        """
        return self._state

    def serialize(self) -> str:
        raise NotImplementedError(
            "Use an instance of the `TFCompressionState` class to serialize the compression state."
        )

    def deserialize(self, string_value: str) -> None:
        """
        Callback to deserialize the compression state.

        :param string_value: A serialized compression state.
        """
        self._state = json.loads(string_value)
