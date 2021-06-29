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

import json
from typing import Any
from typing import Dict
from typing import TypeVar

import tensorflow as tf

from nncf.api.compression import CompressionState
from nncf.common.compression import BaseCompressionAlgorithmBuilder
from nncf.common.compression import BaseCompressionAlgorithmController
from nncf.tensorflow.graph.model_transformer import TFModelTransformer

ModelType = TypeVar('ModelType')


class TFCompressionAlgorithmController(BaseCompressionAlgorithmController, tf.train.experimental.PythonState):
    """
    Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model to enable algorithm-specific compression.
    Hosts entities that are to be used during the training process, such as
    compression scheduler and compression loss.
    """

    def get_compression_state(self) -> 'TFCompressionState':
        if self._builder_state is None:
            raise RuntimeError('Internal error: builder state is not set for the controller')
        return TFCompressionState(builder_state=self._builder_state, compression_ctrl=self)

    def serialize(self) -> str:
        """
        Callback to serialize the object by tf.train.experimental.PythonState.

        :return: State of the compression controller.
        """
        string_value = json.dumps(self.get_state())
        return string_value

    def deserialize(self, state: str) -> None:
        """
        Callback to deserialize the object by tf.train.experimental.PythonState.

        :param state: State of the compression controller.
        """
        state = json.loads(state)
        self.load_state(state)


class TFCompressionAlgorithmBuilder(BaseCompressionAlgorithmBuilder):
    """
    Determines which modifications should be made to the original model in
    order to enable algorithm-specific compression during fine-tuning.
    """

    def _get_state_without_name(self) -> Dict[str, Any]:
        """
        Implementation of get_state that returns state without builder name.

        :return: Returns a dictionary with Python data structures
            (dict, list, tuple, str, int, float, True, False, None) that represents state of the object.
        """
        return {}

    def _load_state_without_name(self, state_without_name: Dict[str, Any]):
        """
        Implementation of load state that takes state without builder name.

        :param state_without_name: Output of `_get_state_without_name()` method.
        """

    def apply_to(self, model: ModelType) -> ModelType:
        """
        Applies algorithm-specific modifications to the model.

        :param model: The original uncompressed model.
        :return: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        """
        transformation_layout = self.get_transformation_layout(model)
        transformer = TFModelTransformer(model)
        transformed_model = transformer.transform(transformation_layout)

        if self.should_init:
            self.initialize(transformed_model)

        return transformed_model


class TFCompressionState(CompressionState, tf.train.experimental.PythonState):
    """
    Contains compression state of the TensorFlow model to unambiguously resume compression from it.
    Consists of builder and controller state - a dictionaries with Python data structures,
    defining how to setup and handle the compression correspondingly
    """

    def __init__(self, compression_ctrl: TFCompressionAlgorithmController = None, builder_state: Dict = None):
        self._compression_ctrl = compression_ctrl
        self._builder_state = builder_state
        self._ctrl_state = None

    @property
    def builder_state(self) -> Dict:
        return self._builder_state

    @property
    def ctrl_state(self) -> Dict:
        if self._compression_ctrl is not None:
            return self._compression_ctrl.get_state()
        return self._ctrl_state

    def load_state(self, state: Dict):
        """
        Initializes object from the state.

        :param state: Output of `get_state()` method.
        """
        self._builder_state = state[self.BUILDER_STATE]
        self._ctrl_state = state[self.CONTROLLER_STATE]

    def serialize(self) -> str:
        """
        Callback to serialize the object by tf.train.experimental.PythonState.

        :return: Serialized compression state.
        """
        string_value = json.dumps(self.get_state())
        return string_value

    def deserialize(self, state: str) -> None:
        """
        Callback to deserialize the object by tf.train.experimental.PythonState.

        :param state: Serialized compression state.
        """
        state = json.loads(state)
        self.load_state(state)
