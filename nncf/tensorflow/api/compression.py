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
from typing import Dict, TypeVar, Any

import tensorflow as tf

from nncf.api.compression import CompressionState
from nncf.common.compression import BaseCompressionAlgorithmController
from nncf.common.compression import BaseCompressionAlgorithmBuilder
from nncf.tensorflow.graph.model_transformer import TFModelTransformer

ModelType = TypeVar('ModelType')


class TFCompressionAlgorithmController(BaseCompressionAlgorithmController, tf.train.experimental.PythonState):
    """
    Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model to enable algorithm-specific compression.
    Hosts entities that are to be used during the training process, such as
    compression scheduler and compression loss.
    """

    def get_compression_state(self) -> CompressionState:
        # TODO(nlyalyus) add support for TF
        return {}

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
