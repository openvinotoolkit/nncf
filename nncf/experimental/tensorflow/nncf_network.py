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

import tensorflow as tf


class NNCFNetwork(tf.keras.Model):
    """
    Wraps the Keras model.
    """

    def __init__(self, model, input_signature, **kwargs):
        """
        Initializes the NNCF network.

        :param model: Keras model.
        :param input_signature: Input signature of the moodel.
        """
        super().__init__(**kwargs)
        self._model = model
        self._input_signature = input_signature

    @property
    def input_signature(self):
        return self._input_signature

    def get_config(self):
        # TODO(andrey-churkin): I will investigate it
        return None

    def call(self, inputs, **kwargs):
        outputs = self._model(inputs, **kwargs)
        return outputs
