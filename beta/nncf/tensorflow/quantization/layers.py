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

import tensorflow as tf

from beta.nncf.tensorflow.layers.custom_objects import NNCF_CUSTOM_OBJECTS
from beta.nncf.tensorflow.layers.custom_objects import NNCF_QUANTIZATION_OPERATONS
from beta.nncf.tensorflow.layers.operation import InputType
from beta.nncf.tensorflow.quantization.quantizers import Quantizer
from beta.nncf.tensorflow.quantization.quantizers import TFQuantizerSpec


@NNCF_CUSTOM_OBJECTS.register()
class FakeQuantize(tf.keras.layers.Layer):
    def __init__(self, config: TFQuantizerSpec, data_format='channels_last', **kwargs):
        """
        Create a FakeQuantize layer.
        """
        super().__init__(**kwargs)
        self.mode = config.mode
        self.data_format = data_format

        self._quantizer = self._create_quantizer(config)
        self._quantizer_weights = {}

    @property
    def num_bits(self):
        return getattr(self._quantizer, 'num_bits', None)

    @property
    def per_channel(self):
        return getattr(self._quantizer, 'per_channel', None)

    @property
    def narrow_range(self):
        return getattr(self._quantizer, 'narrow_range', None)

    @property
    def signed(self):
        return getattr(self._quantizer, 'signed', None)

    @property
    def enabled(self):
        return self._quantizer.enabled

    @enabled.setter
    def enabled(self, v):
        self._quantizer.enabled = v

    def build(self, input_shape):
        self._quantizer_weights = self._quantizer.build(
            input_shape, InputType.INPUTS, self.name, self)

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        return self._quantizer(inputs, self._quantizer_weights, training)

    def register_hook_pre_quantizer(self, hook):
        return self._quantizer.register_hook_pre_call(hook)

    def apply_minmax_initialization(self, min_values, max_values, min_range=0.1, eps=0.01):
        self._quantizer.apply_minmax_initialization(self._quantizer_weights, min_values, max_values, min_range, eps)

    def _create_quantizer(self, qspec: TFQuantizerSpec) -> Quantizer:
        quantizer_cls = NNCF_QUANTIZATION_OPERATONS.get(qspec.mode)
        return quantizer_cls(qspec)

    def get_config(self):
        config = super().get_config()
        config.update({
            'quantizer_config': {
                'mode': self.mode,
                **self._quantizer.get_config()
            },
            'data_format': self.data_format
        })
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        quantizer_config = config.pop('quantizer_config')
        return cls(TFQuantizerSpec(**quantizer_config), **config)
