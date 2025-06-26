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

import tensorflow as tf

from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.tensorflow.layers.custom_objects import NNCF_CUSTOM_OBJECTS
from nncf.tensorflow.layers.custom_objects import NNCF_QUANTIZATION_OPERATIONS
from nncf.tensorflow.layers.operation import InputType
from nncf.tensorflow.quantization.quantizers import Quantizer
from nncf.tensorflow.quantization.quantizers import TFQuantizerSpec


@NNCF_CUSTOM_OBJECTS.register()
class FakeQuantize(tf.keras.layers.Layer):
    def __init__(self, config: TFQuantizerSpec, data_format: str = "channels_last", **kwargs):
        """
        Create a FakeQuantize layer.
        """
        super().__init__(**kwargs)
        self._mode = config.mode
        self.data_format = data_format

        self._op_name = f"{self.name}_quantizer"
        self._quantizer = self._create_quantizer(config, self._op_name)
        self._quantizer_weights = {}

    @property
    def num_bits(self):
        return getattr(self._quantizer, "num_bits", None)

    @property
    def per_channel(self):
        return getattr(self._quantizer, "per_channel", None)

    @property
    def narrow_range(self):
        return getattr(self._quantizer, "narrow_range", None)

    @property
    def signed(self) -> bool:
        """
        Returns `True` for signed quantization, `False` for unsigned.

        :return: `True` for signed quantization, `False` for unsigned.
        """
        if self._quantizer.mode == QuantizationMode.SYMMETRIC:
            return self._quantizer.signed(self._quantizer_weights)
        return True

    @property
    def mode(self) -> str:
        """
        Returns mode of the quantization (symmetric or asymmetric).

        :return: The mode of the quantization.
        """
        return self._mode

    @property
    def op_name(self):
        return self._op_name

    @property
    def enabled(self):
        return self._quantizer.enabled

    @enabled.setter
    def enabled(self, v):
        self._quantizer.enabled = v

    def build(self, input_shape):
        self._quantizer_weights = self._quantizer.build(input_shape, InputType.INPUTS, self.name, self)

    def call(self, inputs, training=None):
        training = self._get_training_value(training)
        return self._quantizer(inputs, self._quantizer_weights, training)

    def register_hook_pre_quantizer(self, hook):
        return self._quantizer.register_hook_pre_call(hook)

    def apply_range_initialization(self, min_values, max_values, min_range=0.1, eps=0.01):
        self._quantizer.apply_range_initialization(self._quantizer_weights, min_values, max_values, min_range, eps)

    def _create_quantizer(self, qspec: TFQuantizerSpec, op_name: str) -> Quantizer:
        quantizer_cls = NNCF_QUANTIZATION_OPERATIONS.get(qspec.mode)
        return quantizer_cls(op_name, qspec)

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
        config.update(
            {"quantizer_config": {**self._quantizer.get_config()["quantizer_spec"]}, "data_format": self.data_format}
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        quantizer_config = config.pop("quantizer_config")
        return cls(TFQuantizerSpec(**quantizer_config), **config)
