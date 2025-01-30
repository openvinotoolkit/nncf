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

import nncf  # noqa: F401
from nncf.common.compression import BaseCompressionAlgorithmController
from nncf.tensorflow.helpers.model_creation import create_compressed_model
from tests.tensorflow.quantization.utils import get_basic_quantization_config

# Do not remove - these imports are for testing purposes.


inputs = tf.keras.Input(shape=(3, 3, 1))
outputs = tf.keras.layers.Conv2D(filters=3, kernel_size=3)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

config = get_basic_quantization_config()
compression_state_to_skip_init = {BaseCompressionAlgorithmController.BUILDER_STATE: {}}
compression_model, compression_ctrl = create_compressed_model(model, config, compression_state_to_skip_init)
