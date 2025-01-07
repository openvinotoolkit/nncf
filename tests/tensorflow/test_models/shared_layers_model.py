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
from tensorflow.keras import layers


def SharedLayersModel(input_shape):
    inputs = tf.keras.Input(input_shape)
    x0 = layers.Conv2D(8, 3, name="c0")(inputs)

    x0s1, x0s2 = tf.split(x0, 2, axis=-1)
    x1 = layers.Conv2D(2, 3, name="c1")(x0s1)
    x2 = layers.Conv2D(2, 3, name="c2")(x0s2)
    c3 = layers.Conv2D(1, 3, name="c3")
    c4 = layers.Conv2D(5, 3, name="c4")
    x31 = c3(x1)
    x32 = c3(x2)
    x41 = c4(x31)
    x42 = c4(x32)
    return tf.keras.Model(inputs=inputs, outputs=[x41, x42])
