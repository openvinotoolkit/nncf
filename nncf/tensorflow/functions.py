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
from typing import Callable


@tf.function
def logit(x):
    return tf.math.log(x / (1 - x))


@tf.custom_gradient
def st_threshold(input_):
    def grad(upstream):
        return upstream
    return tf.round(input_), grad


def get_id_with_multiplied_grad(grad_multiplier: float) -> Callable[[tf.Tensor], tf.Tensor]:
    @tf.custom_gradient
    def id_with_multiplied_grad(x):
        def grad(upstream):
            if grad_multiplier is None:
                return upstream
            return grad_multiplier * upstream
        return x, grad

    return id_with_multiplied_grad
