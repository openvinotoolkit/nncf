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

from beta.nncf.tensorflow.functions import logit


def binary_mask(mask):
    return tf.round(tf.math.sigmoid(mask))


@tf.custom_gradient
def st_binary_mask(mask):
    def grad(upstream):
        return upstream
    return binary_mask(mask), grad


def calc_rb_binary_mask(mask, eps=0.01):
    # TODO: check in distributed mode (mirrored strategy)
    uniform = tf.random.uniform(mask.shape, minval=0, maxval=1)
    mask = mask + logit(tf.clip_by_value(uniform, eps, 1 - eps))
    return st_binary_mask(mask)
