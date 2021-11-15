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

from nncf.common.utils.registry import Registry


aggregator = Registry('TFAggregationFunctions')


def get_aggregation_function(name):
    return aggregator.get(name)


@aggregator.register()
def tf_reduce_min(x, reduction_shape):
    return tf.squeeze(tf.reduce_min(x, axis=reduction_shape))


@aggregator.register()
def tf_reduce_max(x, reduction_shape):
    return tf.squeeze(tf.reduce_max(x, axis=reduction_shape))


@aggregator.register()
def tf_abs(x):
    return tf.math.abs(x)


@aggregator.register()
def tf_min(x1, x2):
    return tf.math.minimum(x1, x2)


@aggregator.register()
def tf_max(x1, x2):
    return tf.math.maximum(x1, x2)


@aggregator.register()
def tf_tensor_min(x, axis):
    return tf_reduce_min(x, axis)


@aggregator.register()
def tf_tensor_max(x, axis):
    return tf_reduce_max(x, axis)


@aggregator.register()
def tf_mean(x, axis):
    return tf.math.reduce_mean(x, axis=axis)


@aggregator.register()
def tf_stack(x):
    return tf.stack(x)


@aggregator.register()
def tf_list_to_extend_stat_history(x):
    return tf.unstack(x)
