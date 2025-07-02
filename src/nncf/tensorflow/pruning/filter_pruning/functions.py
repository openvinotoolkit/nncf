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


def l1_filter_norm(weight_tensor, dim=0):
    """
    Calculates L1 for weight_tensor for the selected dimension.
    """
    transformed_tensor = _pull_tensor(weight_tensor, dim)
    return tf.norm(transformed_tensor, ord=1, axis=1)


def l2_filter_norm(weight_tensor, dim=0):
    """
    Calculates L2 for weight_tensor for the selected dimension.
    """
    transformed_tensor = _pull_tensor(weight_tensor, dim)
    return tf.norm(transformed_tensor, ord=2, axis=1)


def _pull_tensor(weight_tensor, dim=0):
    permutation = list(range(len(weight_tensor.shape)))
    permutation[0], permutation[dim] = permutation[dim], permutation[0]
    weight_tensor = tf.transpose(weight_tensor, perm=permutation)
    return tf.reshape(weight_tensor, [weight_tensor.shape[0], -1])


def tensor_l2_normalizer(weight_tensor):
    return tf.math.l2_normalize(weight_tensor)


def geometric_median_filter_norm(weight_tensor, dim=0):
    """
    Compute geometric median norm for filters.
    :param weight_tensor: tensor with weights
    :param dim: dimension of output channel
    :return: metric value for every weight from weights_tensor
    """
    weight_vec = _pull_tensor(weight_tensor, dim)
    square_norms = tf.reduce_sum(tf.square(weight_vec), axis=1, keepdims=True)
    similar_matrix = tf.sqrt(
        tf.maximum(
            square_norms - 2 * tf.matmul(weight_vec, weight_vec, transpose_b=True) + tf.transpose(square_norms), 0
        )
    )
    similar_sum = tf.reduce_sum(similar_matrix, axis=0)
    return similar_sum


def _l2_distance(x, y):
    return tf.sqrt(tf.reduce_sum(tf.square(x - y)))


FILTER_IMPORTANCE_FUNCTIONS = {
    "L2": l2_filter_norm,
    "L1": l1_filter_norm,
    "geometric_median": geometric_median_filter_norm,
}


def calculate_binary_mask(weight_importance, threshold):
    return tf.cast(weight_importance >= threshold, tf.float32)
