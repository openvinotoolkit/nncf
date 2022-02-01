"""
 Copyright (c) 2022 Intel Corporation
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

from typing import Optional
from typing import Tuple

import tensorflow as tf


def symmetric_quantize(inputs,
                       scale_var,
                       signed_var,
                       num_bits,
                       per_channel,
                       narrow_range,
                       eps,
                       name_prefix='SymmQuant'):
    with tf.name_scope(name_prefix):
        scale_safe = tf.abs(scale_var) + eps
        min_var = scale_safe * signed_var
        max_var = scale_safe
        return _fake_quant_with_min_max_vars(inputs, min_var, max_var, num_bits,
                                             narrow_range, per_channel)


def asymmetric_quantize(inputs,
                        input_low,
                        input_range,
                        num_bits,
                        per_channel,
                        narrow_range,
                        eps,
                        name_prefix='AsymmQuant'):
    with tf.name_scope(name_prefix):
        input_range_safe = tf.abs(input_range) + eps
        min_var = input_low
        max_var = input_low + input_range_safe
        return _fake_quant_with_min_max_vars(inputs, min_var, max_var, num_bits,
                                             narrow_range, per_channel)


def _fake_quant_with_min_max_vars(inputs, min_var, max_var, num_bits, narrow_range,
                                  per_channel):
    if per_channel:
        return tf.quantization.fake_quant_with_min_max_vars_per_channel(
            inputs, min_var, max_var, num_bits=num_bits, narrow_range=narrow_range)
    return tf.quantization.fake_quant_with_min_max_vars(
        inputs, min_var, max_var, num_bits=num_bits, narrow_range=narrow_range)


def calc_asymmetric_range_initialization_params(min_values: tf.Tensor,
                                                max_values: tf.Tensor,
                                                min_range: float = 0.1,
                                                eps: float = 0.01) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Calculates parameters for the asymmetric range initialization.

    :param min_values: Minimum weight values.
    :param max_values: Maximum weight values.
    :param min_range: Minimum range.
    :param eps: Smoothing coefficient for ranges: min_range = maximum(min_range, eps * max_range).
    :return: Parameters for the asymmetric range initialization.
    """
    ranges = max_values - min_values
    max_range = tf.reduce_max(ranges)
    lower_threshold = tf.maximum(eps * max_range, min_range)
    correction = (tf.maximum(ranges, lower_threshold) - ranges) * 0.5
    input_low = min_values - correction
    input_range = ranges + 2 * correction
    return input_low, input_range


def calc_symmetric_range_initialization_params(
        min_values: tf.Tensor,
        max_values: tf.Tensor,
        min_range: float = 0.1,
        eps: float = 0.01,
        signedness_to_force: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Calculates parameters for the symmetric range initialization.

    :param min_values: Minimum weight values.
    :param max_values: Maximum weight values.
    :param min_range: Minimum range.
    :param eps: Smoothing coefficient for ranges: min_range = maximum(min_range, eps * max_range).
    :param signedness_to_force: One of the following values:
        - `True` if the quantizer must be signed
        - `False` if the quantizer must be unsigned
        - `None` if the signed/unsigned attribute should be determined based
        on the incoming activation statistics during range initialization.
    :return: Parameters for the symmetric range initialization.
    """
    signed = -1.0 if signedness_to_force in (True, None) else 0.0
    if signedness_to_force is None:
        sign = tf.reduce_any(tf.less(min_values, 0))
        signed = -1.0 if sign else 0.0

    ranges = tf.maximum(tf.abs(max_values), tf.abs(min_values))
    max_range = tf.reduce_max(ranges)
    lower_threshold = tf.maximum(eps * max_range, min_range)
    scale = tf.maximum(ranges, lower_threshold)
    return signed, scale
