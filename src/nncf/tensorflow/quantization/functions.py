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


def symmetric_quantize(
    inputs, scale_var, signed_var, num_bits, per_channel, narrow_range, eps, name_prefix="SymmQuant"
):
    with tf.name_scope(name_prefix):
        scale_safe = tf.abs(scale_var) + eps
        min_var = scale_safe * signed_var
        max_var = scale_safe
        return _fake_quant_with_min_max_vars(inputs, min_var, max_var, num_bits, narrow_range, per_channel)


def asymmetric_quantize(
    inputs, input_low, input_range, num_bits, per_channel, narrow_range, eps, name_prefix="AsymmQuant"
):
    with tf.name_scope(name_prefix):
        input_range_safe = tf.abs(input_range) + eps
        min_var = input_low
        max_var = input_low + input_range_safe
        return _fake_quant_with_min_max_vars(inputs, min_var, max_var, num_bits, narrow_range, per_channel)


def _fake_quant_with_min_max_vars(inputs, min_var, max_var, num_bits, narrow_range, per_channel):
    if per_channel:
        return tf.quantization.fake_quant_with_min_max_vars_per_channel(
            inputs, min_var, max_var, num_bits=num_bits, narrow_range=narrow_range
        )
    return tf.quantization.fake_quant_with_min_max_vars(
        inputs, min_var, max_var, num_bits=num_bits, narrow_range=narrow_range
    )
