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

import numpy as np
import pytest
import tensorflow as tf

from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.tensorflow.layers.custom_objects import NNCF_QUANTIZATION_OPERATIONS
from nncf.tensorflow.layers.wrapper import NNCFWrapper
from nncf.tensorflow.quantization.quantizers import Quantizer
from nncf.tensorflow.quantization.quantizers import QuantizerConfig
from nncf.tensorflow.quantization.quantizers import TFQuantizerSpec
from nncf.tensorflow.quantization.utils import apply_overflow_fix_to_layer

DIM_SPLIT = 1000
EPS = 1e-6


def check_quantized_values_equals(y_train, y_val, eps, range_len, narrow_range):
    diff = np.abs(y_val - y_train)
    if np.max(diff) > eps:
        # If any point gets in really close to the middle of the quant
        # it can changes its quant due to rounding error
        outlayers = diff[diff > eps]
        quant_len = range_len / (128 - (2 if narrow_range else 1))
        assert (np.abs(outlayers - quant_len) < eps).all(), "Quants are completely different"
        assert False, (
            "Some values moved to the neighbor quant, possibly due to this values gets in "
            "really close to the middle of the quant. "
            f"Position of values: {np.where(diff > eps)[0].tolist()}"
        )


@pytest.mark.parametrize(
    "bits,low,range_,narrow_range,ref",
    [(7, -1, 2, False, -128 / 127), (7, -2, 2, True, -2)],
    ids=["full_range", "narrow_range"],
)
def test_min_adj(bits, low, range_, narrow_range, ref):
    res = Quantizer._min_adj(bits, low, range_, narrow_range).numpy()
    assert abs(res - ref) < EPS


def get_weights_for_overflow_issue_test(low, range_len, narrow_range, init_w_as_middle_points):
    if init_w_as_middle_points:
        quant_len = range_len / (128 - (2 if narrow_range else 1))
        if low > EPS:
            # Range greater than zero
            mid_points = [(i + 1 / 2) * quant_len for i in range(127)]
        elif low + range_len < EPS:
            # Range lower than zero
            mid_points = [-(i + 1 / 2) * quant_len for i in range(127)]
        else:
            # Range with zero
            min_adj = Quantizer._min_adj(7, low, range_len, narrow_range).numpy()
            mid_points = [min_adj + (i + 1 / 2) * quant_len for i in range(127)]

        new_w = mid_points * int(np.round(0.5 + DIM_SPLIT / 128))
        new_w = tf.reshape(tf.constant(new_w[:DIM_SPLIT], dtype=tf.float32), (1, -1))
    else:
        new_w = tf.reshape(
            tf.constant(np.linspace(low - 0.5, low + range_len + 0.5, DIM_SPLIT), dtype=tf.float32), (1, -1)
        )

    return new_w


@pytest.mark.parametrize("per_ch", [False, True], ids=["per_tensor", "per_channel"])
@pytest.mark.parametrize("init_w_as_middle_points", [False, True], ids=["", "middle_points"])
@pytest.mark.parametrize("narrow_range", [False, True], ids=["full_range", "narrow_range"])
class TestQuantizedWeightsEqualAfterFixApplied:
    @pytest.mark.parametrize("signedness_to_force", [True, False], ids=["signed", "unsigned"])
    def test_symmetric_quantized_weights_equal_after_fix_applied(
        self, per_ch, signedness_to_force, init_w_as_middle_points, narrow_range
    ):
        qconfig = QuantizerConfig(
            num_bits=8, mode=QuantizationMode.SYMMETRIC, signedness_to_force=signedness_to_force, per_channel=per_ch
        )
        qspec = TFQuantizerSpec.from_config(qconfig, narrow_range=narrow_range, half_range=True)
        op_name = "quantizer"
        weight_attr = "kernel"

        layer = tf.keras.layers.Dense(DIM_SPLIT)
        layer = NNCFWrapper(layer)
        quantizer_cls = NNCF_QUANTIZATION_OPERATIONS.get(qspec.mode)
        quantizer = quantizer_cls(op_name, qspec)
        layer.registry_weight_operation(weight_attr, quantizer)
        layer.build(1)

        # Set layer weights
        ref_signed_var = -1 if signedness_to_force else 0
        ref_scale = 1
        low = ref_scale * ref_signed_var
        range_len = (1 - ref_signed_var) * ref_scale
        new_w = get_weights_for_overflow_issue_test(low, range_len, narrow_range, init_w_as_middle_points)
        layer.get_layer_weight(weight_attr).assign(new_w)

        # Check quantizer weights
        ops_weights = layer.ops_weights[op_name]
        assert (ops_weights["scale_var"].numpy() == ref_scale).all()
        assert (ops_weights["signed_var"].numpy() == ref_signed_var).all()

        w_int7 = layer(tf.ones((1, 1))).numpy()
        if init_w_as_middle_points:
            quant_len = range_len / (128 - (2 if narrow_range else 1))
            assert (np.abs(np.abs(w_int7 - new_w) - quant_len / 2) < 1e-6).all(), "Middle points calculated incorrectly"

        apply_overflow_fix_to_layer(layer, "kernel", quantizer)
        assert not quantizer._half_range
        w_int8 = layer(tf.ones((1, 1))).numpy()

        check_quantized_values_equals(w_int7, w_int8, EPS, range_len, narrow_range)

    @pytest.mark.parametrize(
        "low,range_len",
        [(-1, 2), (-5, 4), (3, 2)],
        ids=["zero_in_range", "max_less_than_zero", "low_greater_than_zero"],
    )
    def test_asymmetric_quantized_weights_equal_after_fix_applied(
        self, low, range_len, per_ch, init_w_as_middle_points, narrow_range
    ):
        qconfig = QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=per_ch)
        qspec = TFQuantizerSpec.from_config(qconfig, narrow_range=narrow_range, half_range=True)
        op_name = "quantizer"
        weight_attr = "kernel"

        layer = tf.keras.layers.Dense(DIM_SPLIT)
        layer = NNCFWrapper(layer)
        quantizer_cls = NNCF_QUANTIZATION_OPERATIONS.get(qspec.mode)
        quantizer = quantizer_cls(op_name, qspec)
        layer.registry_weight_operation(weight_attr, quantizer)
        layer.build(1)

        # Set layer weights
        new_w = get_weights_for_overflow_issue_test(low, range_len, narrow_range, init_w_as_middle_points)
        layer.get_layer_weight(weight_attr).assign(new_w)

        # Set quantizer weights
        if per_ch:
            low = tf.repeat(tf.constant([low], dtype=tf.float32), repeats=[DIM_SPLIT])
            range_len = tf.repeat(tf.constant([range_len], dtype=tf.float32), repeats=[DIM_SPLIT])

        ops_weights = layer.ops_weights[op_name]
        ops_weights["input_low_var"].assign(low)
        ops_weights["input_range_var"].assign(range_len)

        w_int7 = layer(tf.ones((1, 1))).numpy()
        if init_w_as_middle_points:
            quant_len = range_len / (128 - (2 if narrow_range else 1))
            assert (np.abs(np.abs(w_int7 - new_w) - quant_len / 2) < EPS).all(), "Middle points calculated incorrectly"

        apply_overflow_fix_to_layer(layer, "kernel", quantizer)
        assert not quantizer._half_range
        w_int8 = layer(tf.ones((1, 1))).numpy()

        check_quantized_values_equals(w_int7, w_int8, EPS, range_len, narrow_range)
