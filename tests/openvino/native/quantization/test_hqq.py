# Copyright (c) 2026 Intel Corporation
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

import nncf
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.hqq import HQQ
from nncf.quantization.algorithms.weight_compression.weight_lowering import calculate_integer_quantization_params
from nncf.quantization.algorithms.weight_compression.weight_lowering import (
    reshape_weight_for_grouped_quantization,
)
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor import functions as fns


def _make_weight(shape, seed=42, scale=10.0):
    """Create a deterministic float32 weight tensor."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape).astype(np.float32) * scale
    return Tensor(data)


def _quantization_error(weight: Tensor, scale: Tensor, zero_point: Tensor | None, config: WeightCompressionConfig, reduction_axes) -> float:
    """Compute mean squared quantization error: E[|W - s*(Q - z)|^2]."""
    group_size = config.group_size
    w = weight
    reduction = reduction_axes

    if group_size != -1:
        w, reduction = reshape_weight_for_grouped_quantization(weight, reduction_axes, group_size)

    q = w / scale
    if zero_point is not None:
        q = q + zero_point
    q = fns.round(q)

    num_bits = config.num_bits
    is_asym = config.is_asym_mode
    level_low = 0 if is_asym else -(2 ** (num_bits - 1))
    level_high = 2**num_bits - 1 if is_asym else 2 ** (num_bits - 1) - 1
    q = fns.clip(q, level_low, level_high)

    if zero_point is not None:
        reconstructed = scale * (q - zero_point)
    else:
        reconstructed = scale * q

    diff = w - reconstructed
    return float(fns.mean(diff * diff).data)


@pytest.mark.parametrize("mode,group_size,reduction_axes", [
    (CompressWeightsMode.INT4_ASYM, 16, 1),
    (CompressWeightsMode.INT4_SYM, 16, 1),
    (CompressWeightsMode.INT4_ASYM, -1, 1),
])
def test_hqq_reduces_quantization_error(mode, group_size, reduction_axes):
    """HQQ-optimized params should produce <= quantization error than min-max init."""
    weight = _make_weight((32, 64), seed=7)
    config = WeightCompressionConfig(mode=mode, group_size=group_size)

    hqq = HQQ(num_iterations=20)
    hqq_scale, hqq_zp = hqq._calculate_hqq_params(weight, config, reduction_axes)

    # Baseline: standard min-max initialization
    w = weight
    r = reduction_axes
    if group_size != -1:
        w, r = reshape_weight_for_grouped_quantization(weight, reduction_axes, group_size)
    baseline_scale, baseline_zp = calculate_integer_quantization_params(w, r, config)

    hqq_err = _quantization_error(weight, hqq_scale, hqq_zp, config, reduction_axes)
    baseline_err = _quantization_error(weight, baseline_scale, baseline_zp, config, reduction_axes)

    assert hqq_err <= baseline_err + 1e-6, (
        f"HQQ error ({hqq_err:.6f}) should not exceed min-max error ({baseline_err:.6f}) "
        f"for mode={mode.value}, group_size={group_size}"
    )


def test_hqq_asymmetric_zero_point_rounded():
    """HQQ should return an integer-valued zero point for use with uint4 storage.

    HQQ optimizes z as a continuous float during iterations, but the final value is
    rounded and clipped so that quantization and dequantization use the same integer z.
    The tensor dtype stays float32 (no cast), but all values should be integer-valued.
    """
    weight = _make_weight((32, 64), seed=13, scale=5.0)
    config = WeightCompressionConfig(mode=CompressWeightsMode.INT4_ASYM, group_size=16)

    hqq = HQQ(num_iterations=20)
    _, zero_point = hqq._calculate_hqq_params(weight, config, reduction_axes=1)

    assert zero_point is not None, "Expected non-None zero_point for asymmetric mode"

    zp_np = zero_point.data
    # dtype remains float32 (no explicit cast); values are integer-valued after rounding.
    assert zp_np.dtype == np.float32, f"Expected float32 zero point, got {zp_np.dtype}"
    assert np.allclose(zp_np, np.round(zp_np), atol=1e-5), (
        "HQQ zero point should be integer-valued after rounding for consistent uint4 storage"
    )


def test_hqq_symmetric_no_zero_point():
    """For symmetric modes HQQ should return None as zero point."""
    weight = _make_weight((32, 64), seed=17)
    config = WeightCompressionConfig(mode=CompressWeightsMode.INT4_SYM, group_size=16)

    hqq = HQQ(num_iterations=20)
    _, zero_point = hqq._calculate_hqq_params(weight, config, reduction_axes=1)

    assert zero_point is None, "Expected None zero_point for symmetric mode"


@pytest.mark.parametrize("num_iterations", [0, 1, 5, 20])
def test_hqq_num_iterations_parameter(num_iterations):
    """HQQ should be callable with various num_iterations values including zero."""
    weight = _make_weight((16, 32), seed=3)
    config = WeightCompressionConfig(mode=CompressWeightsMode.INT4_ASYM, group_size=16)

    hqq = HQQ(num_iterations=num_iterations)
    scale, zero_point = hqq._calculate_hqq_params(weight, config, reduction_axes=1)

    assert scale is not None
    assert zero_point is not None
    assert scale.shape == zero_point.shape


def test_hqq_advanced_parameters_exposed():
    """AdvancedHQQParameters must be importable from the nncf public namespace."""
    params = nncf.AdvancedHQQParameters(num_iterations=10)
    assert params.num_iterations == 10


def test_hqq_gptq_mutual_exclusion():
    """Specifying both hqq=True and gptq=True should raise ParameterNotSupportedError."""
    from nncf.quantization.algorithms.weight_compression.algorithm import check_user_compression_configuration

    with pytest.raises(nncf.ParameterNotSupportedError, match="HQQ and GPTQ"):
        check_user_compression_configuration(
            mode=CompressWeightsMode.INT4_ASYM,
            subset_size=128,
            dataset=None,
            ratio=1.0,
            group_size=128,
            all_layers=None,
            awq=None,
            scale_estimation=None,
            gptq=True,
            lora_correction=None,
            hqq=True,
            ignored_scope=None,
            sensitivity_metric=None,
            backup_mode=None,
            compression_format=None,
            advanced_parameters=None,
        )


def test_hqq_int8_unsupported():
    """HQQ should not be accepted for INT8 modes."""
    from nncf.quantization.algorithms.weight_compression.algorithm import check_user_compression_configuration

    with pytest.raises(nncf.ParameterNotSupportedError, match="hqq"):
        check_user_compression_configuration(
            mode=CompressWeightsMode.INT8_ASYM,
            subset_size=128,
            dataset=None,
            ratio=None,
            group_size=None,
            all_layers=None,
            awq=None,
            scale_estimation=None,
            gptq=None,
            lora_correction=None,
            hqq=True,
            ignored_scope=None,
            sensitivity_metric=None,
            backup_mode=None,
            compression_format=None,
            advanced_parameters=None,
        )
