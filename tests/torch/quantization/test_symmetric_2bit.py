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
"""
Tests for symmetric quantization at 2-bit (and other bit-widths).

QuantizeSymmetric and QuantizeSymmetricTorch use different scale conventions:

  QuantizeSymmetric:
      scale = level_high × step_size   (max representable positive value)
      input_low  = scale * (level_low / level_high)
      input_range = scale - input_low

  QuantizeSymmetricTorch (signed-scale, from _calculate_signed_scale):
      scale = ±|level_low| × step_size   (sign selects which side gets more quants)
      input_low differs by sign of scale  (flips asymmetry)
      input_range = (levels - 1) * |scale| / |level_low|

Both are correct; they just interpret `scale` differently.
QuantizeSymmetricTorch's signed-scale design allocates more quants to the
dominant side (positive or negative), which is especially important at low
bit-widths where a single extra level matters.
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 2-bit signed: levels=4, level_low=-2, level_high=1
PARAMS_2BIT = dict(level_low=-2, level_high=1, levels=4)
# 4-bit signed: levels=16, level_low=-8, level_high=7
PARAMS_4BIT = dict(level_low=-8, level_high=7, levels=16)
# 8-bit signed: levels=256, level_low=-128, level_high=127
PARAMS_8BIT = dict(level_low=-128, level_high=127, levels=256)


def _step_size(scale_val: float, level_low: int) -> float:
    """Compute the step size for QuantizeSymmetricTorch's scale convention."""
    return abs(scale_val) / (-level_low)


def _expected_levels(scale_val: float, level_low: int, level_high: int, **_) -> list[float]:
    """
    Compute expected dequantized output levels for QuantizeSymmetricTorch.

    For scale > 0:  levels are {level_low, …, level_high} * step, more quants on negative side.
    For scale < 0:  levels are {-level_high, …, -level_low} * step, more quants on positive side.
    """
    step = _step_size(scale_val, level_low)
    if scale_val > 0:
        return [i * step for i in range(level_low, level_high + 1)]
    return [i * step for i in range(-level_high, -level_low + 1)]


def _run_quantize_symmetric(input_tensor, scale, level_low, level_high, levels):
    """Run QuantizeSymmetric (original, CUDA/CPU extension)."""
    from nncf.torch.quantization.quantize_functions import QuantizeSymmetric

    return QuantizeSymmetric.apply(input_tensor, scale, level_low, level_high, levels)


def _run_quantize_symmetric_torch(input_tensor, scale, level_low, level_high, levels):
    """Run QuantizeSymmetricTorch (pure-torch, used for LoRA)."""
    from nncf.torch.quantization.quantize_functions import QuantizeSymmetricTorch

    input_shape = input_tensor.shape
    return QuantizeSymmetricTorch.apply(input_tensor, input_shape, scale, level_low, level_high, levels)


# ---------------------------------------------------------------------------
# Tests: QuantizeSymmetric (baseline) at 2-bit
# ---------------------------------------------------------------------------


class TestQuantizeSymmetric2Bit:
    """Verify QuantizeSymmetric (original) is correct at 2-bit."""

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            (-3.0, -2.0),
            (-2.0, -2.0),
            (-1.5, -2.0),
            (-1.0, -1.0),
            (-0.5, 0.0),
            (0.0, 0.0),
            (0.5, 0.0),
            (1.0, 1.0),
            (1.5, 1.0),
        ],
    )
    def test_scale_1(self, input_val, expected):
        scale = torch.tensor(1.0)
        x = torch.tensor(input_val)
        out = _run_quantize_symmetric(x, scale, **PARAMS_2BIT)
        assert out.item() == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# Tests: QuantizeSymmetricTorch at 2-bit — correct with its own scale convention
# ---------------------------------------------------------------------------


class TestQuantizeSymmetricTorch2Bit:
    """
    QuantizeSymmetricTorch at 2-bit with the signed-scale convention.

    scale is produced by _calculate_signed_scale: scale = max_abs / |level_low|
    step_size = |scale| / |level_low|

    For 2-bit (level_low=-2, level_high=1) with scale=1.0 (positive -> negatives dominate):
        step = 1.0/2 = 0.5
        range: [-scale, level_high/|level_low| * scale] = [-1.0, 0.5]
        output levels: {-2, -1, 0, 1} * 0.5 = {-1.0, -0.5, 0.0, 0.5}
        -> 2 levels negative, 1 level positive (more quants for negatives)
    """

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            # scale=1.0 positive -> step=0.5, range [-1.0, 0.5], output {-1.0, -0.5, 0.0, 0.5}
            (-2.0, -1.0),  # clipped to input_low
            (-1.0, -1.0),  # exact min output
            (-0.8, -1.0),  # rounds to -1.0
            (-0.6, -0.5),  # rounds to -0.5
            (-0.5, -0.5),  # exact
            (-0.25, 0.0),  # rounds to 0
            (0.0, 0.0),  # exact zero
            (0.25, 0.0),  # rounds to 0
            (0.5, 0.5),  # exact max output
            (1.0, 0.5),  # clipped to input_high
        ],
    )
    def test_positive_scale(self, input_val, expected):
        """Positive scale: more quants allocated to negatives."""
        scale = torch.tensor(1.0)
        x = torch.tensor([input_val])
        out = _run_quantize_symmetric_torch(x, scale, **PARAMS_2BIT)
        assert out.item() == pytest.approx(expected, abs=1e-5)

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            # scale=-1.0 negative -> step=0.5, range [-0.5, 1.0], output {-0.5, 0.0, 0.5, 1.0}
            (-1.0, -0.5),  # clipped to input_low
            (-0.5, -0.5),  # exact min output
            (-0.25, 0.0),  # rounds to 0
            (0.0, 0.0),  # exact zero
            (0.25, 0.0),  # rounds to 0
            (0.5, 0.5),  # exact
            (0.6, 0.5),  # rounds to 0.5
            (0.8, 1.0),  # rounds to 1.0
            (1.0, 1.0),  # exact max output
            (2.0, 1.0),  # clipped to input_high
        ],
    )
    def test_negative_scale(self, input_val, expected):
        """Negative scale: more quants allocated to positives."""
        scale = torch.tensor(-1.0)
        x = torch.tensor([input_val])
        out = _run_quantize_symmetric_torch(x, scale, **PARAMS_2BIT)
        assert out.item() == pytest.approx(expected, abs=1e-5)

    @pytest.mark.parametrize("scale_val", [0.5, 1.0, 2.0, -0.5, -1.0, -2.0])
    def test_output_has_exactly_4_levels(self, scale_val):
        """All outputs must land on exactly 4 quantization levels."""
        scale = torch.tensor(scale_val)
        x = torch.linspace(-5 * abs(scale_val), 5 * abs(scale_val), 1000)
        out = _run_quantize_symmetric_torch(x, scale, **PARAMS_2BIT)
        actual = sorted(set(round(v, 5) for v in out.tolist()))
        expected = sorted(round(v, 5) for v in _expected_levels(scale_val, **PARAMS_2BIT))
        assert actual == pytest.approx(expected, abs=1e-5)

    @pytest.mark.parametrize("scale_val", [0.5, 1.0, 2.0, -0.5, -1.0, -2.0])
    def test_zero_maps_to_zero(self, scale_val):
        """Zero input always maps to zero output."""
        scale = torch.tensor(scale_val)
        x = torch.tensor([0.0])
        out = _run_quantize_symmetric_torch(x, scale, **PARAMS_2BIT)
        assert out.item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Tests: QuantizeSymmetricTorch at 4-bit (signed-scale asymmetry)
# ---------------------------------------------------------------------------


class TestQuantizeSymmetricTorch4Bit:
    """
    4-bit with signed scale.  step = |scale| / 8.
    scale > 0: range [-scale, 7/8*scale], 8 neg + 7 pos levels
    scale < 0: range [-7/8*|scale|, |scale|], 7 neg + 8 pos levels
    """

    @pytest.mark.parametrize("scale_val", [0.5, 1.0, 2.0, -0.5, -1.0, -2.0])
    def test_output_has_exactly_16_levels(self, scale_val):
        scale = torch.tensor(scale_val)
        x = torch.linspace(-5 * abs(scale_val), 5 * abs(scale_val), 5000)
        out = _run_quantize_symmetric_torch(x, scale, **PARAMS_4BIT)
        actual = sorted(set(round(v, 6) for v in out.tolist()))
        expected = sorted(round(v, 6) for v in _expected_levels(scale_val, **PARAMS_4BIT))
        assert actual == pytest.approx(expected, abs=1e-5)

    @pytest.mark.parametrize("scale_val", [1.0, -1.0])
    def test_zero_maps_to_zero(self, scale_val):
        scale = torch.tensor(scale_val)
        x = torch.tensor([0.0])
        out = _run_quantize_symmetric_torch(x, scale, **PARAMS_4BIT)
        assert out.item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Tests: QuantizeSymmetricTorch at 8-bit
# ---------------------------------------------------------------------------


class TestQuantizeSymmetricTorch8Bit:
    """8-bit verification."""

    @pytest.mark.parametrize("scale_val", [0.5, 1.0, 2.0])
    def test_output_has_exactly_256_levels(self, scale_val):
        scale = torch.tensor(scale_val)
        x = torch.linspace(-2 * abs(scale_val), 2 * abs(scale_val), 50000)
        out = _run_quantize_symmetric_torch(x, scale, **PARAMS_8BIT)
        actual = sorted(set(round(v, 8) for v in out.tolist()))
        expected = sorted(round(v, 8) for v in _expected_levels(scale_val, **PARAMS_8BIT))
        assert actual == pytest.approx(expected, abs=1e-5)

    @pytest.mark.parametrize("scale_val", [0.5, 1.0, 2.0])
    def test_zero_maps_to_zero(self, scale_val):
        scale = torch.tensor(scale_val)
        x = torch.tensor([0.0])
        out = _run_quantize_symmetric_torch(x, scale, **PARAMS_8BIT)
        assert out.item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Tests: scale convention difference (documenting, not a bug)
# ---------------------------------------------------------------------------


class TestScaleConventionDifference:
    """
    QuantizeSymmetric and QuantizeSymmetricTorch use different scale conventions
    and thus produce different outputs for the same scale value.
    This is by design, not a bug.
    """

    def test_different_scale_semantics_2bit(self):
        """
        With scale=1.0 at 2-bit:
          QuantizeSymmetric:      step = scale / level_high = 1.0, levels = {-2, -1, 0, 1}
          QuantizeSymmetricTorch: step = |scale| / |level_low| = 0.5, levels = {-1, -0.5, 0, 0.5}
        """
        scale = torch.tensor(1.0)
        x = torch.linspace(-5, 5, 1000)

        out_qs = _run_quantize_symmetric(x, scale, **PARAMS_2BIT)
        qs_levels = sorted(set(round(v, 5) for v in out_qs.tolist()))

        out_qst = _run_quantize_symmetric_torch(x, scale, **PARAMS_2BIT)
        qst_levels = sorted(set(round(v, 5) for v in out_qst.tolist()))

        # Different step sizes, both have 4 levels
        assert len(qs_levels) == 4
        assert len(qst_levels) == 4
        assert qs_levels == pytest.approx([-2.0, -1.0, 0.0, 1.0], abs=1e-5)
        assert qst_levels == pytest.approx([-1.0, -0.5, 0.0, 0.5], abs=1e-5)

    def test_equivalent_when_scale_converted_2bit(self):
        """
        QuantizeSymmetricTorch with scale_torch = QuantizeSymmetric's scale * |level_low| / level_high
        should produce the same output levels (just different step).
        """
        qs_scale = torch.tensor(1.0)  # QuantizeSymmetric convention
        # Convert to QuantizeSymmetricTorch convention:
        # step = qs_scale / level_high = 1.0, qst_scale = step * |level_low| = 2.0
        qst_scale = qs_scale * (-PARAMS_2BIT["level_low"]) / PARAMS_2BIT["level_high"]

        x = torch.linspace(-3, 3, 500)
        out_qs = _run_quantize_symmetric(x, qs_scale, **PARAMS_2BIT)
        out_qst = _run_quantize_symmetric_torch(x, qst_scale, **PARAMS_2BIT)
        torch.testing.assert_close(out_qst, out_qs)


# ---------------------------------------------------------------------------
# Tests: formula internals
# ---------------------------------------------------------------------------


class TestFormulaInternals:
    """Verify the generalized input_low / input_range formulas."""

    @pytest.mark.parametrize(
        "bits, params",
        [
            (2, PARAMS_2BIT),
            (4, PARAMS_4BIT),
            (8, PARAMS_8BIT),
        ],
    )
    def test_input_range_formula(self, bits, params):
        """
        The generalized formula (levels-1)*|scale|/|level_low| should equal
        the old formula |( 2 + 1/level_low ) * scale| for all standard signed ranges.
        """
        scale = 1.0
        level_low = params["level_low"]
        levels = params["levels"]

        new_range = (levels - 1) * abs(scale) / (-level_low)
        old_range = abs((2 + 1 / level_low) * scale)
        assert new_range == pytest.approx(old_range, rel=1e-12)

    @pytest.mark.parametrize(
        "scale_val, level_low, level_high, levels",
        [
            (1.0, -2, 1, 4),  # 2-bit positive scale
            (-1.0, -2, 1, 4),  # 2-bit negative scale
            (1.0, -8, 7, 16),  # 4-bit positive scale
            (-1.0, -8, 7, 16),  # 4-bit negative scale
        ],
    )
    def test_input_low_and_range_consistency(self, scale_val, level_low, level_high, levels):
        """
        input_low + input_range should map to the correct upper bound.
        """
        scale = torch.tensor(scale_val)
        input_low = torch.where(scale > 0, -scale, -scale / level_low * level_high)
        input_range = (levels - 1) * torch.abs(scale) / (-level_low)
        upper = input_low + input_range

        step = abs(scale_val) / (-level_low)
        if scale_val > 0:
            assert input_low.item() == pytest.approx(level_low * step, abs=1e-10)
            assert upper.item() == pytest.approx(level_high * step, abs=1e-10)
        else:
            # Flipped: more quants on positive side
            assert input_low.item() == pytest.approx(-level_high * step, abs=1e-10)
            assert upper.item() == pytest.approx(-level_low * step, abs=1e-10)


# ---------------------------------------------------------------------------
# Tests: gradient flow (QuantizeSymmetricTorch backward)
# ---------------------------------------------------------------------------


class TestQuantizeSymmetricTorchGradient:
    """Ensure backward pass runs and produces non-zero gradients."""

    @pytest.mark.parametrize("bits_params", [PARAMS_2BIT, PARAMS_4BIT, PARAMS_8BIT])
    def test_gradient_flows(self, bits_params):
        scale = torch.tensor(1.0, requires_grad=True)
        x = torch.randn(10, requires_grad=True)
        out = _run_quantize_symmetric_torch(x, scale, **bits_params)
        loss = out.sum()
        loss.backward()
        assert scale.grad is not None
        assert x.grad is not None

    def test_gradient_with_negative_scale_2bit(self):
        scale = torch.tensor(-1.0, requires_grad=True)
        x = torch.randn(10, requires_grad=True)
        out = _run_quantize_symmetric_torch(x, scale, **PARAMS_2BIT)
        loss = out.sum()
        loss.backward()
        assert scale.grad is not None
        assert x.grad is not None
