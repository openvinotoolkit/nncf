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

# fmt: off
F8E4M3_LUT = np.array(
    [
        0.0,      0.001953125, 0.00390625, 0.005859375, 0.0078125, 0.009765625, 0.01171875, 0.013671875,    # noqa
        0.015625, 0.017578125, 0.01953125, 0.021484375, 0.0234375, 0.025390625, 0.02734375, 0.029296875,    # noqa
        0.03125,  0.03515625,  0.0390625,  0.04296875,  0.046875,  0.05078125,  0.0546875,  0.05859375,     # noqa
        0.0625,   0.0703125,   0.078125,   0.0859375,   0.09375,   0.1015625,   0.109375,   0.1171875,     # noqa
        0.125,    0.140625,    0.15625,    0.171875,    0.1875,    0.203125,    0.21875,    0.234375,     # noqa
        0.25,     0.28125,     0.3125,     0.34375,     0.375,     0.40625,     0.4375,     0.46875,     # noqa
        0.5,      0.5625,      0.625,      0.6875,      0.75,      0.8125,      0.875,      0.9375,     # noqa
        1.0,      1.125,       1.25,       1.375,       1.5,       1.625,       1.75,       1.875,     # noqa
        2.0,      2.25,        2.5,        2.75,        3.0,       3.25,        3.5,        3.75,     # noqa
        4.0,      4.5,         5.0,        5.5,         6.0,       6.5,         7.0,        7.5,     # noqa
        8.0,      9.0,         10.0,       11.0,        12.0,      13.0,        14.0,       15.0,     # noqa
        16.0,     18.0,        20.0,       22.0,        24.0,      26.0,        28.0,       30.0,     # noqa
        32.0,     36.0,        40.0,       44.0,        48.0,      52.0,        56.0,       60.0,     # noqa
        64.0,     72.0,        80.0,       88.0,        96.0,      104.0,       112.0,      120.0,     # noqa
        128.0,    144.0,       160.0,      176.0,       192.0,     208.0,       224.0,      240.0,     # noqa
        256.0,    288.0,       320.0,      352.0,       384.0,     416.0,       448.0,      np.nan,     # noqa
    ],
    dtype=np.float32,
)
# fmt: on


def f16_to_f8e4m3_bits_numpy(x: np.ndarray) -> np.ndarray:
    """
    Convert an array of f16 values (or their uint16 bit patterns) to
    f8e4m3 bit patterns (uint8) using a fully vectorized NumPy
    port of _f16_to_f8e4m3_bits_scalar.
    """
    # f16 layout
    f16_s_mask = np.uint16(0x8000)
    f16_e_mask = np.uint16(0x7C00)
    f16_e_bias = 15
    f16_e_size = 5
    f16_m_mask = np.uint16(0x03FF)
    f16_m_size = 10

    # f8 e4m3 layout
    f8e4m3_e_size = 4
    f8e4m3_e_mask = np.uint16(0x78)
    f8e4m3_e_bias = 7
    f8e4m3_e_max = 0x0F
    f8e4m3_m_size = 3
    f8e4m3_m_mask = np.uint16(0x07)

    byte_shift = 8

    # f8 masks in uint16 domain
    f8_e_mask = np.uint16(f8e4m3_e_mask << byte_shift)  # 0x7800
    f8_m_mask = np.uint16(f8e4m3_m_mask << byte_shift)  # 0x0700
    f8_m_hidden_one_mask = np.uint16(0x0800)  # hidden 1 for subnormals

    # rounding constants
    round_half = np.uint16(0x01FF)
    round_norm = np.uint16(0x007F)
    round_even = np.uint16(0x0080)
    round_odd = np.uint16(0x0180)

    # min exponent for which subnormals are representable
    f8_e_subnormal_min = -10

    # sign bit: f16 sign -> f8 sign position (bit 15 -> bit 7)
    f8_bits = ((x & f16_s_mask) >> byte_shift).astype(np.uint16)

    f16_e_field = x & f16_e_mask
    is_naninf = f16_e_field == f16_e_mask
    is_zero = f16_e_field == 0
    is_normal = (~is_naninf) & (~is_zero)

    nan_pattern = np.uint16(f8e4m3_e_mask | f8e4m3_m_mask)

    # --- Case 1: f16 NaN / Inf -> f8 NaN (no Inf) ---
    f8_bits = np.where(is_naninf, f8_bits | nan_pattern, f8_bits)

    # --- Case 2: normalized f16 ---
    # f8_biased_exp = (f16_e_field >> f16_m_size) - (f16_e_bias - f8e4m3_e_bias)
    f8_biased_exp = (f16_e_field >> f16_m_size).astype(np.int32) - (f16_e_bias - f8e4m3_e_bias)

    # fractional = (inp & f16_m_mask) << (f16_e_size - f8e4m3_e_size)
    fractional_norm = ((x & f16_m_mask) << (f16_e_size - f8e4m3_e_size)).astype(np.uint16)

    exp_ge0 = (f8_biased_exp >= 0) & is_normal

    # Rounding for normalized part (exp >= 0)
    # if (fractional & round_half) == round_odd or (fractional & round_norm) != 0:
    cond_round_norm = (((fractional_norm & round_half) == round_odd) | ((fractional_norm & round_norm) != 0)) & exp_ge0

    # fractional += round_even where cond_round_norm
    frac_tmp = fractional_norm.astype(np.uint32) + np.where(cond_round_norm, round_even, np.uint16(0)).astype(np.uint32)
    fractional_norm = (frac_tmp & 0xFFFF).astype(np.uint16)

    # if (fractional & f8_e_mask) != 0: f8_biased_exp += 1
    exp_inc = np.where(exp_ge0 & ((fractional_norm & f8_e_mask) != 0), 1, 0).astype(np.int32)
    f8_biased_exp_after = f8_biased_exp + exp_inc

    # fractional &= f8_m_mask
    fractional_norm &= f8_m_mask

    # Overflow / normalized / subnormal classification
    overflow_mask = is_normal & (f8_biased_exp_after > f8e4m3_e_max)
    normal_mask = is_normal & (f8_biased_exp_after > 0) & (~overflow_mask)
    # For subnormals, the scalar code uses f8_biased_exp (after possible increment),
    # but increment is only applied when exp >= 0, so exp <= 0 path is unchanged.
    subnormal_mask = is_normal & (f8_biased_exp_after <= 0) & (~overflow_mask)

    # --- Overflow -> NaN ---
    f8_bits = np.where(overflow_mask, f8_bits | nan_pattern, f8_bits)

    # --- Normalized f8 ---
    # exp_field = (f8_biased_exp & (f8e4m3_e_mask >> f8e4m3_m_size)) << f8e4m3_m_size
    exp_field = ((f8_biased_exp_after & (f8e4m3_e_mask >> f8e4m3_m_size)) << f8e4m3_m_size).astype(np.uint16)
    mant_norm = (fractional_norm >> byte_shift).astype(np.uint16)

    f8_bits_norm = f8_bits | exp_field | mant_norm
    f8_bits = np.where(normal_mask, f8_bits_norm, f8_bits)

    # --- Subnormal f8 ---
    # fractional = f8_m_hidden_one_mask | ((inp & f16_m_mask) << (f16_e_size - f8e4m3_e_size))
    fractional_sub = f8_m_hidden_one_mask | ((x & f16_m_mask) << (f16_e_size - f8e4m3_e_size))

    # f8_exp = f8_biased_exp - f8e4m3_e_bias
    f8_exp = (f8_biased_exp_after - f8e4m3_e_bias).astype(np.int32)

    # shift = 1 - f8_exp
    shift = 1 - f8_exp

    # sticky_mask = 0 if f8_exp < f8_e_subnormal_min else ((1 << shift) - 1)
    # we avoid invalid shifts by clipping / masking
    valid_sub = f8_exp >= f8_e_subnormal_min
    shift_pos = np.maximum(shift, 0)
    sticky_mask32 = np.where(valid_sub, (np.uint32(1) << shift_pos) - 1, 0).astype(np.uint32)
    sticky_mask16 = (sticky_mask32 & np.uint32(0xFFFF)).astype(np.uint16)

    # sticky = 1 if (fractional & sticky_mask) != 0 else 0
    sticky = ((fractional_sub & sticky_mask16) != 0) & valid_sub

    # fractional = 0 if f8_exp < f8_e_subnormal_min else (fractional >> (1 - f8_biased_exp))
    shift2 = 1 - f8_biased_exp_after
    shift2_pos = np.maximum(shift2, 0)
    frac_shifted = (fractional_sub.astype(np.uint32) >> shift2_pos).astype(np.uint16)
    frac_shifted = np.where(valid_sub, frac_shifted, np.uint16(0))

    # Rounding for subnormal:
    # if (((fractional & round_half) == round_odd and sticky == 0)
    #     or (fractional & round_norm) != 0
    #     or sticky != 0):
    cond_round_sub = (
        (((frac_shifted & round_half) == round_odd) & (~sticky)) | ((frac_shifted & round_norm) != 0) | sticky
    ) & subnormal_mask

    frac_tmp_sub = frac_shifted.astype(np.uint32) + np.where(cond_round_sub, round_even, np.uint16(0)).astype(np.uint32)
    fractional_sub_final = (frac_tmp_sub & 0xFFFF).astype(np.uint16)

    mant_sub = (fractional_sub_final >> byte_shift).astype(np.uint16)
    f8_bits = np.where(subnormal_mask, f8_bits | mant_sub, f8_bits)

    # Case: f16 zero / subnormal -> sign + zero exponent/mantissa
    # Already handled by initialization + not touching zero_mask entries.

    return (f8_bits & np.uint16(0x00FF)).astype(np.uint8)


def fp32_to_fp8e4m3(x: np.ndarray) -> np.ndarray:
    """
    Bit-exact to ov::float8_e4m3(float):
        float32 -> float16 -> f8e4m3 bits -> float via LUT
    """
    x = np.asarray(x, dtype=np.float32)
    x_f16 = x.astype(np.float16)
    h_bits = x_f16.view(np.uint16)

    f8_bits = f16_to_f8e4m3_bits_numpy(h_bits)

    # Decode exactly like C++: LUT for magnitude + sign bit
    idx = f8_bits & 0x7F
    mag = F8E4M3_LUT[idx.astype(np.int32)]

    sign = np.where((f8_bits & 0x80) != 0, -1.0, 1.0)
    out = sign * mag
    return out.astype(np.float32)
