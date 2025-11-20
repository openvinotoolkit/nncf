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


def _f16_to_f8e4m3_bits_scalar(h_bits: int) -> int:
    """Exact port of ov::f16_to_f8e4m3_bits for a single float16 bit-pattern."""
    # f16 layout
    f16_s_mask = 0x8000
    f16_e_mask = 0x7C00
    f16_e_bias = 15
    f16_e_size = 5
    f16_m_mask = 0x03FF
    f16_m_size = 10

    # f8 e4m3 layout
    f8e4m3_e_size = 4
    f8e4m3_e_mask = 0x78
    f8e4m3_e_bias = 7
    f8e4m3_e_max = 0x0F
    f8e4m3_m_size = 3
    f8e4m3_m_mask = 0x07

    byte_shift = 8

    # f8 masks in uint16 domain
    f8_e_mask = f8e4m3_e_mask << byte_shift  # 0x7800
    f8_m_mask = f8e4m3_m_mask << byte_shift  # 0x0700
    f8_m_hidden_one_mask = 0x0800  # hidden 1 for subnormals

    # rounding constants (same as C++)
    round_half = 0x01FF
    round_norm = 0x007F
    round_even = 0x0080
    round_odd = 0x0180

    # min exponent for which subnormals are representable
    f8_e_subnormal_min = -10

    inp = int(h_bits) & 0xFFFF

    # sign bit: f16 sign -> f8 sign position (bit 15 -> bit 7)
    f8_bits = (inp & f16_s_mask) >> byte_shift

    f16_e_field = inp & f16_e_mask

    if f16_e_field == f16_e_mask:
        # f16 NaN / Inf -> f8 NaN (no Inf)
        f8_bits |= f8e4m3_e_mask | f8e4m3_m_mask
    elif f16_e_field != 0:
        # normalized f16
        f8_biased_exp = (f16_e_field >> f16_m_size) - (f16_e_bias - f8e4m3_e_bias)
        # *** IMPORTANT FIX: shift by (f16_e_size - f8e4m3_e_size) = 5 - 4 = 1 ***
        fractional = (inp & f16_m_mask) << (f16_e_size - f8e4m3_e_size)

        # normalized f8 part (exp >= 0)
        if f8_biased_exp >= 0:
            if (fractional & round_half) == round_odd or (fractional & round_norm) != 0:
                fractional += round_even
                if (fractional & f8_e_mask) != 0:
                    f8_biased_exp += 1
            fractional &= f8_m_mask

        # now set exponent & mantissa
        if f8_biased_exp > f8e4m3_e_max:
            # overflow -> NaN (no Inf)
            f8_bits |= f8e4m3_e_mask | f8e4m3_m_mask
        elif f8_biased_exp > 0:
            # normalized f8
            exp_field = (f8_biased_exp & (f8e4m3_e_mask >> f8e4m3_m_size)) << f8e4m3_m_size
            f8_bits |= exp_field
            f8_bits |= fractional >> byte_shift
        else:
            # subnormal f8
            fractional = f8_m_hidden_one_mask | ((inp & f16_m_mask) << (f16_e_size - f8e4m3_e_size))
            f8_exp = f8_biased_exp - f8e4m3_e_bias
            shift = 1 - f8_exp
            sticky_mask = 0 if f8_exp < f8_e_subnormal_min else ((1 << shift) - 1)
            sticky = 1 if (fractional & sticky_mask) != 0 else 0

            fractional = 0 if f8_exp < f8_e_subnormal_min else (fractional >> (1 - f8_biased_exp))

            if (
                ((fractional & round_half) == round_odd and sticky == 0)
                or (fractional & round_norm) != 0
                or sticky != 0
            ):
                fractional += round_even

            f8_bits |= fractional >> byte_shift
    else:
        # f16 zero / subnormal -> sign + zero exponent/mantissa
        # (f8_bits already contains the sign)
        pass

    return f8_bits & 0xFF


_f16_to_f8e4m3_bits_vec = np.vectorize(_f16_to_f8e4m3_bits_scalar, otypes=[np.uint8])


def fp32_to_fp8e4m3_values(x: np.ndarray) -> np.ndarray:
    """
    Bit-exact to ov::float8_e4m3(float):
        float32 -> float16 -> f8e4m3 bits -> float via LUT
    """
    x = np.asarray(x, dtype=np.float32)
    x_f16 = x.astype(np.float16)
    h_bits = x_f16.view(np.uint16)

    f8_bits = _f16_to_f8e4m3_bits_vec(h_bits)

    # Decode exactly like C++: LUT for magnitude + sign bit
    idx = f8_bits & 0x7F
    mag = F8E4M3_LUT[idx.astype(np.int32)]

    sign = np.where((f8_bits & 0x80) != 0, -1.0, 1.0)
    out = sign * mag
    return out.astype(np.float32)
