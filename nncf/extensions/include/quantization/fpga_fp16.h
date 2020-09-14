// Copyright 2015-2017 Intel Corporation.
//
// The source code, information and material ("Material") contained herein is
// owned by Intel Corporation or its suppliers or licensors, and title to such
// Material remains with Intel Corporation or its suppliers or licensors. The
// Material contains proprietary information of Intel or its suppliers and
// licensors. The Material is protected by worldwide copyright laws and treaty
// provisions. No part of the Material may be used, copied, reproduced,
// modified, published, uploaded, posted, transmitted, distributed or disclosed
// in any way without Intel's prior express written permission. No license under
// any patent, copyright or other intellectual property rights in the Material
// is granted to or conferred upon you, either expressly, by implication,
// inducement, estoppel or otherwise. Any license under such intellectual
// property rights must be express and approved by Intel in writing.
//
// Unless otherwise agreed by Intel in writing, you may not remove or alter this
// notice or any other notice embedded in Materials by Intel or Intel's
// suppliers or licensors in any way.

#ifndef __FPGA_FP16_H__
#define __FPGA_FP16_H__
// -------------------------------------------------------------------------- //
// fpga_fp16.h:
//
// The purpose of this file is to define the "half_t" type, which is our
// version of the "half" type that's part of OpenCL. The half_t type itself is
// just a struct with an unsigned short inside. We also create functions for
// the basic operations that you can do on the half type, and switch between
// the native implementation and an OpenCL C implementation depending on macro
// settings.
//
// We use this instead of half directly for two reasons:
//   1. Neither the host nor the emulator support the half type.
//   2. We can support the DISABLE_SUBNORM_FLOATS macro, which disables support
//      for subnormal values in order to save area.
// -------------------------------------------------------------------------- //

#include "fpga_common_utils.h"
// FP32_EXP_BIAS (=127) - FP16_EXP_BIAS(=15)
#define FP32_EXPONENT_BIAS_DIFF (112)

SAFE_TYPEDEF(ushort, half_t);

CONSTANT half_t zero_half = SAFE_TYPEDEF_ZERO;

#if defined(INTELFPGA_CL) && !defined(EMULATOR)
#define NATIVE_FP16_AVAILABLE
#endif

// Conversion functions to convert between half_t and ushort. These functions
// are named "_as_" instead of "_to_" like all the others, because they simply
// reintrepret the raw binary data rather than convert the value. This makes
// them more like the "as_half" and "as_ushort" OpenCL functions. For example,
// ushort_as_half(1) does not product a half equal to 1.0.
STATIC half_t ushort_as_half(ushort val) {
  half_t res;
  SAFE_TYPEDEF_VAL(res) = val;
  return res;
}

STATIC ushort half_as_ushort(half_t val) {
  return SAFE_TYPEDEF_VAL(val);
}

// conversion functions to convert between half_t and half
#ifdef INTELFPGA_CL
STATIC half half_to_real_half(half_t val) {
  // TODO: replace this with an as_half() call when the compiler supports it
  // (see case 355848)
  union {
    ushort ushort_val;
    half half_val;
  } temp_union;
  temp_union.ushort_val = half_as_ushort(val);
  return temp_union.half_val;
}

STATIC half_t real_half_to_half(half val) {
  return ushort_as_half(as_ushort(val));
}
#endif

STATIC float half_to_float(half_t orig_half)
{
#if !defined(NATIVE_FP16_AVAILABLE) || defined(DISABLE_NATIVE_FP16_CAST)

    // This implementation was written to be essentially exactly the same as the
    // Verilog implementation that's part of the compiler. I used that Verilog as
    // a starting point and re-wrote it in OpenCL C.
    ushort orig_half_raw = half_as_ushort(orig_half);
    uint sign = BIT_SEL(orig_half_raw, 15, 15);
    uint exponent = BIT_SEL(orig_half_raw, 14, 10);
    uint mantissa = BIT_SEL(orig_half_raw, 9, 0);

#if defined(DISABLE_SUBNORM_FLOATS)
    if (exponent == 0) {
        mantissa = 0;
    }
    else {
        mantissa <<= 13;
        exponent += FP32_EXPONENT_BIAS_DIFF;
    }
#else // Subnormal support
    // Examples for leading 1 count
    // 10 0000 0000 => 0
    // 01 0000 0000 => 1
    // 00 1000 0000 => 2
    if (exponent == 0) {
        int count_bit_3 = BIT_SEL(mantissa, 9, 2) == 0;
        int count_bit_2 = (BIT_SEL(mantissa, 9, 2) != 0 && BIT_SEL(mantissa, 9, 6) == 0);
        int count_bit_1 =
            (count_bit_3 && BIT_SEL(mantissa, 1, 0) == 0) ||
            (!count_bit_3 &&
            ((!count_bit_2 && BIT_SEL(mantissa, 9, 8) == 0) || (count_bit_2 && BIT_SEL(mantissa, 5, 4) == 0)));
        int count_bit_0 =
            (BIT_SEL(mantissa, 9, 8) == 1) ||
            (BIT_SEL(mantissa, 9, 6) == 1) ||
            (BIT_SEL(mantissa, 9, 4) == 1) ||
            (BIT_SEL(mantissa, 9, 2) == 1) ||
            (BIT_SEL(mantissa, 9, 0) == 1);
        int count =
            (count_bit_3 << 3) | (count_bit_2 << 2) |
            (count_bit_1 << 1) | (count_bit_0 << 0);

        int shifted_mantissa = BIT_SEL(mantissa << count, 8, 0);
        mantissa = shifted_mantissa << 14;

        exponent = FP32_EXPONENT_BIAS_DIFF - count;
    }
    else {
        mantissa = mantissa << 13;
        exponent = FP32_EXPONENT_BIAS_DIFF + exponent;
    }
#endif

    uint result = (sign << 31) | (exponent << 23) | (mantissa << 0);
    return as_float(result);

#else
    return (float)half_to_real_half(orig_half);
#endif
}

STATIC float half_to_float_disable_subnorms(half_t orig_half)
{
#if !defined(NATIVE_FP16_AVAILABLE) || defined(DISABLE_NATIVE_FP16_CAST)

    // This implementation was written to be essentially exactly the same as the
    // Verilog implementation that's part of the compiler. I used that Verilog as
    // a starting point and re-wrote it in OpenCL C.
    ushort orig_half_raw = half_as_ushort(orig_half);
    uint sign = BIT_SEL(orig_half_raw, 15, 15);
    uint exponent = BIT_SEL(orig_half_raw, 14, 10);
    uint mantissa = BIT_SEL(orig_half_raw, 9, 0);

    if (exponent == 0) {
        mantissa = 0;
    }
    else {
        mantissa <<= 13;
        exponent += FP32_EXPONENT_BIAS_DIFF;
    }

    uint result = (sign << 31) | (exponent << 23) | (mantissa << 0);
    return as_float(result);

#else
    return (float)half_to_real_half(orig_half);
#endif
}


STATIC half_t float_to_half(float orig_float)
{
#if !defined(NATIVE_FP16_AVAILABLE) || defined(DISABLE_NATIVE_FP16_CAST) 

  // This implementation was written to be essentially exactly the same as the
  // Verilog implementation that's part of the compiler. I used that Verilog as
  // a starting point and re-wrote it in OpenCL C.
  uint orig_float_raw = as_uint(orig_float);
  uint sign     = BIT_SEL(orig_float_raw, 31, 31);
  uint exponent = BIT_SEL(orig_float_raw, 30, 23);
  uint mantissa = BIT_SEL(orig_float_raw, 22, 0);

  if (exponent <= FP32_EXPONENT_BIAS_DIFF) {
#if defined(DISABLE_SUBNORM_FLOATS)
    mantissa = 0;
#else
    switch (exponent) {
      case (FP32_EXPONENT_BIAS_DIFF-0) : mantissa = (1 << 12) | (BIT_SEL(mantissa, 22, 12) << 1) | (BIT_SEL(mantissa, 11, 0) != 0); break;
      case (FP32_EXPONENT_BIAS_DIFF-1) : mantissa = (1 << 11) | (BIT_SEL(mantissa, 22, 13) << 1) | (BIT_SEL(mantissa, 12, 0) != 0); break;
      case (FP32_EXPONENT_BIAS_DIFF-2) : mantissa = (1 << 10) | (BIT_SEL(mantissa, 22, 14) << 1) | (BIT_SEL(mantissa, 13, 0) != 0); break;
      case (FP32_EXPONENT_BIAS_DIFF-3) : mantissa = (1 <<  9) | (BIT_SEL(mantissa, 22, 15) << 1) | (BIT_SEL(mantissa, 14, 0) != 0); break;
      case (FP32_EXPONENT_BIAS_DIFF-4) : mantissa = (1 <<  8) | (BIT_SEL(mantissa, 22, 16) << 1) | (BIT_SEL(mantissa, 15, 0) != 0); break;
      case (FP32_EXPONENT_BIAS_DIFF-5) : mantissa = (1 <<  7) | (BIT_SEL(mantissa, 22, 17) << 1) | (BIT_SEL(mantissa, 16, 0) != 0); break;
      case (FP32_EXPONENT_BIAS_DIFF-6) : mantissa = (1 <<  6) | (BIT_SEL(mantissa, 22, 18) << 1) | (BIT_SEL(mantissa, 17, 0) != 0); break;
      case (FP32_EXPONENT_BIAS_DIFF-7) : mantissa = (1 <<  5) | (BIT_SEL(mantissa, 22, 19) << 1) | (BIT_SEL(mantissa, 18, 0) != 0); break;
      case (FP32_EXPONENT_BIAS_DIFF-8) : mantissa = (1 <<  4) | (BIT_SEL(mantissa, 22, 20) << 1) | (BIT_SEL(mantissa, 19, 0) != 0); break;
      case (FP32_EXPONENT_BIAS_DIFF-9) : mantissa = (1 <<  3) | (BIT_SEL(mantissa, 22, 21) << 1) | (BIT_SEL(mantissa, 20, 0) != 0); break;
      case (FP32_EXPONENT_BIAS_DIFF-10): mantissa = (1 <<  2) | (BIT_SEL(mantissa, 22, 22) << 1) | (BIT_SEL(mantissa, 21, 0) != 0); break;
      default:  mantissa = 0; break;
   } 
#endif
    exponent = 0;
  } else if (exponent > (FP32_EXPONENT_BIAS_DIFF + 0x01e)) {
    // exponent is too large for half precision, return infinity
    exponent = 0x1f;
    mantissa = 0;
  } else {
    // normal situation, translate the value
    exponent = exponent - FP32_EXPONENT_BIAS_DIFF;
    mantissa = (1 << 13) | (BIT_SEL(mantissa, 22, 11) << 1) | (BIT_SEL(mantissa, 10, 8) != 0);
  }

  // Note : not rounding if we overflow, since the overflow would ripple into
  //        the exponent, and then we would have to check that the exponent does not overflow
  //

  if ((BIT_SEL(mantissa, 12,3) != BIT_MASK_RANGE(9,0))  &&
      ((BIT_SEL(mantissa, 3, 0) == BIT_MASK_RANGE(3, 2)) || (BIT_SEL(mantissa, 2, 0) > 4))) {
    mantissa = BIT_SEL(mantissa, 12, 3) + 1;
  } else {
    mantissa = BIT_SEL(mantissa, 12, 3);
  }

  ushort result = (sign << 15) | (exponent << 10) | (mantissa << 0);
  return ushort_as_half(result);

#else
  return real_half_to_half((half)orig_float);
#endif
}
STATIC half_t float_to_half_disable_subnorm(float orig_float)
{
#if !defined(NATIVE_FP16_AVAILABLE) || defined(DISABLE_NATIVE_FP16_CAST) 

    // This implementation was written to be essentially exactly the same as the
    // Verilog implementation that's part of the compiler. I used that Verilog as
    // a starting point and re-wrote it in OpenCL C.
    uint orig_float_raw = as_uint(orig_float);
    uint sign = BIT_SEL(orig_float_raw, 31, 31);
    uint exponent = BIT_SEL(orig_float_raw, 30, 23);
    uint mantissa = BIT_SEL(orig_float_raw, 22, 0);

    if (exponent <= FP32_EXPONENT_BIAS_DIFF) {

        mantissa = 0;
        exponent = 0;
    }
    else if (exponent > (FP32_EXPONENT_BIAS_DIFF + 0x01e)) {
        // exponent is too large for half precision, return infinity
        exponent = 0x1f;
        mantissa = 0;
    }
    else {
        // normal situation, translate the value
        exponent = exponent - FP32_EXPONENT_BIAS_DIFF;
        mantissa = (1 << 13) | (BIT_SEL(mantissa, 22, 11) << 1) | (BIT_SEL(mantissa, 10, 8) != 0);
    }

    // Note : not rounding if we overflow, since the overflow would ripple into
    //        the exponent, and then we would have to check that the exponent does not overflow
    //

    if ((BIT_SEL(mantissa, 12, 3) != BIT_MASK_RANGE(9, 0)) &&
        ((BIT_SEL(mantissa, 3, 0) == BIT_MASK_RANGE(3, 2)) || (BIT_SEL(mantissa, 2, 0) > 4))) {
        mantissa = BIT_SEL(mantissa, 12, 3) + 1;
    }
    else {
        mantissa = BIT_SEL(mantissa, 12, 3);
    }

    ushort result = (sign << 15) | (exponent << 10) | (mantissa << 0);
    return ushort_as_half(result);

#else
    return real_half_to_half((half)orig_float);
#endif
}

// -------------------------------------------------------------------------- //
// math operations:

STATIC half_t add_half(half_t a, half_t b) {
#if defined(NATIVE_FP16_AVAILABLE)
  return real_half_to_half(half_to_real_half(a) + half_to_real_half(b));
#else
  return float_to_half(half_to_float(a) + half_to_float(b));
#endif
}

STATIC half_t subtract_half(half_t a, half_t b) {
#if defined(NATIVE_FP16_AVAILABLE)
  return real_half_to_half(half_to_real_half(a) - half_to_real_half(b));
#else
  return float_to_half(half_to_float(a) - half_to_float(b));
#endif
}

STATIC half_t multiply_half(half_t a, half_t b) {
#if defined(NATIVE_FP16_AVAILABLE)
  return real_half_to_half(half_to_real_half(a) * half_to_real_half(b));
#else
  return float_to_half(half_to_float(a) * half_to_float(b));
#endif
}

STATIC half_t divide_half(half_t a, half_t b) {
#if defined(NATIVE_FP16_AVAILABLE)
  return real_half_to_half( (float) half_to_real_half(a) / half_to_real_half(b));
#else
  return float_to_half( (float) half_to_float(a) / half_to_float(b));
#endif
}

STATIC half_t multiply_half_by_int(half_t a, int b) {
#if defined(NATIVE_FP16_AVAILABLE)
  return real_half_to_half(half_to_real_half(a) * b);
#else
  return float_to_half(half_to_float(a) * b);
#endif
}

STATIC half_t divide_half_by_int(half_t a, int b) {
#if defined(NATIVE_FP16_AVAILABLE)
  return real_half_to_half( (float) half_to_real_half(a) / b);
#else
  return float_to_half( (float) half_to_float(a) / b);
#endif
}

STATIC half_t max_half(half_t a, half_t b) {
  return ushort_as_half(max_raw_float(half_as_ushort(a), half_as_ushort(b),
        sizeof(ushort)*8));
}

STATIC half_t create_half(bool sign, uchar exponent, uint mantissa) {
  return ushort_as_half(
      (BIT_MASK_VAL(sign,     1) << 15) |
      (BIT_MASK_VAL(exponent, 5) << 10) |
      (BIT_MASK_VAL(mantissa, 10)));
}

STATIC bool half_sign(half_t val) {
  return BIT_IS_SET(half_as_ushort(val), 15);
}

STATIC uchar half_exponent(half_t val) {
  return BIT_SEL(half_as_ushort(val), 14, 10);
}

STATIC uint half_mantissa(half_t val) {
  return BIT_SEL(half_as_ushort(val), 9, 0);
}

#endif // __FPGA_FP16_H__
