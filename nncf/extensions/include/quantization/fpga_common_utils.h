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

#ifndef __FPGA_COMMON_UTILS_H__
#define __FPGA_COMMON_UTILS_H__
// -------------------------------------------------------------------------- //
// fpga_common_utils.h:
//
// Generic utilities for FPGA and non-FPGA code
// -------------------------------------------------------------------------- //

#ifdef INTELFPGA_CL
#define STATIC
#define CONSTANT constant
#define GLOBAL global
#define VOLATILE volatile
#elif defined(CUDA_CODE)
#define STATIC __device__
#define CONSTANT static const
#define GLOBAL
#define VOLATILE
#else
#define STATIC
#define CONSTANT static const
#define GLOBAL
#define VOLATILE
#endif

#if !defined(INTELFPGA_CL) || defined(EMULATOR)
#define DEBUG_PRINTF printf
#else
#define DEBUG_PRINTF if (false) printf
#endif

#if defined(INTELFPGA_CL) && (AOC_VERSION >= 181)
#define CONSTANT_STRING_LITERAL constant
#else
#define CONSTANT_STRING_LITERAL
#endif

// Print #define int value
#define VALUE_TO_STRING(x) #x
#define VALUE(x) VALUE_TO_STRING(x)
#define VAR_NAME_VALUE(var) #var "="  VALUE(var)

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

// Add some typedefs and functions which exist in OpenCL C but not in regular
// C. This allows the same header file to be used on the host.
#ifndef INTELFPGA_CL
// include this for pow()
#include <cmath>
// include this in order to get printf(), which is built-in in OpenCL
#include <cstdio>

typedef unsigned int       uint;
typedef unsigned short     ushort;
typedef unsigned char      uchar;

struct address_coordinates_4d {
  int k;
  int c;
  int h;
  int w;
};
#ifndef EFI_LIB
STATIC float as_float(uint val) {
  return *((float*)&val);
}
STATIC uint as_uint(float val) {
  return *((uint*)&val);
}
STATIC float pown(float base, int exponent) {
  return pow(base, exponent);
}
#endif
#endif

// Create typedefs for 64-bit ints (the underlying type differs on host vs OpenCL)
#ifdef INTELFPGA_CL
typedef long               int_64_t;
typedef unsigned long      uint_64_t;
#define STR_FORMAT_UINT64  "%lu"
#else
typedef long long          int_64_t;
typedef unsigned long long uint_64_t;
#define STR_FORMAT_UINT64  "%llu"
#endif

#define NEXT_DIVISIBLE(X, Y) ( ( (X) % (Y) ) == 0 ? (X) : ( (X) + (Y) - ( (X) % (Y) ) ) )
#define MYMIN2(X, Y) ( ( (X) > (Y) ) ? (Y) : (X) )
#define MYMIN3(X, Y, Z) ( ( (X) < (Y) ) ? ( (X) < (Z) ? (X) : (Z) ) : ( (Y) < (Z) ? (Y) : (Z) ) )
#define X_OR_SHIFT_Y(X,Y) ( ( X ) | ( ( X ) >> ( Y ) ) )
#define NEXT_POWER_OF_2(X) ( X_OR_SHIFT_Y(X_OR_SHIFT_Y(X_OR_SHIFT_Y(X_OR_SHIFT_Y(X_OR_SHIFT_Y((X)-1, 1), 2), 4), 8), 16) + 1 )
#define MYMAX3(X, Y, Z) ( (X) >= (Y) ? ( (X) >= (Z) ? (X) : (Z) ) : ( (Y) >= (Z) ? (Y) : (Z) ) )
#define MYMAX2(X, Y) ( (X) >= (Y) ? (X) : (Y) )
#define MYMAX4(V, W, X, Y) ( MYMAX2( MYMAX2(V, W), MYMAX2(X, Y) ) )
#define MYMAX5(V, W, X, Y, Z) ( MYMAX2( MYMAX2(V, W), MYMAX3(X, Y, Z) ) )
#define MYMAX8(S, T, U, V, W, X, Y, Z) ( MYMAX2( MYMAX4(S, T, U, V), MYMAX4(W, X, Y, Z) ) )
#define IS_POWER_OF_2(X) ( ((X)>0) && ( ( (X) & ((X)-1) ) == 0 ) )

#define BIT_MASK(num_bits) ((1ULL << (num_bits))-1)
#define BIT_MASK_VAL(value, num_bits) ((value) & BIT_MASK(num_bits))
#define BIT_MASK_SINGLE(bit) (1ULL << (bit))
#define BIT_MASK_RANGE(start_bit, end_bit) (BIT_MASK((start_bit)-(end_bit)+1) << (end_bit))
#define BIT_SEL(value, start_bit, end_bit) (((value) & BIT_MASK_RANGE(start_bit, end_bit)) >> (end_bit))
#define BIT_IS_SET(value, bit_num) (((value) & (1ULL << (bit_num))) != 0)

#define BIT_MASK_CAPACITY(a,b) ((a) & BIT_MASK(CLOG2((b))))

#define BIT_SET(value, start_bit, end_bit, set_value) do { \
  int __end_bit = (end_bit); \
  unsigned long __mask = BIT_MASK_RANGE(start_bit, __end_bit); \
  value = (value & ~__mask) | (((set_value) << __end_bit) & __mask); \
} while(0)

#define SIGN_EXTEND(value, old_width, new_width) \
  (BIT_MASK_VAL(value, old_width) | (BIT_IS_SET((value), (old_width)-1) ? \
  BIT_MASK_RANGE((new_width)-1,(old_width)) : 0))

#define SIGN_SHIFT_RIGHT(value, width, shift) \
  SIGN_EXTEND((value) >> (shift), ((width)-(shift)), (width))

#ifndef EFI_LIB
#if 0
STATIC uint to_2s_complement(bool sign, uint value, int input_width) {
  value &= BIT_MASK(input_width);
  if (sign) value = -value;
  return value & BIT_MASK(input_width+1);
}

STATIC uint from_2s_complement(uint value, int input_width, bool* output_sign,
    bool* overflow) {
  value &= BIT_MASK(input_width);
  *output_sign = BIT_IS_SET(value, input_width-1);
  if (*output_sign) value = -value;
  *overflow = BIT_IS_SET(value, input_width-1);
  return value & BIT_MASK(input_width-1);
}
#endif
#endif

#define LOG2_2(x)  ((x) & 0x2        ? 1                       : 0)
#define LOG2_4(x)  ((x) & 0xC        ? 2  + LOG2_2((x)  >>  2) : LOG2_2(x))
#define LOG2_8(x)  ((x) & 0xF0       ? 4  + LOG2_4((x)  >>  4) : LOG2_4(x))
#define LOG2_16(x) ((x) & 0xFF00     ? 8  + LOG2_8((x)  >>  8) : LOG2_8(x))
#define LOG2(x)    ((x) & 0xFFFF0000 ? 16 + LOG2_16((x) >> 16) : LOG2_16(x))
#define CLOG2(x)   (LOG2((x)-1)+1)

#define GCD(X, Y) ( ( (X) % (Y) ) == 0 ? (Y) : ( (Y) % (X) ) == 0 ? (X) : ( ( (X) % 2 ) == 0 && ( (Y) % 2 ) == 0 ) ? 2 : 1 )

#define ARRAY_LENGTH(name) (sizeof(name) / sizeof((name)[0]))

#define ROTATE_ARRAY_LEFT(name, type, length) do { \
  type save_value = name[0]; \
  _Pragma("unroll") \
  for (int i = 0; i < (length)-1; i++) { \
    name[i] = name[i+1]; \
  } \
  name[(length)-1] = save_value; \
} while (0)

#ifndef EFI_LIB
// These functions convert integers into ASCII strings with the binary
// representation of the number. They're useful for debugging code that does a
// lot of bit manipulation.
STATIC void num_to_string(uint_64_t val, int width, char str_buf[]) {
  int buf_idx = 0;
  for (int bit = (width-1); bit >= 0; bit--) {
    str_buf[buf_idx++] = (val & (1ULL << bit)) ? '1' : '0';
    if (bit % 4 == 0) {
      str_buf[buf_idx++] = ' ';
    }
  }
  str_buf[buf_idx] = 0;
}
STATIC void uchar_to_string(uchar val, char str_buf[]) {
  num_to_string(val, 8, str_buf);
}
STATIC void ushort_to_string(ushort val, char str_buf[]) {
  num_to_string(val, 16, str_buf);
}
STATIC void uint_to_string(uint val, char str_buf[]) {
  num_to_string(val, 32, str_buf);
}
STATIC void ulong_to_string(uint_64_t val, char str_buf[]) {
  num_to_string(val, 64, str_buf);
}
#endif

#define ARRAY_LENGTH(name) (sizeof(name) / sizeof((name)[0]))

// TODO: [shaneoco] it's possible this EMULATOR ifdef isn't necessary, and
// the hardware compile will simply ignore the extra code with the printf
#ifdef EMULATOR
#define ARRAY_LOOKUP(name, idx) ({ \
  int __idx = (idx); \
  if (__idx < 0 || __idx >= ARRAY_LENGTH(name)) { \
    DEBUG_PRINTF("Error: out of bounds array access (name=%s, idx=%d, length=%lu) at %s:%d\n", \
        #name, __idx, ARRAY_LENGTH(name), __FILE__, __LINE__); \
    } \
  (name); \
})[(idx)]
#else
#define ARRAY_LOOKUP(name, idx) (name)[idx]
#endif

#ifndef PRAGMA
#ifdef INTELFPGA_CL
#define PRAGMA _Pragma
#elif _WIN32
#define PRAGMA __pragma
#else
#define PRAGMA _Pragma
#endif
#endif

// The PACK_WRITE and PACK_READ macros allow you to pack data into an array
// such that there are no gaps (i.e. so that there are no constant zeros). This
// is useful for local memories and DDR accesses where we need all of the bits
// that are actually used to be left-aligned in the array.
#define PACK_WRITE(pack, offset, width, data) do { \
  PRAGMA("unroll") \
  for (int src_idx = 0; src_idx < (width); src_idx++) { \
    int dst_idx = (offset) + src_idx; \
    int dst_array_index = dst_idx / (sizeof(pack[0])*8); \
    int dst_array_offset = dst_idx % (sizeof(pack[0])*8); \
    ARRAY_LOOKUP(pack, dst_array_index) &= ~(1ULL << dst_array_offset); \
    ARRAY_LOOKUP(pack, dst_array_index) |= (uint_64_t)((((data) >> src_idx) & 1) << dst_array_offset); \
  } \
} while (0)

#define PACK_READ(pack, offset, width, data) do { \
  data = 0; \
  PRAGMA("unroll") \
  for (int dst_idx = 0; dst_idx < (width); dst_idx++) { \
    int src_idx = (offset) + dst_idx; \
    int src_array_index = src_idx / (sizeof(pack[0])*8); \
    int src_array_offset = src_idx % (sizeof(pack[0])*8); \
    data |= (((ARRAY_LOOKUP(pack, src_array_index) >> src_array_offset) & 1) << dst_idx); \
  } \
} while (0)

// PACK_WRITE_2D and PACK_READ_2D perform the same function as PACK_WRITE and
// PACK_READ, except they pack into a two-dimensional array rather than a
// one-dimensional array. Sometimes a two-dimensional array gets better QoR
// than a one-dimensional array (though it does not make sense to me). For
// example, in the PE, if the filter cache is very wide (for example, if
// C_VECTOR == 16), then the OpenCL compiler will add unnecessary cycles to the
// schedule and waste registers. If I just change the array to be a 2D array,
// arbitrarily pick a size of 2 for the first dimension, and for the second
// dimension take the size I wanted and divide it by 2, then the extra cycle
// goes away. Perhaps the compiler uses the dimensions of the array in some
// heuristic, and with the 2D array, no individual dimension is larger than
// some cut-off?
#define PACK_WRITE_2D(pack, offset, width, data) do { \
  PRAGMA("unroll") \
  for (int src_idx = 0; src_idx < (width); src_idx++) { \
    int dst_idx = (offset) + src_idx; \
    int dst_array_index_0 = (dst_idx / (sizeof(pack[0])*8)); \
    int dst_array_index_1 = (dst_idx % (sizeof(pack[0])*8)) / (sizeof(pack[0][0])*8); \
    int dst_array_offset  = (dst_idx % (sizeof(pack[0])*8)) % (sizeof(pack[0][0])*8); \
    ARRAY_LOOKUP(ARRAY_LOOKUP(pack, dst_array_index_0), dst_array_index_1) &= ~(1ULL << dst_array_offset); \
    ARRAY_LOOKUP(ARRAY_LOOKUP(pack, dst_array_index_0), dst_array_index_1) |= (uint_64_t)((((data) >> src_idx) & 1) << dst_array_offset); \
  } \
} while (0)

#define PACK_READ_2D(pack, offset, width, data) do { \
  data = 0; \
  PRAGMA("unroll") \
  for (int dst_idx = 0; dst_idx < (width); dst_idx++) { \
    int src_idx = (offset) + dst_idx; \
    int src_array_index_0 = (src_idx / (sizeof(pack[0])*8)); \
    int src_array_index_1 = (src_idx % (sizeof(pack[0])*8)) / (sizeof(pack[0][0])*8); \
    int src_array_offset  = (src_idx % (sizeof(pack[0])*8)) % (sizeof(pack[0][0])*8); \
    data |= (((ARRAY_LOOKUP(ARRAY_LOOKUP(pack, src_array_index_0), src_array_index_1) >> src_array_offset) & 1) << dst_idx); \
  } \
} while (0)

#ifndef EFI_LIB
STATIC void inc_and_saturate(int* variable, int width, int increment) {
  *variable &= BIT_MASK(width);

  if (*variable > ((int)BIT_MASK(width) - increment)) {
    *variable = BIT_MASK(width);
  } else {
    *variable = (*variable + increment) & BIT_MASK(width);
  }
}

STATIC int next_power_of_2(int x) {
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}

STATIC bool is_power_of_2(int x) {
  return ( x >0 ) && ( ( x & (x-1) ) == 0 );
}

// Multiply unsigned int n by 3
STATIC uint mulu3(uint n) {
  return (n << 1) + n;
}

STATIC uint mulu5(uint n) {
  return (n << 2) + n;
}

STATIC uint mulu6(uint n) {
  return mulu3(n) << 1;
}

STATIC uint mulu7(uint n) {
  return mulu6(n) + n;
}

STATIC uint mulu9(uint n) {
  return (n << 3) + n;
}

STATIC uint mulu10(uint n) {
  return mulu5(n) << 1;
}

// Optimized code for integer divide by constant found on:
//   www.hackersdelight.org
STATIC uint divu3(uint n) {
  uint q, r;
  q = (n >> 2) + (n >> 4);
  q = q + (q >> 4);
  q = q + (q >> 8);
  q = q + (q >> 16);
  r = n - mulu3(q);
  return q + ((r + 5 + (r<<2)) >> 4);
}

STATIC uint divu5(uint n) {
  uint q, r;
  q = (n >> 1) + (n >> 2);
  q = q + (q >> 4);
  q = q + (q >> 8);
  q = q + (q >> 16);
  q = q >> 2;
  r = n - mulu5(q);
  return q + (mulu7(r) >> 5);
}

STATIC uint divu6(uint n) {
  uint q, r;
  q = (n >> 1) + (n >> 3);
  q = q + (q >> 4);
  q = q + (q >> 8);
  q = q + (q >> 16);
  q = q >> 2;
  r = n - mulu6(q);
  return q + ((r + 2) >> 3);
}

STATIC uint divu7(uint n) {
  uint q, r;
  q = (n >> 1) + (n >> 4);
  q = q + (q >> 6);
  q = q + (q>>12) + (q>>24);
  q = q >> 2;
  r = n - mulu7(q);
  return q + ((r + 1) >> 3);
}

// Multiplication of an unsigned int x by a constant unsigned int n
STATIC uint mul_by_constant(uint x, const uint n) {
  if(is_power_of_2(n)) {
    return x << LOG2(n);
  }
  else if(n == 3) {
    return mulu3(x);
  }
  else if (n == 5) {
    return mulu5(x);
  }
  else if(n == 6) {
    return mulu6(x);
  }
  else if(n == 7) {
    return mulu7(x);
  }
  else if(n == 10) {
    return mulu10(x);
  }
  else if(n == 12) {
    return mulu3(x) << 2;
  }
  else if(n == 24) {
    return mulu3(x) << 3;
  }
  else if(n == 36) {
    return mulu9(x) << 2;
  }
  else if(n == 48) {
    return mulu3(x) << 4;
  }
  else if(n == 72) {
    return mulu9(x) << 3;
  }
  else {
#ifdef INTELFPGA_CL
    // The unit test for this function calls it with lots of random numbers, so
    // this warning prints a lot. Therefore, only print the warning if
    // INTELFPGA_CL is set, which indicates this code is being compiled by the
    // OpenCL compiler. This will mean the warning only prints when the code is
    // run in the OpenCL emulator and not in the unit test.
    DEBUG_PRINTF("Warning: using hardware-unoptimized mul_by_constant() for %d \n", n);
#endif
    return x * n;
  }
}

// Division of an unsigned int x by a constant unsigned int n
STATIC uint div_by_constant(uint x, const uint n) {
  if(is_power_of_2(n)) {
    return x >> LOG2(n);
  }
  else if(n == 3) {
    return divu3(x);
  }
  else if(n == 5) {
    return divu5(x);
  }
  else if(n == 6) {
    return divu6(x);
  }
  else if(n == 7) {
    return divu7(x);
  }
  else if(n == 10) {
    return divu5(x) >> 1;
  }
  else if(n == 12) {
    return divu3(x) >> 2;
  }
  else if(n == 24) {
    return divu3(x) >> 3;
  }
  else if(n == 48) {
    return divu6(x) >> 3;
  }
  else {
#ifdef INTELFPGA_CL
    // The unit test for this function calls it with lots of random numbers, so
    // this warning prints a lot. Therefore, only print the warning if
    // INTELFPGA_CL is set, which indicates this code is being compiled by the
    // OpenCL compiler. This will mean the warning only prints when the code is
    // run in the OpenCL emulator and not in the unit test.
    DEBUG_PRINTF("Warning: using hardware-unoptimized div_by_constant() for x=%d n=%d\n", x, n);
#endif
    return x / n;
  }
}

// Modulus of an unsigned int x by a constant unsigned int n
STATIC uint mod_by_constant(uint x, const uint n) {
  if(is_power_of_2(n)) {
    return x & (n-1);
  }
  else if(n == 3 || n == 5 || n == 6 || n == 7 || n == 10 || n == 12 || n == 24 || n == 48) {
    return BIT_MASK_CAPACITY(x - mul_by_constant(div_by_constant(x, n), n), n);
  }
  else {
#ifdef INTELFPGA_CL
    // The unit test for this function calls it with lots of random numbers, so
    // this warning prints a lot. Therefore, only print the warning if
    // INTELFPGA_CL is set, which indicates this code is being compiled by the
    // OpenCL compiler. This will mean the warning only prints when the code is
    // run in the OpenCL emulator and not in the unit test.
    DEBUG_PRINTF("Warning: using hardware-unoptimized mod_by_constant()\n");
#endif
    return x % n;
  }
}

// Modulus of a signed integer by a constant unsigned integer. The result 
// should always be positive. e.g. positive_modular(8, 3) = 2,
// positive_modular(-8, 3) = 1.
STATIC int positive_modular(int x, const uint n){
    uint sign_mask = BIT_MASK_SINGLE(mul_by_constant(sizeof(int), 8) - 1);
    uint sign_remainder = BIT_MASK_SINGLE(mul_by_constant(sizeof(int), 8) - 1) % n;
    uint real_sign_remainder = (x & sign_mask) ? n - sign_remainder : 0;

    return mod_by_constant(
       real_sign_remainder
        + BIT_MASK_VAL(x, mul_by_constant(sizeof(int), 8) - 1)
      ,
      n
    );
}

// Ceiling of an unsigned int x by a constant unsigned int n
// Use this whenever x can vary, but n is constant
STATIC uint ceil_by_constant(uint x, const uint n) {
  return div_by_constant(x + n - 1, n);
}

// Specialized x/6 for 5-bit x. Smaller than 32-bit div_by_constant above.
STATIC unsigned char div6_5bit (unsigned char x) {
  // x/6 = (x/2)/3 = y/3, where y is in [0,15) range
  unsigned char y = x >> 1;
  const unsigned char div_table[16] = {0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4, 5};
  return div_table[y];
}
#endif

// Finds the smallest multiple of Y larger than X.
// Assumes that X and Y are constant unsigned ints, and Y is constant
// Use this when both X and Y are constant
#define MYCEIL(X, Y) (((X) + (Y) - 1) / (Y))

// To catch errors, a type can be wrapped in a struct when typedefed, in order
// to prevent the type from being able to be implicitly casted to any other
// type. Unfortunately, this has to be disabled when building for the
// hardware, because of case 357848 (for some reason the struct creates an LSU
// which uses extra area).
// TODO: Clean this up when case 357848 is fixed.
#if !defined(INTELFPGA_CL) || defined(EMULATOR)
#define SAFE_TYPEDEF(existing_type, new_type) \
  typedef struct { existing_type val; } new_type;
#define SAFE_TYPEDEF_ZERO {0}
#define SAFE_TYPEDEF_VAL(x) x.val
#else
#define SAFE_TYPEDEF(existing_type, new_type) \
  typedef existing_type new_type;
#define SAFE_TYPEDEF_ZERO 0
#define SAFE_TYPEDEF_VAL(x) x
#endif

// ZERO_INIT is used to initialize a struct to all-zeros. Of course, this is
// way more complicated than it needs to be. Here are the requirements:
//   1. On the host, the number of nested curly braces must be at MOST equal to
//   the number of nested structs/arrays. If there are too many you get a
//   compiler error.
//   2. In the OpenCL code, the number of nested curly braces must be AT LEAST
//   equal to the number of nested structs/arrays. If there aren't enough you
//   get an error.
//   3. Unfortunately, the correct amount of nesting can change in some cases
//   depending on architecture parameters. This means there is no simple
//   value we can use to initialize to zero in all cases.
// Therefore the solution for now is to use the ZERO_INIT macro. On the host,
// it ignores it's argument and initialises with {0}, which always works. For
// OpenCL code, it passes it's argument through. The macro should be passed a
// zero-initializer with enough brackets so that we don't get compile errors in
// any architecture.
#ifdef INTELFPGA_CL
#define ZERO_INIT(opencl_zero) {opencl_zero}
#else
#define ZERO_INIT(opencl_zero) {0}
#endif

// These macros allow you to concatenate two tokens AFTER any macro variables
// in the arguments have been expanded
#define INNER_CAT(a, b) a ## b
#define CAT(a, b) INNER_CAT(a, b)
#define INNER_CAT3(a, b, c) a ## b ## c
#define CAT3(a, b, c) INNER_CAT3(a, b, c)
#define INNER_CAT4(a, b, c, d) a ## b ## c ## d
#define CAT4(a, b, c, d) INNER_CAT4(a, b, c, d)

// These macros allow you to stringify a macro value
#define INNER_STR(a) #a
#define STR(a) INNER_STR(a)

// Define an ERROR_EXIT() macro which causes the current process to immediately
// exit with an error. When running the OpenCL emulator using gdb, using
// __builtin_trap() causes gdb to stop at the exact line of code with the
// ERROR_EXIT(), which makes it easy to inspect local variables at the time of
// the error. As well, exiting with __builtin_trap() will definitely cause any
// test to fail, while simply printing an error message may not, if the test
// does not know to look for the error message.
#if defined(INTELFPGA_CL) && defined(EMULATOR)
  extern int fflush (void *);
  #define ERROR_EXIT() do { fflush(0); __builtin_trap(); } while (0)
#elif defined(INTELFPGA_CL) && !defined(EMULATOR)
  // ERROR_EXIT has no effect in hardware compiles
  #define ERROR_EXIT()
#else
  #include <cstdlib>
  #define ERROR_EXIT() exit(1)
#endif

#define ERROR_EXIT_WITH_MESSAGE(desc) do { \
  DEBUG_PRINTF("Error (%s:%d): %s\n", __FILE__, __LINE__, desc); \
  ERROR_EXIT(); \
} while (0)

// Define an assert() statement since OpenCL doesn't have one
#if !defined(INTELFPGA_CL)
#include <cassert>
#elif defined(EMULATOR)
#define assert(cond) do { \
  if (!(cond)) { \
    ERROR_EXIT_WITH_MESSAGE("assertion failure (" #cond ")"); \
  } \
} while (0)
#else
#define assert(cond)
#endif

typedef uint prbs31_state_t;
#define init_prbs31_state ((uint)1)

STATIC uint prbs31(uint* state) {
  uint new_bit = ((*state >> 30) ^ (*state >> 27)) & 1;
  *state = ((*state << 1) | new_bit) & BIT_MASK(31);
  return *state;
}

#ifdef EMULATOR
#define write_channel_nb_debug(channel, value, rand_state, stall_percent) \
  (prbs31(rand_state) >= ((stall_percent)*(BIT_MASK(31)/100)) ? \
   write_channel_nb_intel(channel, value) : false)

#define read_channel_nb_debug(channel, success, rand_state, stall_percent, stall_value) \
  (prbs31(rand_state) >= ((stall_percent)*(BIT_MASK(31)/100)) ? \
   read_channel_nb_intel(channel, success) : (*(success) = false, (stall_value)))
#else
#define write_channel_nb_debug(channel, value, rand_state, stall_percent) \
  write_channel_nb_intel(channel, value)

#define read_channel_nb_debug(channel, success, rand_state, stall_percent, stall_value) \
  read_channel_nb_intel(channel, success)
#endif

#ifndef EFI_LIB
// TODO: [shaneoco] The compiler should really be taking care of making
// a balanced tree of comparisons
STATIC uchar max_uchar_2(uchar values[]) {
  return values[0] > values[1] ? values[0] : values[1];
}

STATIC uchar max_uchar_3(uchar values[]) {
  uchar max_01 = values[0] > values[1] ? values[0] : values[1];

  return max_01 > values[2] ? max_01 : values[2];
}

STATIC uchar max_uchar_4(uchar values[]) {
  uchar max_01 = values[0] > values[1] ? values[0] : values[1];
  uchar max_23 = values[2] > values[3] ? values[2] : values[3];

  return max_01 > max_23 ? max_01 : max_23;
}

STATIC uchar max_uchar_6(uchar values[]) {
  uchar max_01 = values[0] > values[1] ? values[0] : values[1];
  uchar max_23 = values[2] > values[3] ? values[2] : values[3];
  uchar max_45 = values[4] > values[5] ? values[4] : values[5];

  uchar max_0123 = max_01 > max_23 ? max_01 : max_23;
  return max_0123 > max_45 ? max_0123 : max_45;
}

STATIC uchar max_uchar_8(uchar values[]) {
  uchar max_01 = values[0] > values[1] ? values[0] : values[1];
  uchar max_23 = values[2] > values[3] ? values[2] : values[3];
  uchar max_45 = values[4] > values[5] ? values[4] : values[5];
  uchar max_67 = values[6] > values[7] ? values[6] : values[7];

  uchar max_0123 = max_01 > max_23 ? max_01 : max_23;
  uchar max_4567 = max_45 > max_67 ? max_45 : max_67;

  return max_0123 > max_4567 ? max_0123 : max_4567;
}

STATIC uchar max_uchar_12(uchar values[]) {
  uchar max_0 = max_uchar_4(&values[0]);
  uchar max_1 = max_uchar_4(&values[4]);
  uchar max_2 = max_uchar_4(&values[8]);

  uchar max_01 = max_0 > max_1 ? max_0 : max_1;
  return max_01 > max_2 ? max_01 : max_2;
}

STATIC uchar max_uchar_16(uchar values[]) {
  uchar max_0 = max_uchar_8(&values[0]);
  uchar max_1 = max_uchar_8(&values[8]);

  return max_0 > max_1 ? max_0 : max_1;
}

STATIC uchar max_uchar_24(uchar values[]) {
  uchar max_0 = max_uchar_8(&values[0]);
  uchar max_1 = max_uchar_8(&values[8]);
  uchar max_2 = max_uchar_8(&values[16]);

  uchar max_01 = max_0 > max_1 ? max_0 : max_1;
  return max_01 > max_2 ? max_01 : max_2;
}

STATIC uchar max_uchar_32(uchar values[]) {
  uchar max_0 = max_uchar_16(&values[0]);
  uchar max_1 = max_uchar_16(&values[16]);

  return max_0 > max_1 ? max_0 : max_1;
}

STATIC uchar max_uchar_48(uchar values[]) {
  uchar max_0 = max_uchar_24(&values[0]);
  uchar max_1 = max_uchar_24(&values[24]);

  return max_0 > max_1 ? max_0 : max_1;
}

STATIC uchar max_uchar(int num_values, uchar values[]) {
  switch (num_values) {
    case 1:  return values[0];
    case 2:  return max_uchar_2(values);
    case 3:  return max_uchar_3(values);
    case 4:  return max_uchar_4(values);
    case 6:  return max_uchar_6(values);
    case 8:  return max_uchar_8(values);
    case 12: return max_uchar_12(values);
    case 16: return max_uchar_16(values);
    case 24: return max_uchar_24(values);
    case 32: return max_uchar_32(values);
    case 48: return max_uchar_48(values);
    default:
//      ERROR_EXIT_WITH_MESSAGE("Invalid number of values for max_uchar()");
      return 0;
  }
}

STATIC uint max_raw_float(uint a, uint b, int width) {
  bool a_is_pos = !BIT_IS_SET(a, width-1);
  bool b_is_pos = !BIT_IS_SET(b, width-1);
  uint a_abs = a & BIT_MASK(width-1);
  uint b_abs = b & BIT_MASK(width-1);
  bool a_gt_b = a_abs > b_abs;

  if (a_is_pos == b_is_pos) {
    return ((a_is_pos && a_gt_b) || (!a_is_pos && !a_gt_b)) ? a : b;
  } else {
    return a_is_pos ? a : b;
  }
}

STATIC float max_float(float a, float b) {
  return as_float(max_raw_float(as_uint(a), as_uint(b), sizeof(float)*8));
}

STATIC float create_float(bool sign, uchar exponent, uint mantissa) {
  return as_float((uint)((BIT_MASK_VAL(sign,     1) << 31) |
                         (BIT_MASK_VAL(exponent, 8) << 23) |
                         (BIT_MASK_VAL(mantissa, 23))));
}

STATIC bool float_sign(float val) {
  return BIT_IS_SET(as_uint(val), 31);
}

STATIC uchar float_exponent(float val) {
  return (uchar)(BIT_SEL(as_uint(val), 30, 23));
}

STATIC uint float_mantissa(float val) {
  return BIT_SEL(as_uint(val), 22, 0);
}

// When the compiler cannot infer if <offset> is aligned to <alignment>, use
// the following routine to convey this information explicitly.
//
// IMPORTANT: make sure <offset> is indeed aligned to <alignment> for
// correctness!!
//
STATIC uint safe_align_offset(uint offset, const uint alignment) {
  // Use a regular multiply here so that it is clear to the compiler that
  // <aligned_offset> is a multiple of <alignment>
  uint aligned_offset = div_by_constant(offset, alignment) * alignment;
#ifdef EMULATOR
  if (offset != aligned_offset) {
    DEBUG_PRINTF("Error: expect offset %u to be aligned to %u!!\n",
        offset, alignment);
    ERROR_EXIT();
  }
#endif
  return aligned_offset;
}
#endif
/* fp32 support - extract and recreate float */
#define FLOAT_SIGN(x) ((uint)(BIT_SEL((x), 31, 31)))
#define FLOAT_EXP(x) ((uint)(BIT_SEL((x), 30, 23)))
#define FLOAT_MANT(x) ((uint)(BIT_SEL((x), 22, 0)))

#define AS_FLOAT(sign, exp, mant) as_float(((sign & ((uint)(BIT_MASK(1)))) << 31) | \
                                           ((exp  & ((uint)(BIT_MASK(8)))) << 23) | \
                                            (mant & ((uint)(BIT_MASK(23)))))

#ifndef EFI_LIB
STATIC int align_address(int address, const uint wordsize){
  int aligned_address = (div_by_constant(address,wordsize) & BIT_MASK(31-CLOG2(wordsize)))
    * (wordsize);
  return aligned_address;
}
#endif

#endif // __FPGA_COMMON_UTILS_H__
