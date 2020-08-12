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

#ifndef __FPGA_TYPES_BLOCKFP_H__
#define __FPGA_TYPES_BLOCKFP_H__


#define SHORT_MSB 15
#define AUX_LSB (0)
  STATIC ushort mask_dla_type(ushort val, int low_bit) {
    val &= BIT_MASK_RANGE((SHORT_MSB), (low_bit));
    return val;
  }
  
  STATIC ushort dla_type_to_raw(ushort val, int low_bit) {
      // Why mask? dla type should already be masked
    return mask_dla_type(val, low_bit);
  }
  
  STATIC ushort raw_to_dla_type(ushort val, int low_bit) {
      ushort res = val;
    return mask_dla_type(res, low_bit);
  }

  STATIC ushort raw_to_rounded_dla_type(ushort val, int low_bit) {
    bool round_up = ((low_bit) <= 0) ? false : BIT_IS_SET(val, (low_bit)-1); 
      ushort val_rounded = (round_up) ? val + (1 << (low_bit)) : val; 
    /* If the mantissa overflows, it would cause the exponent to increment, */
    /* which is the desired behaviour */
    if (round_up && ((val_rounded & BIT_MASK_RANGE((SHORT_MSB)-1, (low_bit))) == 0)) {
      /* Guard against overflowing into the sign bit */
      val_rounded = val;
    }
    ushort res = val_rounded;
    ushort ret = mask_dla_type(res,  low_bit);
    return ret;
  }

  STATIC half_t dla_type_to_half(ushort val, int low_bit) {
      return ushort_as_half(dla_type_to_raw(val, low_bit));
  }

  STATIC float dla_type_to_float(ushort val, int low_bit) {
      return half_to_float(dla_type_to_half(val, low_bit));
  }

  STATIC ushort half_to_dla_type(half_t val, int low_bit) {
      return raw_to_rounded_dla_type(half_as_ushort(val), low_bit);
  }

  STATIC ushort float_to_dla_type(float val, int low_bit) {
      return raw_to_rounded_dla_type(half_as_ushort(float_to_half(val)), low_bit);
  }



#if 0
// --- pe_output_value_t ---------------------------------------------------- //

DEFINE_BASIC_TYPE(pe_output_value_t, pe_output_value, ushort, 15, 0);

STATIC half_t pe_output_value_to_half(pe_output_value_t val) {
  return ushort_as_half(pe_output_value_to_raw(val));
}

STATIC float pe_output_value_to_float(pe_output_value_t val) {
  return half_to_float(pe_output_value_to_half(val));
}

STATIC pe_output_value_t float_to_pe_output_value(float val) {
  return raw_to_pe_output_value(half_as_ushort(float_to_half(val)));
}

// --- pe_feature_dot_vec_t/pe_filter_dot_vec_t ----------------------------- //

typedef struct {
  uchar exponent;
  uint  mantissa[C_VECTOR];
} pe_feature_dot_vec_t;
CONSTANT pe_feature_dot_vec_t zero_pe_feature_dot_vec = ZERO_INIT({{{{0}}}});

typedef struct {
  pe_feature_dot_vec_t v[W_VECTOR];
} pe_input_vec_t;
CONSTANT pe_input_vec_t zero_pe_input_vec = ZERO_INIT({{{{0}}}});

typedef struct {
  uchar exponent;
  uint  mantissa[C_VECTOR];
} pe_filter_dot_vec_t;
CONSTANT pe_filter_dot_vec_t zero_pe_filter_dot_vec = ZERO_INIT({{{{0}}}});

typedef struct {
  pe_filter_dot_vec_t v;
} pe_filter_vec_t;
CONSTANT pe_filter_vec_t zero_pe_filter_vec = ZERO_INIT({{{0}}});

typedef struct {
  uint  mantissa[C_VECTOR];
  uchar exponent;
} pe_packed_filter_bank_t;
CONSTANT pe_packed_filter_bank_t zero_pe_packed_filter_bank = {{0}};

typedef struct {
  pe_packed_filter_bank_t banks[NUM_FILTER_CACHE_BANKS];
} pe_packed_filter_vec_t;
CONSTANT pe_packed_filter_vec_t zero_pe_packed_filter_vec = ZERO_INIT({{{{0}}}});

typedef struct {
  pe_input_vec_t v[DOT_PRODUCT_NUM_FEATURES];
} dot_product_features_t;
CONSTANT dot_product_features_t zero_dot_product_features = ZERO_INIT({{{{{{0}}}}}});

typedef struct {
  pe_packed_filter_vec_t v[FILTERS_PER_PE];
} dot_product_filters_t;
CONSTANT dot_product_filters_t zero_dot_product_filters = ZERO_INIT({{{{{{0}}}}}});

typedef struct {
  uint v[DOT_PRODUCT_NUM_FEATURES][FILTERS_PER_PE][PE_OUTPUT_WIDTH];
} dot_product_blockfp_return_t;
CONSTANT dot_product_blockfp_return_t zero_dot_product_blockfp_return = ZERO_INIT({{{{{0}}}}});

typedef struct {
  pe_output_value_t v[PE_OUTPUT_WIDTH];
} pe_output_vec_t;
CONSTANT pe_output_vec_t zero_pe_output_vec = ZERO_INIT({{{{0}}}});

typedef struct {
  pe_output_vec_t v[DOT_PRODUCT_NUM_FEATURES][FILTERS_PER_PE];
} dot_product_return_t;
CONSTANT dot_product_return_t zero_dot_product_return = ZERO_INIT({{{{{0}}}}});

STATIC pe_feature_dot_vec_t mask_pe_feature_dot_vec(pe_feature_dot_vec_t val) {
  val.exponent &= BIT_MASK(BLOCKFP_EXPONENT_WIDTH);
  #pragma unroll
  for (int i = 0; i < C_VECTOR; i++) {
    val.mantissa[i] &= BIT_MASK(BLOCKFP_INPUT_DOT_WIDTH);
  }
  return val;
}

STATIC pe_input_vec_t mask_pe_input_vec(pe_input_vec_t pe_input_vec) {
  #pragma unroll
  for(int w = 0; w < W_VECTOR; w++) {
    pe_input_vec.v[w] = mask_pe_feature_dot_vec(pe_input_vec.v[w]);
  }
  return pe_input_vec;
}

STATIC pe_filter_dot_vec_t mask_pe_filter_dot_vec(pe_filter_dot_vec_t val) {
  val.exponent &= BIT_MASK(BLOCKFP_EXPONENT_WIDTH);
  #pragma unroll
  for (int i = 0; i < C_VECTOR; i++) {
    val.mantissa[i] &= BIT_MASK(BLOCKFP_FILTER_DOT_WIDTH);
  }
  return val;
}

STATIC pe_filter_vec_t mask_pe_filter_vec(pe_filter_vec_t pe_filter_vec) {
  pe_filter_vec.v = mask_pe_filter_dot_vec(pe_filter_vec.v);
  return pe_filter_vec;
}

STATIC pe_packed_filter_bank_t mask_pe_packed_filter_bank(
    pe_packed_filter_bank_t pe_packed_filter_bank) {
  pe_packed_filter_bank.exponent &= BIT_MASK(BLOCKFP_EXPONENT_WIDTH);
  #pragma unroll
  for (int c = 0; c < C_VECTOR; c++) {
    pe_packed_filter_bank.mantissa[c] &= BIT_MASK(BLOCKFP_FILTER_DOT_WIDTH);
  }

  return pe_packed_filter_bank;
}

STATIC pe_packed_filter_vec_t mask_pe_packed_filter_vec(
    pe_packed_filter_vec_t pe_packed_filter_vec) {
  #pragma unroll
  for(int bank = 0; bank < NUM_FILTER_CACHE_BANKS; bank++) {
    pe_packed_filter_vec.banks[bank] =
      mask_pe_packed_filter_bank(pe_packed_filter_vec.banks[bank]);
  }
  return pe_packed_filter_vec;
}

STATIC pe_output_vec_t mask_pe_output_vec(pe_output_vec_t pe_output_vec) {
  #pragma unroll
  for(int i = 0; i < PE_OUTPUT_WIDTH; i++) {
    pe_output_vec.v[i] = mask_pe_output_value(pe_output_vec.v[i]);
  }
  return pe_output_vec;
}

STATIC float pe_feature_dot_vec_to_float(pe_feature_dot_vec_t pe_feature_dot_vec,
    int index) {
  char exponent = pe_feature_dot_vec.exponent - 15 - (BLOCKFP_INPUT_DOT_WIDTH - 2);
  // When we are 2x packing into the DSP, we store mantissas in sign+magnitude
  // format rather than 2's complement
  int mantissa = pe_feature_dot_vec.mantissa[index] & BIT_MASK(BLOCKFP_INPUT_DOT_WIDTH-1);
  if (BIT_IS_SET(pe_feature_dot_vec.mantissa[index], BLOCKFP_INPUT_DOT_WIDTH-1)) {
    mantissa = -mantissa;
  }
  return (float)mantissa * pown(2.0, exponent);
}

STATIC float pe_filter_dot_vec_to_float(pe_filter_dot_vec_t pe_filter_dot_vec,
    int index) {
  char exponent = pe_filter_dot_vec.exponent - 15 - (BLOCKFP_FILTER_DOT_WIDTH - 2);
  // When we are 2x packing into the DSP, we store mantissas in sign+magnitude
  // format rather than 2's complement
  int mantissa = pe_filter_dot_vec.mantissa[index] & BIT_MASK(BLOCKFP_FILTER_DOT_WIDTH-1);
  if (BIT_IS_SET(pe_filter_dot_vec.mantissa[index], BLOCKFP_FILTER_DOT_WIDTH-1)) {
    mantissa = -mantissa;
  }
  return (float)mantissa * pown(2.0, exponent);
}

#define COMPARE_pe_input_vec_t_TO_REF(name, stats, ref_value, emu_value, \
    coord_format_str, coord_0, coord_1, coord_2, coord_3, coord_4, coord_5, coord_6, coord_7, coord_8, coord_9, coord_10) do { \
  for (int w = 0; w < W_VECTOR; w++) { \
    for (int c = 0; c < C_VECTOR; c++) { \
      stats.num_values++; \
      float ref_float = pe_feature_dot_vec_to_float(ref_value.v[w], c); \
      float emu_float = pe_feature_dot_vec_to_float(emu_value.v[w], c); \
      bool ref_match = (ref_value.v[w].exponent == emu_value.v[w].exponent && \
          ref_value.v[w].mantissa[c] == emu_value.v[w].mantissa[c]); \
      if (ref_match) { stats.num_match++; } else { stats.num_mismatch++; } \
      if (CAT(_PRINT_MATCHING_, name) || !ref_match) { \
        printf("%s%s " coord_format_str " w%d_c%d ref=%f (man=%d,exp=%d) " \
            "emu=%f (man=%d,exp=%d) correct=%lu/%lu (%.2f%%)\n", \
            ref_match ? "" : "Error:COMPARE_TO_REF ", #name, \
            coord_0, coord_1, coord_2, coord_3, coord_4, coord_5, coord_6, coord_7, coord_8, coord_9, coord_10, \
            w, c, ref_float, \
            ref_value.v[w].mantissa[c], ref_value.v[w].exponent, emu_float, \
            emu_value.v[w].mantissa[c], emu_value.v[w].exponent, \
            stats.num_match, stats.num_values, \
            ((float)stats.num_match / (float)stats.num_values)*100.0); \
        stats.num_printed++; \
      } \
    } \
  } \
} while (0)

// -------------------------------------------------------------------------- //

// this function expects 2's complement numbers
STATIC void block_normalize(int num_values, uint values[], uchar* exponent,
    int old_width, int new_width, int max_shift) {

  if (old_width < new_width) {
    ERROR_EXIT_WITH_MESSAGE("block_normalize() cannot increase width");
  }

  bool msb_used[32] = {0};
  if (max_shift > ARRAY_LENGTH(msb_used)) {
    ERROR_EXIT_WITH_MESSAGE("max_shift too large for msb_used");
  }

  #pragma unroll
  for (int value_idx = 0; value_idx < num_values; value_idx++) {
    #pragma unroll
    for (int msb_idx = 0; msb_idx < max_shift; msb_idx++) {
      bool this_msb_bit = BIT_IS_SET(values[value_idx], old_width-1-msb_idx);
      bool next_msb_bit = BIT_IS_SET(values[value_idx], old_width-1-msb_idx-1);
      if (this_msb_bit != next_msb_bit) msb_used[msb_idx] = true;
    }
  }

  bool shift_found = false;
  int shift = 0;
  #pragma unroll
  for (int this_shift = max_shift; this_shift >= 0; this_shift--) {
    if (!shift_found) {

      // don't use this shift if it would result in losing msb bits
      bool use_shift = true;
      #pragma unroll
      for (int check_msb_idx = 0; check_msb_idx < ARRAY_LENGTH(msb_used); check_msb_idx++) {
        if (check_msb_idx < this_shift && msb_used[check_msb_idx]) use_shift = false;
      }

      if (use_shift) {
        shift = this_shift;
        shift_found = true;
      }
    }
  }

  if(shift_found){
    *exponent += (old_width - new_width - shift);

    bool overflow = !BIT_IS_SET(*exponent, 7) && BIT_IS_SET(*exponent, 5);
    bool underflow = BIT_IS_SET(*exponent, 7) && BIT_IS_SET(*exponent, 5);
    if (underflow) *exponent = 0;
    if (overflow)  *exponent = BIT_MASK(5);
    *exponent &= BIT_MASK(5);

    #pragma unroll
    for (int value_idx = 0; value_idx < num_values; value_idx++) {
      // store in a uint_64_t temporarily to make sure we have room to shift
      // left and then right
      uint_64_t value = values[value_idx];
      value <<= shift;
      value >>= (old_width - new_width);
      values[value_idx] = value & BIT_MASK(new_width);
    }
  }
}


//[TODO] remove this
STATIC pe_filter_vec_t transform_filter(data_value_t filter[C_VECTOR]) {

  half_t processed_filter[C_VECTOR];

  #pragma unroll
  for(int c = 0; c < C_VECTOR; c++) {
    processed_filter[c] = ushort_as_half(data_value_to_raw(filter[c]));
  }

  pe_filter_vec_t transformed_filter = zero_pe_filter_vec;

  bool  signs[C_VECTOR] = {0};
  uchar exponents[C_VECTOR] = {0};
  uint  mantissas[C_VECTOR] = {0};

  #pragma unroll
  for(int c = 0; c < C_VECTOR; c++) {
    // extract value
    ushort raw_value = half_as_ushort(processed_filter[c]);
    signs    [c] = BIT_SEL(raw_value, 15, 15);
    exponents[c] = BIT_SEL(raw_value, 14, 10);
    mantissas[c] = BIT_SEL(raw_value,  9,  0);
  }

  uchar max_exponent = max_uchar(C_VECTOR, exponents);

  block_align_mantissas(C_VECTOR, signs, exponents, mantissas,
      max_exponent, /* input_width */ 10,
      /* output_width */ BLOCKFP_FILTER_DOT_WIDTH,
      /* add_implicit_one */ true, /* use_2s_complement_output */ false);

  transformed_filter.v.exponent = max_exponent;
  #pragma unroll
  for (int c = 0; c < C_VECTOR; c++) {
    transformed_filter.v.mantissa[c] = mantissas[c];
    transformed_filter.v.mantissa[c] &= BIT_MASK(BLOCKFP_FILTER_DOT_WIDTH);
  }

  return transformed_filter;
}

STATIC pe_packed_filter_vec_t pack_pe_filter_vec(pe_filter_vec_t pe_filter_vec) {
  pe_packed_filter_vec_t pe_packed_filter_vec = zero_pe_packed_filter_vec;

  pe_packed_filter_vec.banks[0].exponent = pe_filter_vec.v.exponent;

  #pragma unroll
  for (int c = 0; c < C_VECTOR; c++) {
    pe_packed_filter_vec.banks[0].mantissa[c] = pe_filter_vec.v.mantissa[c];
  }

  return pe_packed_filter_vec;
}

STATIC pe_array_block_t pe_filter_vec_to_pe_array_block(pe_filter_dot_vec_t filter_dot) {
  pe_array_block_t result = zero_pe_array_block;

  result.exponent = filter_dot.exponent;
  #pragma unroll
  for (int c = 0; c < C_VECTOR; c++) {
    result.mantissa[c] = filter_dot.mantissa[c];
  }

  return result;
}

STATIC pe_array_block_t pe_feature_vec_to_pe_array_block(pe_feature_dot_vec_t feature_dot) {
  pe_array_block_t result = zero_pe_array_block;

  result.exponent = feature_dot.exponent;
  #pragma unroll
  for (int c = 0; c < C_VECTOR; c++) {
    result.mantissa[c] = feature_dot.mantissa[c];
  }

  return result;
}
#endif
#endif // __FPGA_TYPES_BLOCKFP_H__
