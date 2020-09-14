#pragma once
// Cosmetically modified from dla/compiler/core/src/dla_floating_point_utils.cpp
//   - Removed all stl::vectors<>
//   - Fixed subnorm hanlding in unblock_filters

// Re-enable to test the functions in this file as a stand-alone program
//#define STANDALONE_COMPILE

#ifdef STANDALONE_COMPILE
#include <stdint.h>
#include <assert.h>
#include <stdio.h>

// From fpga_common_utils.h
#define BIT_MASK(num_bits) ((1ULL << (num_bits))-1)
#define BIT_MASK_VAL(value, num_bits) ((value) & BIT_MASK(num_bits))
#define BIT_MASK_SINGLE(bit) (1ULL << (bit))
#define BIT_MASK_RANGE(start_bit, end_bit) (BIT_MASK((start_bit)-(end_bit)+1) << (end_bit))
#define BIT_SEL(value, start_bit, end_bit) (((value) & BIT_MASK_RANGE(start_bit, end_bit)) >> (end_bit))
#define BIT_IS_SET(value, bit_num) (((value) & (1ULL << (bit_num))) != 0)


#define FP32_WIDTH 32
#define FP32_EXPONENT_WIDTH 8
#define FP32_MANTISSA_WIDTH 23
#define FP32_EXPONENT_BIAS 127
#define FP16_EXPONENT_WIDTH 5
#define FP16_EXPONENT_BIAS 15
#define FP16_EXPONENT_MAX_VALUE 31
#endif


STATIC uint32_t get_max_exponent(uint32_t block_size, uint32_t* exponents) {
    uint32_t max_value = exponents[0];
    for (uint32_t i = 1; i < block_size; i++) {
        if (max_value < exponents[i]) max_value = exponents[i];
    }
    return max_value;
}

STATIC void block_align_mantissas(uint32_t block_size,
    uint32_t input_width,
    uint32_t output_width,
    const uint32_t block_exponent,
    bool*     signs,
    uint32_t* exponents,
    uint32_t* mantissas,
    uint32_t* mantissas_out) {

    for (uint32_t i = 0; i < block_size; i++) {
        bool sign = signs[i];
        uint32_t exponent = exponents[i];
        uint32_t mantissa = mantissas[i] & BIT_MASK(input_width);

        // Output mantissa width including the explicit 1, without the sign bit
        int out_mantissa_width = output_width - 1;

        // input mantissa width excluding G/R/S bits
        int in_mantissa_width = input_width - 3;

        // shift mantissa into the output position
        int shift = in_mantissa_width - out_mantissa_width;

        // shift mantissa to line up with the common exponent
        shift += (block_exponent - exponent);

        // Calculate rounding for mantissa
        bool round_up = false;
        bool lsb = (shift >= 0) ? BIT_IS_SET(mantissa, shift + 3) : BIT_IS_SET(mantissa, 3);
        bool guard_bit = (shift >= 0) ? BIT_IS_SET(mantissa, shift + 2) : BIT_IS_SET(mantissa, 2);
        bool round_bit = (shift >= 0) ? BIT_IS_SET(mantissa, shift + 1) : BIT_IS_SET(mantissa, 1);
        bool sticky_bit;
        if (shift >= 0) {
            sticky_bit = (mantissa & BIT_MASK(shift)) != 0;
        }
        else {
            sticky_bit = BIT_IS_SET(mantissa, 0);
        }

        // Round to nearest odd number. This is to avoid probability of having to
        // round up saturated mantissas.
        if (guard_bit) {
            if (round_bit || sticky_bit) {
                round_up = true;
            }
            else { // tie situation - GRS == 100
                if (!lsb) { // Round to nearest odd
                    round_up = true;
                }
            }
        }

        shift += 3; // get rid of GRS bits

        if (shift >= (int32_t)input_width) {
            mantissa = 0;
        }
        else {
            if (shift > 0) {
                mantissa >>= shift;
            }
            else {
                mantissa <<= shift;
            }
        }

        // round if mantissa is not saturated
        if (mantissa < (uint32_t)(1 << out_mantissa_width) - 1) {
            mantissa += round_up;
        }

        mantissa |= (sign << out_mantissa_width);

        mantissas_out[i] = mantissa & BIT_MASK(output_width);
    }
}


STATIC void block_filters(uint32_t block_size,
    uint32_t blockfp_filter_width,
    uint32_t filter_exp_width,
    float*   input_filters,
    uint32_t* mantissas_out,
    uint32_t* max_exp_out) {

    assert(filter_exp_width == FP16_EXPONENT_WIDTH);

    bool signs[32];
    uint32_t exponents[32];
    uint32_t mantissas[32];

    for (uint32_t i = 0; i < block_size; i++) {
        signs[i] = 0;
        exponents[i] = 0;
        mantissas[i] = 0;
    }

    unsigned int new_fp32_mantissa_size = FP32_MANTISSA_WIDTH +
        1 + // explicit 1
        3;  // G/R/S bits

    for (uint32_t i = 0; i < block_size; i++) {
        uint32_t number = *(reinterpret_cast<uint32_t*>(&(input_filters[i])));

        // Extract FP32 fields
        // sign = fp32_bits[31]
        // exp  = fp32_bits[30:23]
        // man  = fp32_bits[22:0]
        unsigned int sign = (number & 0x80000000) >> (FP32_MANTISSA_WIDTH + FP32_EXPONENT_WIDTH);
        unsigned int exponent = (number & 0x7f800000) >> FP32_MANTISSA_WIDTH;
        unsigned int mantissa = (number & 0x007fffff);

        // Include the implied 1 to the 23-bit mantissa
        mantissa = (exponent != 0) ? ((1 << FP32_MANTISSA_WIDTH) | mantissa) : mantissa;

        // Shift mantissa by 3 locations to make room for g/r/s
        // From now on we are dealing with 27 mantissa bits:
        // 1 (explicit 1) + 23 (FP32 mantissa bits) + 3 (guard, round, sticky)
        mantissa = mantissa << 3;

        if (filter_exp_width == FP16_EXPONENT_WIDTH) {
            if (exponent == 0x0ff) {
                // Handle special case: nan/inf
                // Saturating to biggest/smallest possible number
                exponent = FP16_EXPONENT_MAX_VALUE;
                mantissa = (1 << new_fp32_mantissa_size) - 1; // Biggest unsigned 27-bit number
            }
            else if (exponent > FP16_EXPONENT_MAX_VALUE + FP32_EXPONENT_BIAS - FP16_EXPONENT_BIAS) {
                // Too big for 5 bits exponent
                // Saturating to biggest/smallest possible number
                exponent = FP16_EXPONENT_MAX_VALUE;
                mantissa = (1 << new_fp32_mantissa_size) - 1; // Biggest unsigned 27-bit number
            }
            else if (exponent <= FP32_EXPONENT_BIAS - FP16_EXPONENT_BIAS - new_fp32_mantissa_size) {
                // Any exponent below 85 is too small to be represented
                // Setting to 0
                exponent = 0;
                mantissa = 0;
            }
            else if (exponent < FP32_EXPONENT_BIAS - FP16_EXPONENT_BIAS) {
                // Any exponent between 85 and 111 will be subnormal
                int shift = FP32_EXPONENT_BIAS - FP16_EXPONENT_BIAS - exponent;
                exponent = 0;
                mantissa = mantissa >> (shift);
            }
            else if (exponent != 0) {
                // Exponent -127 removes the bias in single precision,
                // and +15 adds the bias for half-precision.
                exponent = exponent - FP32_EXPONENT_BIAS + FP16_EXPONENT_BIAS;
            }
        }

        signs[i] = sign;
        exponents[i] = exponent;
        mantissas[i] = mantissa;
    }

    uint32_t max_exponent = get_max_exponent(block_size, exponents);
    block_align_mantissas(block_size,
        new_fp32_mantissa_size,
        blockfp_filter_width,
        max_exponent,
        signs,
        exponents,
        mantissas,
        mantissas_out);

    *max_exp_out = max_exponent;
}



STATIC void unblock_filters(uint32_t block_size,
    uint32_t blocked_width,
    uint32_t filter_exp_width,
    uint32_t* filter_values,
    uint32_t max_exponent,
    float*   result_out) {

    assert(filter_exp_width == FP16_EXPONENT_WIDTH);

    for (uint32_t i = 0; i < block_size; ++i) {

        bool sign = (filter_values[i] >> (blocked_width - 1)) & 0x1;
        uint32_t filter_magnitude = filter_values[i] & BIT_MASK(blocked_width - 1);

        uint32_t result = 0;
        uint32_t exponent = max_exponent;

        // Remove FP16 bias, add FP32 bias
        exponent += (FP32_EXPONENT_BIAS - FP16_EXPONENT_BIAS);

        uint32_t sub = 0;
        if (filter_magnitude != 0) {
            // Renormalize
            while (((filter_magnitude >> (blocked_width - 2)) & 0x1) == 0) {
                sub++;
                filter_magnitude <<= 1;
            }
            // Handle underflow
            if (sub < exponent) {
                exponent -= sub;
            }
            else {
                exponent = 0;
                filter_magnitude = 0;
            }

            // Remove leading 1
            filter_magnitude &= BIT_MASK(blocked_width - 2);

            filter_magnitude <<= (FP32_MANTISSA_WIDTH - (blocked_width - 2));

            result = (sign << (FP32_WIDTH - 1)) |
                ((exponent & BIT_MASK(FP32_EXPONENT_WIDTH)) << FP32_MANTISSA_WIDTH) |
                (filter_magnitude & BIT_MASK(FP32_MANTISSA_WIDTH));
        }
        else {
            result = sign << (FP32_WIDTH - 1);
        }
        result_out[i] = *((float *)&result);
    }
}

STATIC void dla_sw_block_c_vec(float *inout,
    uint32_t block_size, uint32_t exp_width, uint32_t mantissa_width) {

    uint32_t mantissas[32];
    uint32_t max_exp;
    // blocking will handling subnorms and then block the values
    block_filters(block_size, mantissa_width, exp_width, inout, mantissas, &max_exp);
    // unblocking will convert mantissa/exp back into FP32 float
    unblock_filters(block_size, mantissa_width, exp_width, mantissas, max_exp, inout);
}

#ifdef STANDALONE_COMPILE
// Re-enable to test the functions above as a stand-alone program

int main() {
    float in[] = {/*4.362244606018066, 1.3265305757522583, 1.34183669090271, -0.0092373955994844436645508, */
                    6.103515625e-05,  // smallest normal number = 2^-14
                    5.340576172e-05,  // = 2^-15 x 1.11 = 0.111 x 2^-14
                    4.577636719e-05,
                    3.814697266e-05,
                    3.051757812e-05,
                    2.288818359e-05,
                    4.577636719e-06 };

    int exp_size = 5;
    int mant_size = 5; // includes sign and implicit 1
    int block_size = sizeof(in) / sizeof(in[0]);
    for (int i = 0; i < block_size; i++) {
        printf("%.15g -- INPUT\n", in[i]);
        dla_block_c_vec(in + i, 1, 5, 5, false, false);
        printf("%.15g -- OUTPUT\n\n", in[i]);
    }
    return 0;
}
#endif