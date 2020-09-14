// FPGA hardware rounding model for FP9 (aka int5bfp blocking and FP16 auxiliary precision)

#define AUX_DATA_VALUE_MANTISSA_WIDTH     10

#define FP32_WIDTH 32
#define FP32_EXPONENT_WIDTH 8
#define FP32_MANTISSA_WIDTH 23
#define FP32_EXPONENT_BIAS 127

#define FP16_EXPONENT_WIDTH 5
#define FP16_EXPONENT_BIAS 15
#define FP16_EXPONENT_MAX_VALUE 31

#define ROUND_DURING_BLOCK_ALIGN

#include "fpga_common_utils.h"
#include "fpga_fp16.h"
#include "fpga_types_blockfp.h"


// Extract FP32 fields
// sign = fp32_bits[31]
// exp  = fp32_bits[30:23]
// man  = fp32_bits[22:0]
STATIC uint32_t extract_sign(float x) {
    uint32_t number = *(reinterpret_cast<const uint32_t*>(&x));
    uint32_t sign = (number & 0x80000000) >> (FP32_MANTISSA_WIDTH + FP32_EXPONENT_WIDTH);
    return sign;
}
STATIC uint32_t extract_mantissa(float x) {
    uint32_t number = *(reinterpret_cast<const uint32_t*>(&x));
    uint32_t mantissa = (number & 0x007fffff);
    return mantissa;
}
STATIC uint32_t extract_exponent(float x) {
    uint32_t number = *(reinterpret_cast<const uint32_t*>(&x));
    uint32_t exponent = (number & 0x7f800000) >> FP32_MANTISSA_WIDTH;
    return exponent;
}
STATIC float reassemble_float(uint32_t sign, uint32_t exponent, uint32_t mantissa, uint32_t input_mantissa_width) {

    int32_t exp;
    if (mantissa == 0) {
        exp = 0;
    }
    else {
        int pos = input_mantissa_width;
        for (int j = input_mantissa_width; j >= 0; j--) {
            if ((mantissa >> j) & 0x1) {
                pos = j;
                break;
            }
        }
        mantissa <<= (input_mantissa_width - pos);
        mantissa &= ((1 << input_mantissa_width) - 1);
        mantissa <<= (23 - input_mantissa_width);
        exp = exponent - (input_mantissa_width - pos);
    }

    uint32_t result_uint = (sign << (FP32_EXPONENT_WIDTH + FP32_MANTISSA_WIDTH)) |
        ((exp & BIT_MASK(FP32_EXPONENT_WIDTH)) << FP32_MANTISSA_WIDTH) |
        (mantissa & BIT_MASK(FP32_MANTISSA_WIDTH));

    float result = *(reinterpret_cast<float*>(&result_uint));
    return result;
}

STATIC uint block_align_mantissa(bool sign, uchar exponent, uint mantissa,
    uchar new_exponent, int input_width, int output_width,
    bool add_implicit_one) {

    if (add_implicit_one) mantissa |= (1 << input_width);
    int mantissa_width = add_implicit_one ? input_width + 1 : input_width;

    bool round_up = false;

    // shift mantissa into the output position, leaving one extra 0 in front to
    // represent the sign
    if (output_width >= (mantissa_width + 1)) {
        mantissa <<= (output_width - (mantissa_width + 1));
    }
    else {
        round_up = BIT_IS_SET(mantissa, (mantissa_width + 1) - output_width - 1);
        mantissa >>= ((mantissa_width + 1) - output_width);
    }

    // shift mantissa to line up with the common exponent
    uchar shift = (new_exponent - exponent);
#ifdef ROUND_DURING_BLOCK_ALIGN
    if (shift != 0) {
        // NOTE: the bit mask has a barrel shifter.
        bool all_zeros_shifted_out = ((mantissa & BIT_MASK(shift - 1)) == 0);
        mantissa >>= shift - 1;
        round_up = BIT_IS_SET(mantissa, 0);
        // Round to even LSB to break tie: do not round if the LSB is 0
        if (all_zeros_shifted_out) {
            round_up = round_up && BIT_IS_SET(mantissa, 1);
        }
        mantissa >>= 1;
    }

    // Round
    uint rounded_mantissa = mantissa + round_up;
    if (BIT_IS_SET(rounded_mantissa, output_width - 1)) {
        // Undo rounding: don't round up if the truncated mantissa was all 1s
        rounded_mantissa = mantissa;
    }
    mantissa = rounded_mantissa & BIT_MASK(output_width - 1);

#else
    mantissa >>= shift;
#endif

    // handle subnormals and zero
    if (exponent == 0) {
        mantissa = 0;
    }

    return mantissa & BIT_MASK(output_width);
}


STATIC uint transform_input_one(ushort raw_input, int output_width, uchar max_exponent) {

    ushort raw_value = raw_input;
    bool sign = BIT_SEL(raw_value, 15, 15);
    uchar exponent = BIT_SEL(raw_value, 14, 10);
    uint mantissa = BIT_SEL(raw_value, 9, 0);

    // std::cout << "transform: " << "exp=" << (int)exponent << ", mant=" << mantissa << ", max_exp=" << (int)max_exponent << "\n";
    uint out_mantissa = block_align_mantissa(sign, exponent, mantissa,
        max_exponent, /* input_width */ 10,
        /* output_width */ output_width,
        /* add_implicit_one */ true);

    return out_mantissa;
}

STATIC float dla_v1_hw_round_subnorm(float in, uint32_t exp_width, int mantissa_width, bool is_input_layer) {

    // exp_width is unused directly by this function. 
    // Conversion functions below (esp. float_to_half()) assume that
    // exponent width is FP16_EXPONENT_WIDTH;
    assert(exp_width == FP16_EXPONENT_WIDTH);

    // full conversion is: FP32 -> FP16 
    half_t step1 = float_to_half(in);

    float step2;
    int mantissa_lsb = 10 - mantissa_width;


    if (is_input_layer)
    {
        // INPUT IMAGE ONLY
        // Input image only, FP16 -> FP9 conversion is done by truncation.
        // raw_to_data_value will re-interpret FP16 value as an FP9 value
        step2 = dla_type_to_float(raw_to_dla_type(half_as_ushort(step1), mantissa_lsb), mantissa_lsb);
    }
    else
    {
        // ALL OTHER ACTIVATIONS 
        // half_as_ushort                 does bit re-interpretations
        // raw_to_aux_data_value          does bit re-interpretations.
        // aux_data_value_to_data_value() does FP16 -> FP9 conversion. 
        // data_value_to_float()          does FP9 -> FP32.
        step2 = dla_type_to_float(raw_to_rounded_dla_type(half_as_ushort(step1), mantissa_lsb), mantissa_lsb);
    }

    return step2;
}


STATIC float dla_v1_hw_block_align(float in, uint32_t max_exp, uint32_t mantissa_width, uint32_t exp_width) {

    // max_exp here will be biased to FP32. Re-bias to new exp width 
    uint32_t rebiased_max_exp = max_exp - FP32_EXPONENT_BIAS + FP16_EXPONENT_BIAS;
    int mantissa_lsb = 10 - mantissa_width;

    // line below is just bit-reinterpretation. Rik - really? Not for generic float input, perhaps if float is already preconditioned
    ushort in_raw = dla_type_to_raw(float_to_dla_type(in, mantissa_lsb), mantissa_lsb);
    // FP9 -> blocked FP9
    uint new_mantissa = transform_input_one(in_raw, mantissa_width, rebiased_max_exp);

    uint32_t sign = extract_sign(in);

    // mantissa_width includes sign. reassemble_float needs mantissa_width without the sign bit.
    float result = reassemble_float(sign, max_exp, new_mantissa, mantissa_width - 2);

    //std::cout << "dla_v1_block_align: in_raw=" << in_raw << ", new_mantissa=" << new_mantissa << "\n";
    //std::cout << "dla_v1_block_align: " << in << ", exp=" << rebiased_max_exp << " -> " << result << "\n";
    return result;
}
