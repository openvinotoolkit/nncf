#include "dla_hw_model.h"
#include "dla_floating_point_utils.h"


// When we have a lower number of exponent bits, we may need to deal with floats becoming subnormal
// Unclear what happens if the exponent is too big
STATIC float dla_v1_sw_round_subnorm(float in, uint32_t exp_width) {
    float *temp = &in;
    uint32_t bits = *((uint32_t *)temp);
    int32_t exp = ((bits >> 23) & 0xFF);
    uint32_t mantissa = bits & ((1 << 23) - 1);

    int32_t min_exp = -1 * (1 << (exp_width - 1)) + 1 + 127;

    if (exp == min_exp) {
        bool grd = 1;
        bool rnd = (mantissa >> 22) & 0x1;
        uint32_t sticky_mask = (1 << 22) - 1;
        bool sticky = (mantissa & sticky_mask) != 0;
        if ((grd && rnd) || (grd && sticky)) {
            exp = exp + 1;
            mantissa = 0;
        }
        else {
            exp = 0;
            mantissa = 0;
        }
    }
    else if (exp < min_exp) {
        exp = 0;
        mantissa = 0;
    }
    uint32_t sign_t = in < 0.0;
    uint32_t hbits = ((sign_t << 31) | ((exp) << 23) | mantissa);
    float result;
    result = *((float *)&hbits);
    return result;
}

STATIC float dla_v1_sw_block_align(float in_out, uint32_t max_exp, uint32_t mantissa_width,
    uint32_t exp_width) {
    float *temp = &in_out;
    uint32_t bits = *((uint32_t *)temp);
    int32_t exp = ((bits >> 23) & 0xFF);
    uint32_t mantissa = bits & ((1 << 23) - 1);

    int32_t diff_exp = max_exp - exp;

    bool grd = false;
    bool rnd = false;
    bool sticky = false;

    if (diff_exp == 0 && exp != 0) {
        if (mantissa_width <= 22) {
            grd = (mantissa >> (22 - mantissa_width)) & 0x1;
        }
        if (mantissa_width <= 21) {
            rnd = (mantissa >> (21 - mantissa_width)) & 0x1;
        }
        if (mantissa_width <= 20) {
            uint32_t sticky_mask = (1 << (21 - mantissa_width)) - 1;
            sticky = (mantissa & sticky_mask) != 0;
        }

        mantissa = (mantissa >> (23 - mantissa_width));
        bool lsb = mantissa & 0x1;
        // Don't change the exp since we're block aligning
        if (mantissa != ((1 << (mantissa_width)) - 1)
            && ((grd && rnd) || (grd && sticky) || (grd && lsb))) {
            mantissa += 1;
        }
        mantissa <<= (23 - mantissa_width);
    }
    else if (diff_exp != 0 && exp != 0) {
        mantissa = mantissa | (1 << 23);

        int32_t shift = 23 - mantissa_width + (diff_exp - 3);

        if (shift >= 0) {
            grd = (mantissa >> (shift + 2)) & 0x1;
            rnd = (mantissa >> (shift + 1)) & 0x1;
            uint32_t sticky_mask = (1 << (shift + 1)) - 1;
            sticky = (mantissa & sticky_mask) != 0;
        }

        mantissa = (mantissa >> (23 - mantissa_width));
        mantissa >>= diff_exp;
        bool lsb = mantissa & 0x1;
        if ((mantissa != ((1 << (mantissa_width + 1)) - 1)) &&
            ((grd && rnd) || (grd && sticky) || (grd && lsb))) {
            mantissa += 1;
        }
        if (mantissa == 0) {
            exp = 0;
        }
        else {
            int pos = mantissa_width;
            for (int j = mantissa_width; j >= 0; j--) {
                if ((mantissa >> j) & 0x1) {
                    pos = j;
                    break;
                }
            }
            mantissa <<= (mantissa_width - pos);
            mantissa &= ((1 << mantissa_width) - 1);
            mantissa <<= (23 - mantissa_width);
            exp = max_exp - (mantissa_width - pos);
        }
    }
    else if (exp == 0) {
        mantissa = 0;
    }

    uint32_t sign_t = in_out < 0.0;
    uint32_t hbits = ((sign_t << 31) | ((exp) << 23) | mantissa);
    float result;
    result = *((float *)&hbits);
    return result;
}

STATIC float round_subnorm(float in, uint32_t exp_width, uint32_t mantissa_width, bool sw_rnd, bool is_input_layer) {
    float rounded;
    if (sw_rnd) {
        rounded = dla_v1_sw_round_subnorm(in, exp_width);
    }
    else {
        rounded = dla_v1_hw_round_subnorm(in, exp_width, mantissa_width, is_input_layer);
    }
    return rounded;
}

STATIC float block_align(float in, uint32_t max_exp, uint32_t mantissa_width,
    uint32_t exp_width, bool sw_rnd) {
    float aligned;
    if (sw_rnd) {
        aligned = dla_v1_sw_block_align(in, max_exp, mantissa_width, exp_width);
    }
    else {
        // dla_v1_hw code, for mantissa_width, needs total number bits, including implicit 1 and sign
        aligned = dla_v1_hw_block_align(in, max_exp, mantissa_width, exp_width);
    }
    return aligned;
}


#ifndef __NVCC__
STATIC void dla_block_c_vec(float *inout, uint32_t block_size, uint32_t exp_width, uint32_t mantissa_width, bool sw_rnd, bool is_input_layer) {

    uint32_t max_exp = 0;
    for (uint32_t i = 0; i < block_size; i++) {
        inout[i] = round_subnorm(inout[i], exp_width, mantissa_width, sw_rnd, is_input_layer);
        float *temp = &inout[i];
        uint32_t bits = *((uint32_t *)temp);
        uint32_t exp = ((bits >> 23) & 0xFF);
        if (exp > max_exp) {
            max_exp = exp;
        }
    }
    for (uint32_t i = 0; i < block_size; i++) {
        inout[i] = block_align(inout[i], max_exp, mantissa_width, exp_width, sw_rnd);
    }
}

#else // NVCC defined -> CUDA-specific code
__device__ void dla_block_c_vec_cuda(float *inout, uint32_t *max_exp, uint32_t idx, uint32_t block_size, uint32_t exp_width, uint32_t mantissa_width, bool sw_rnd, bool is_input_layer) {

    // This must be PER BLOCK (for folded version)
    // __shared__ uint32_t max_exp;
    uint32_t i = idx;

    inout[i] = round_subnorm(inout[i], exp_width, mantissa_width, sw_rnd, is_input_layer);

    __syncthreads();

    // only first thread calculates max exp. not bothering with fancy reduction here
    if (i == 0) {
        *max_exp = 0;
        for (uint32_t b = 0; b < block_size; b++) {
            float *temp = &inout[b];
            uint32_t bits = *((uint32_t *)temp);
            uint32_t exp = ((bits >> 23) & 0xFF);
            if (exp > *max_exp) {
                *max_exp = exp;
            }
        }
    }

    __syncthreads();

    inout[i] = block_align(inout[i], *max_exp, mantissa_width, exp_width, sw_rnd);
}
#endif
