#include "common_cpu_funcs.h"
#include "common_defs.h"

#include "../../../include/quantization/dla_sw_model.h"

namespace {

template <typename scalar_t>
at::Tensor q_cpu_forward(
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        scalar_t levels) {
    at::Tensor s = (levels - 1) / input_range;
    auto output = at::max(at::min(input, input_low + input_range), input_low);
    output -= input_low;
    output *= s;
    output = output.round_();
    output = output.div_(s);
    output += input_low;
    return output;
}

template <typename scalar_t>
std::vector<at::Tensor> q_cpu_backward(
        at::Tensor grad_output,
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        scalar_t levels,
        scalar_t levels_low,
        scalar_t levels_high,
        bool is_asymmetric) {
    auto output = q_cpu_forward<scalar_t>(input, input_low, input_range, levels);
    auto reverted_range = 1 / input_range;
    scalar_t alpha = levels_low / levels_high;
    auto mask_hi = input.gt(input_low + input_range);
    auto mask_lo = input.lt(input_low);
    auto err = at::sub(output, input);
    err.mul_(reverted_range);
    err = err.masked_fill_(mask_hi, 1);
    err = err.masked_fill_(mask_lo, alpha);
    err = err.mul_(grad_output);

    auto grad_input_range = err;

    sum_like(grad_input_range, input_range);

    auto grad_input = grad_output.clone();
    auto outside_mask = mask_hi.add_(mask_lo);
    grad_input = grad_input.masked_fill_(outside_mask, 0);

    if (is_asymmetric) {
        auto grad_input_low = grad_output.clone();
        auto all_ones = torch::ones_like(outside_mask);
        grad_input_low = grad_input_low.masked_fill_(at::__xor__(all_ones, outside_mask), 0);

        sum_like(grad_input_low, input_low);
        return {grad_input, grad_input_low, grad_input_range};
    }
    auto dummy_variable = torch::autograd::make_variable(at::empty(input_low.sizes()), true);
    return {grad_input, dummy_variable, grad_input_range};
}

#define CHECK_INPUT(x) CHECK_CPU(x)

at::Tensor q_forward(
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels) {
    CHECK_INPUT(input);
    CHECK_INPUT(input_low);
    CHECK_INPUT(input_range);
    TORCH_CHECK(input_low.dim() == input_range.dim(), "input_low and input_range have different dimensionality");
    int64_t scale_dim = input_range.dim();
    for (int i = 0; i < scale_dim; i++)
    {
        TORCH_CHECK(input_low.size(i) == input_range.size(i), "input_low and input_range have different dimension sizes");
    }

    at::Tensor output;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "q_cpu_forward", ([&] {
      output = q_cpu_forward<scalar_t>(input, input_low, input_range, levels);
    }));

    return output;
}

std::vector<at::Tensor> q_backward(
        at::Tensor grad_output,
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels,
        int level_low,
        int level_high,
        bool is_asymmetric) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(input_low);
    CHECK_INPUT(input_range);

    std::vector<at::Tensor> results;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "q_cpu_backward", ([&] {
        results = q_cpu_backward<scalar_t>(grad_output, input, input_low, input_range, levels, level_low, level_high, is_asymmetric);
    }));

    return results;
}
///////////////////
/// BlockFP support
///////////////////
void block_align_floats_body(float* out, float* in, uint32_t exp_width,
    uint32_t mantissa_width, uint32_t block_size,  uint32_t N, uint32_t C, uint32_t HxW, bool sw_rnd, int blockIdx, int threadIdx) {

    
  int n = blockIdx / HxW;
  int hw = blockIdx % HxW;
  int c = threadIdx;
  int32_t max_exp = 0;
  for (int b = 0; b < block_size && (c * block_size + b < C); ++b) {
    int c_idx = c * block_size + b;
    int idx = (n * C + c_idx) * HxW + hw;
    uint32_t bits;      //violating strict aliasing!
    out[idx] = round_subnorm(in[idx], exp_width, mantissa_width, sw_rnd, false /* not input layer */);
    float *temp = &out[idx];
    bits = *((uint32_t *)temp);
    int32_t exp = ((bits >> 23) & 0xFF);
    if (exp > max_exp) {
      max_exp = exp;
    }
  }
  for (int b = 0; (b < block_size) && (c * block_size + b < C); ++b) {
    int c_idx = c * block_size + b;
    int idx = (n * C + c_idx) * HxW + hw;
    out[idx] = block_align(out[idx], max_exp, mantissa_width, exp_width, sw_rnd);
  }
}



at::Tensor bfp_forward(
    at::Tensor input,
    uint32_t exp_width, 
    uint32_t mantissa_width, 
    uint32_t block_size,
    uint32_t is_weights) {

    uint32_t N = input.size(0);
    uint32_t C = input.size(1);
    uint32_t HxW = 1;

    for (int d = input.dim()-1; d >=2; d--)
    {
      HxW *= input.size(d);
    }
    
    bool sw_rnd = is_weights;
    
    auto output = at::empty_like(input);
    for (int blockIdx = 0; blockIdx < N * HxW; blockIdx ++)
    {
        for (int threadIdx = 0; threadIdx < std::ceil(C/ (float) (block_size)); threadIdx++)
        {
            block_align_floats_body(
                (float*)output.data_ptr(), //output.data<scalar_t>(),
                (float*)input.data_ptr(),
                exp_width, mantissa_width, block_size, 
                N, C, HxW, sw_rnd, blockIdx, threadIdx);
        }
    }
    return output;
}

void block_align_floats_folded_body(float *out, float *in, uint32_t exp_width, uint32_t mantissa_width,
      uint32_t block_size, uint32_t N, uint32_t C, uint32_t H, uint32_t W, uint32_t SY, uint32_t SX, uint32_t PY, uint32_t PX, bool sw_rnd, int n, int h) {
    int32_t *max_exps = new int32_t[(int)ceil(C * SY * SX / (float) block_size)];
    int32_t w_end = ceil((W + 2 * PX) / (float) SX);
    for (int w = 0; w < w_end; ++w) {
      // ceil(C * SY * SX / block_size) is how many blocks there are
      int32_t block = 0;
  
      for (int i = 0; i < ceil(C * SY * SX / (float) block_size); ++i) {
        max_exps[i] = 0;
      }
  
      // Blocking along these loops
      for (int c = 0; c < C; ++c) {
        for (int sy = 0; sy < SY; ++sy) {
          for (int sx = 0; sx < SX; ++sx) {
            int h_idx = h * SY + sy - PY;
            int w_idx = w * SX + sx - PX;
            int idx = (((n * C) + c) * H + h_idx) * W + w_idx;
            if ((w_idx < W) &&
                (h_idx < H) &&
                (w_idx >= 0) &&
                (h_idx >= 0)) {
              uint32_t bits;
              out[idx] = round_subnorm(in[idx], exp_width, mantissa_width, sw_rnd, true /* input layer */);
              float *temp = &out[idx];
              bits = *((uint32_t *)temp);
              int32_t exp = ((bits >> 23) & 0xFF);
              if (exp > max_exps[block / block_size]) {
                max_exps[block / block_size] = exp;
              }
            }
            block++;
          }
        }
      }
      block = 0;
      for (int c = 0; c < C; ++c) {
        for (int sy = 0; sy < SY; ++sy) {
          for (int sx = 0; sx < SX; ++sx) {
            int h_idx = h * SY + sy - PY;
            int w_idx = w * SX + sx - PX;
            int idx = (((n * C) + c) * H + h_idx) * W + w_idx;
            if ((w_idx < W) && 
                (h_idx < H) &&
                (w_idx >= 0) &&
                (h_idx >= 0)) {
              out[idx] = block_align(out[idx], max_exps[block / block_size], mantissa_width, exp_width, sw_rnd);
            }
            block++;
          }
        }
      }
    }
    delete[] max_exps;
  }

at::Tensor bfp_forward_fold(at::Tensor input,
                uint32_t exp_width, 
                uint32_t mantissa_width, 
                uint32_t block_size,
                uint32_t is_weights,
                unsigned int PX,
                unsigned int PY,
                unsigned int strideX,
                unsigned int strideY) {


  auto output = at::empty_like(input);
  uint32_t N = input.size(0);
  uint32_t C = input.size(1);
  uint32_t H = input.size(2);
  uint32_t W = input.size(3);

  bool sw_rnd = is_weights;
    
//////

  for (int blockIdx = 0; blockIdx < N ; blockIdx ++)
  {
      for (int threadIdx = 0; threadIdx < std::ceil((H +  PY*2) / (float) strideY) ; threadIdx++)
      {
          block_align_floats_folded_body(
              (float*)output.data_ptr(), //output.data<scalar_t>(),
              (float*)input.data_ptr(),
              exp_width, mantissa_width, block_size, 
              N, C, H, W, strideY, strideX, PY, PX, sw_rnd, blockIdx, threadIdx);
      }
  }


///////

    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("Quantize_forward", &q_forward, "Quantize forward");
  m.def("Quantize_backward", &q_backward, "Quantize backward");
  m.def("Quantize_blockfp", &bfp_forward, "input");
  m.def("Quantize_blockfp_fold", &bfp_forward_fold, "input");
  
}
