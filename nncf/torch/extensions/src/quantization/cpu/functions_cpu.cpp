#include "common_cpu_funcs.h"
#include "common_defs.h"

namespace {

template <typename scalar_t>
at::Tensor q_cpu_forward(
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        scalar_t levels) {
    at::Tensor s = (levels - 1) / input_range;
    auto output = at::max(at::min(input, input_low + input_range), input_low);
    // zero_point is referred as ZP in docs
    auto zero_point = (-input_low * s).round_();
    output -= input_low;
    output *= s;
    output -= zero_point;
    output = output.round_();
    output = output.div_(s);
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

    DISPATCH_TENSOR_DATA_TYPES(input.type(), "q_cpu_forward", ([&] {
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
    DISPATCH_TENSOR_DATA_TYPES(input.type(), "q_cpu_backward", ([&] {
        results = q_cpu_backward<scalar_t>(grad_output, input, input_low, input_range, levels, level_low, level_high, is_asymmetric);
    }));

    return results;
}


}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("Quantize_forward", &q_forward, "Quantize forward");
  m.def("Quantize_backward", &q_backward, "Quantize backward");
}
