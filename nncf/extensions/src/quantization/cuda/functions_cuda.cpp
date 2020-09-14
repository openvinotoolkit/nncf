#include <torch/torch.h>
#include <vector>

#include "common_defs.h"
#include "quantization/functions_cuda_impl.h"

#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor bfp_cuda_forward(
        at::Tensor input,
        unsigned int exponentBits, 
        unsigned int mantissaBits, 
        unsigned int blocksize,
        unsigned int isWeights);

at::Tensor bfp_cuda_forward_fold(
        at::Tensor input,
        unsigned int exponentBits, 
        unsigned int mantissaBits, 
        unsigned int blocksize,
        unsigned int isWeights,
        unsigned int padX,
        unsigned int padY, 
        unsigned int fold3dW,
        unsigned int fold3dH);

at::Tensor q_forward(
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels) {
    CHECK_INPUT(input);
    CHECK_INPUT(input_low);
    CHECK_INPUT(input_range);

    return q_cuda_forward(input, input_low, input_range, levels);
}

at::Tensor bfp_forward(
        at::Tensor input,
        unsigned int exponentBits, 
        unsigned int mantissaBits, 
        unsigned int blocksize,
        unsigned int isWeights) {
    CHECK_INPUT(input);
    return bfp_cuda_forward(input, exponentBits, mantissaBits, blocksize, isWeights);
}

at::Tensor bfp_forward_fold(
        at::Tensor input,
        unsigned int exponentBits, 
        unsigned int mantissaBits, 
        unsigned int blocksize,
        unsigned int isWeights,
        unsigned int offsetX,
        unsigned int offsetY, 
        unsigned int foldW,
        unsigned int foldH) {
    CHECK_INPUT(input);
    return bfp_cuda_forward_fold(input, exponentBits, mantissaBits, blocksize, isWeights, offsetX, offsetY, foldW, foldH);
}

std::vector<at::Tensor> q_backward(
        at::Tensor grad_output,
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels,
        int level_low,
        int level_high) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(input_low);
    CHECK_INPUT(input_range);
    return q_cuda_backward(grad_output, input, input_low, input_range, levels, level_low, level_high);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("Quantize_forward", &q_forward, "Quantize forward");
  m.def("Quantize_backward", &q_backward, "Quantize backward");
  m.def("Quantize_blockfp", &bfp_forward, "Blockfp forward");
  m.def("Quantize_blockfp_fold", &bfp_forward_fold, "Blockfp forward fold");
  
  
}
