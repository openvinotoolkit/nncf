#ifndef _BINARIZATION_FUNCTIONS_CUDA_IMPL_H_
#define _BINARIZATION_FUNCTIONS_CUDA_IMPL_H_

at::Tensor wb_cuda_forward(
        at::Tensor input,
        bool per_channel);

at::Tensor ab_cuda_forward(
        at::Tensor input,
        at::Tensor scale,
        at::Tensor thresholds);

std::vector<at::Tensor> ab_cuda_backward(
        at::Tensor grad_output,
        at::Tensor input,
        at::Tensor scale,
        at::Tensor output);

#endif // _BINARIZATION_FUNCTIONS_CUDA_IMPL_H_
