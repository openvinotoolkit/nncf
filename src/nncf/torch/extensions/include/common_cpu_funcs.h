#ifndef _COMMON_CPU_FUNCS_H_
#define _COMMON_CPU_FUNCS_H_

#include <torch/torch.h>
#include "dispatch.h"

void sum_like(at::Tensor& target_tensor, const at::Tensor& ref_tensor);
void sum_to_act_channels(at::Tensor& target_tensor);

#endif // _COMMON_CPU_FUNCS_H_
