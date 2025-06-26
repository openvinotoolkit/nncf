#include "common_cpu_funcs.h"

void sum_like(at::Tensor& target_tensor, const at::Tensor& ref_tensor)
{
    if (target_tensor.numel() == 1)
    {
        target_tensor = target_tensor.sum().view_as(ref_tensor);
    }
    else
    {
        auto dim_count = ref_tensor.dim();
        for (int64_t dim_idx = 0; dim_idx < dim_count; dim_idx++)
        {
            if (ref_tensor.size(dim_idx) == 1)
            {
                target_tensor = target_tensor.sum(dim_idx, true);
            }
        }
    }
}

void sum_to_act_channels(at::Tensor& target_tensor)
{
    // Sum over N
    target_tensor = target_tensor.sum(0, /*keepdims=*/ true);

    // Sum over H, W and the rest
    auto dim_count = target_tensor.dim();
    for (int64_t dim_idx = 2; dim_idx < dim_count; dim_idx++)
    {
        target_tensor = target_tensor.sum(dim_idx, /*keepdims=*/ true);
    }
}
