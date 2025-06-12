# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice


def get_tensor_meta(x: torch.tensor) -> torch.tensor:
    """
    Helper function for the 4D shape & stride calculation based on tensor.

    :param x: Torch tensor.
    :return: Torch tensor with merged shape and stride for input tensor.
    """
    shape = list(x.shape)
    stride = list(x.stride())
    while len(shape) < 4:
        shape = [1] + shape
        stride = [0] + stride
    return torch.tensor(shape + stride, dtype=torch.int32).to(x.device)


@triton.jit
def calculate_ptr_offsets(offsets: tl.tensor, meta_ptr: torch.tensor, BLOCK_SIZE: tl.constexpr) -> tuple[tl.tensor]:
    """
    Helper kernel to compute offsets for each tensor based on meta information.

    :param offsets: Offsets for the current run.
    :param meta_ptr: Pointer to the meta information for current tensor.
    :param BLOCK_SIZE: Block size of the current run.
    :return: Tuple with the updated offsets and calculated pointers.
    """
    pointers = tl.full([BLOCK_SIZE], 0, tl.int32)
    size = 4
    for i in tl.range(0, size):
        shape = tl.load(meta_ptr + i)
        stride = tl.load(meta_ptr + i + size)
        index = offsets % shape
        offsets //= shape

        pointers += index * stride
    offsets *= tl.load(meta_ptr + (size - 1))
    return offsets, pointers


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 128}),
    ],
    key=["BLOCK_SIZE"],
)
@triton.jit
def forward_kernel(
    input__ptr: torch.tensor,
    input__meta_ptr: torch.tensor,
    input_low_ptr: torch.tensor,
    input_low_meta_ptr: torch.tensor,
    input_range_ptr: torch.tensor,
    input_range_meta_ptr: torch.tensor,
    levels: int,
    output_ptr: torch.tensor,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """
    "
    Forward kernel implementation based on reference formula - nncf/torch/quantization/reference.py

    :param input__ptr: Memory pointer to input_ torch.tensor.
    :param input_low_ptr: Memory pointer to input_low torch.tensor.
    :param input_range_ptr: Memory pointer to input_range torch.tensor.
    :param levels: Levels value as scalar.
    :param output_ptr: Memory pointer to output torch.tensor that would be filled with return value.
    :param last_dim: Scalar to calculate loading offset for input_low/range pointers.
    :param is_per_tensor: Bool value for offset correction in per-tensor case.
    :param BLOCK_SIZE: Size of the memory block for current process.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tmp = offsets

    tmp, input__ptrs = calculate_ptr_offsets(
        tmp,
        input__meta_ptr,
        BLOCK_SIZE,
    )

    tmp, input_low_ptrs = calculate_ptr_offsets(
        tmp,
        input_low_meta_ptr,
        BLOCK_SIZE,
    )

    tmp, input_range_ptrs = calculate_ptr_offsets(
        tmp,
        input_range_meta_ptr,
        BLOCK_SIZE,
    )

    mask = tl.full([BLOCK_SIZE], True, tl.int1)

    input_ = tl.load(input__ptr + input__ptrs, mask)
    input_low = tl.load(input_low_ptr + input_low_ptrs, mask)
    input_range = tl.load(input_range_ptr + input_range_ptrs, mask)

    # Clip operation
    output_clip_ = tl.maximum(input_, input_low)
    input_high = input_low + input_range
    output_clip = tl.minimum(output_clip_, input_high)

    # Input low from output subtraction
    output_sub_1 = output_clip - input_low

    # Scale calculation
    one = 1.0
    scale_ = levels - one
    scale = scale_ / input_range

    # Output scaling
    output_scale = output_sub_1 * scale

    # Zero point calculation
    s_input_low = -input_low
    zero_point_ = s_input_low * scale
    zero_point = libdevice.nearbyint(zero_point_)

    # Zero point from output subtraction
    output_sub_2 = output_scale - zero_point

    # Output descaling
    output_ = libdevice.nearbyint(output_sub_2)
    output = output_ / scale

    tl.store(output_ptr + input__ptrs, output, mask)


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 128}),
    ],
    key=["BLOCK_SIZE"],
)
@triton.jit
def backward_kernel(
    grad_output_ptr: torch.tensor,
    grad_output_meta_ptr: torch.tensor,
    input__ptr: torch.tensor,
    input__meta_ptr: torch.tensor,
    input_low_ptr: torch.tensor,
    input_low_meta_ptr: torch.tensor,
    input_range_ptr: torch.tensor,
    input_range_meta_ptr: torch.tensor,
    levels: int,
    level_low: int,
    level_high: int,
    grad_input_ptr: torch.tensor,
    grad_low_ptr: torch.tensor,
    grad_range_ptr: torch.tensor,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """
    "
    Backward kernel implementation based on reference formula - nncf/torch/quantization/reference.py
    :param grad_output_ptr: Memory pointer to grad_output torch.tensor.
    :param input__ptr: Memory pointer to input_ torch.tensor.
    :param input_low_ptr: Memory pointer to input_low torch.tensor.
    :param input_range_ptr: Memory pointer to input_range torch.tensor.
    :param levels: Levels value as scalar.
    :param level_low: Level low value as scalar.
    :param level_high: Level high value as scalar.
    :param grad_input_ptr: Memory pointer to grad_input torch.tensor that would be filled with return value.
    :param grad_low_ptr: Memory pointer to grad_low torch.tensor that would be filled with return value.
    :param grad_range_ptr: Memory pointer to grad_range torch.tensor that would be filled with return value.
    :param last_dim: Scalar to calculate loading offset for input_low/range pointers.
    :param is_per_tensor: Bool value for offset correction in per-tensor case.
    :param BLOCK_SIZE: Size of the memory block for current process.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tmp = offsets

    tmp, grad_output_ptrs = calculate_ptr_offsets(
        tmp,
        grad_output_meta_ptr,
        BLOCK_SIZE,
    )

    tmp, input__ptrs = calculate_ptr_offsets(
        tmp,
        input__meta_ptr,
        BLOCK_SIZE,
    )

    tmp, input_low_ptrs = calculate_ptr_offsets(
        tmp,
        input_low_meta_ptr,
        BLOCK_SIZE,
    )

    tmp, input_range_ptrs = calculate_ptr_offsets(
        tmp,
        input_range_meta_ptr,
        BLOCK_SIZE,
    )

    mask = tl.full([BLOCK_SIZE], True, tl.int1)

    grad_output = tl.load(grad_output_ptr + grad_output_ptrs, mask)
    input_ = tl.load(input__ptr + input__ptrs, mask)
    input_low = tl.load(input_low_ptr + input_low_ptrs, mask)
    input_range = tl.load(input_range_ptr + input_range_ptrs, mask)

    # Mask high calculation
    input_high = input_low + input_range
    mask_hi = input_ > input_high

    # Mask low calculation
    mask_lo = input_ < input_low

    # Mask in calculation
    mask_c = 1.0
    mask_in_ = mask_c - mask_hi
    mask_in = mask_in_ - mask_lo

    # Output calculation/forward kernel implementation
    #   Clip operation
    output_clip_ = tl.maximum(input_, input_low)
    output_clip = tl.minimum(output_clip_, input_high)

    #   Input low from output subtraction
    output_sub_1 = output_clip - input_low

    #   Scale calculation
    one = 1.0
    scale_ = levels - one
    scale = scale_ / input_range

    #   Output scaling
    output_scale = output_sub_1 * scale

    #   Zero point calculation
    s_input_low = -input_low
    zero_point_ = s_input_low * scale
    zero_point = libdevice.nearbyint(zero_point_)

    #   Zero point from output subtraction
    output_sub_2 = output_scale - zero_point

    #   Output descaling
    output_ = libdevice.nearbyint(output_sub_2)
    output = output_ / scale

    # Error calculation
    err_ = output - input_

    # Signed range calculation
    zeros = tl.full([1], 0, tl.int32)
    input_range_above_zero_ = zeros < input_range
    input_range_above_zero = input_range_above_zero_.to(tl.int8)
    input_range_below_zero_ = input_range < zeros
    input_range_below_zero = input_range_below_zero_.to(tl.int8)
    range_sign_ = input_range_above_zero - input_range_below_zero
    range_sign = range_sign_.to(input_range.dtype)

    # Reciprocal calculation
    reciprocal_ = input_range * range_sign
    reciprocal = one / reciprocal_

    err = err_ * reciprocal

    # Range gradient calculation
    err_mask_in_ = err * mask_in
    level_low_level_high_div = level_low / level_high
    range_levels_ = range_sign * level_low_level_high_div
    range_mask_lo_ = range_levels_ * mask_lo
    err_range_ = err_mask_in_ + range_mask_lo_
    grad_range_ = err_range_ + mask_hi
    grad_range = grad_output * grad_range_

    # Input gradient calculation
    mask_hi_lo = mask_hi + mask_lo
    grad_input = grad_output * mask_in

    # Low gradient calculation
    grad_low = grad_output * mask_hi_lo

    tl.store(grad_input_ptr + input__ptrs, grad_input, mask)
    tl.store(grad_low_ptr + input_low_ptrs, grad_low, mask)
    tl.store(grad_range_ptr + input_range_ptrs, grad_range, mask)


def forward(input_: torch.tensor, input_low: torch.tensor, input_range: torch.tensor, levels: int) -> torch.tensor:
    """
    Wrapper for the forward kernel.
    It contains preparation steps like output memory allocation via tensor creation,
    additional values calculation and CUDA context management based on the input tensors.
    :param input_: input_ as torch.tensor.
    :param input_low: input_low as torch.tensor.
    :param input_range: input_range as torch.tensor.
    :param levels: Levels value.
    :return: Calculated output value as torch.tensor.
    """
    output = torch.empty_like(input_)

    input__meta = get_tensor_meta(input_)
    input_low_meta = get_tensor_meta(input_low)
    input_range_meta = get_tensor_meta(input_range)

    with torch.cuda.device(input_.device):
        grid = lambda meta: (triton.cdiv(input_.numel(), meta["BLOCK_SIZE"]),)
        forward_kernel[grid](
            input_,
            input__meta,
            input_low,
            input_low_meta,
            input_range,
            input_range_meta,
            levels,
            output,
        )

    return output


def backward(
    grad_output: torch.tensor,
    input_: torch.tensor,
    input_low: torch.tensor,
    input_range: torch.tensor,
    levels: int,
    level_low: int,
    level_high: int,
    is_asymmetric: bool = False,
) -> tuple[torch.tensor]:
    """
    Wrapper for the backward kernel.
    It contains preparation steps like output memory allocation via tensor creation,
    additional values calculation and CUDA context management based on the input tensors.
    :param grad_output: input_ as torch.tensor.
    :param input_: input_ as torch.tensor.
    :param input_low: input_low as torch.tensor.
    :param input_range: input_range as torch.tensor.
    :param levels: Levels value.
    :return: Calculated grad_input, grad_low and grad_range as tuple of torch.tensor values.
    """
    grad_input = torch.empty_like(input_)
    grad_low = torch.empty_like(input_low)
    grad_range = torch.empty_like(input_range)

    grad_output_meta = get_tensor_meta(grad_output)
    input__meta = get_tensor_meta(input_)
    input_low_meta = get_tensor_meta(input_low)
    input_range_meta = get_tensor_meta(input_range)

    with torch.cuda.device(input_.device):
        grid = lambda meta: (
            triton.cdiv(input_.numel(), meta["BLOCK_SIZE"])
            * triton.cdiv(input_low.numel(), meta["BLOCK_SIZE"])
            * triton.cdiv(grad_range.numel(), meta["BLOCK_SIZE"]),
        )

        backward_kernel[grid](
            grad_output,
            grad_output_meta,
            input_,
            input__meta,
            input_low,
            input_low_meta,
            input_range,
            input_range_meta,
            levels,
            level_low,
            level_high,
            grad_input,
            grad_low,
            grad_range,
        )

    return grad_input, grad_low, grad_range
