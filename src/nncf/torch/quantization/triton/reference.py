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


def get_4d_tensor_meta(x: torch.tensor) -> torch.tensor:
    """
    Helper function for meta information creation.

    :param x: Torch tensor.
    :returns: Torch tensor as meta with 4D shape + tensor.
    """
    shape = list(x.shape)
    stride = list(x.stride())
    size = len(shape)

    for i in range(4):
        if i >= size:
            shape += [1]
            stride += [0]
        elif shape[i] == 1:
            stride[i] = 0

    return torch.tensor(shape + stride, dtype=torch.int32).to(x.device)


@triton.jit
def read_shape(meta: torch.tensor) -> tuple[tl.tensor]:
    """
    Helper kernel for the shapes loading from meta tensor.

    :param meta: Torch tensor with meta information (shape + stride).
    :returns: Tuple of 4D shapes.
    """
    s0 = tl.load(meta + 0)
    s1 = tl.load(meta + 1)
    s2 = tl.load(meta + 2)
    s3 = tl.load(meta + 3)
    return s0, s1, s2, s3


@triton.jit
def read_stride(meta: torch.tensor) -> tuple[tl.tensor]:
    """
    Helper kernel for the strides loading from meta tensor.

    :param meta: Torch tensor with meta information (shape + stride).
    :returns: Tuple of 4D strides.
    """
    st0 = tl.load(meta + 4)
    st1 = tl.load(meta + 5)
    st2 = tl.load(meta + 6)
    st3 = tl.load(meta + 7)
    return st0, st1, st2, st3


@triton.jit
def calculate_total_elements(meta: torch.tensor) -> tl.tensor:
    """
    Helper kernel for the total elements calculation based on meta tensor.

    :param meta: Torch tensor with meta information (shape + stride).
    :returns: Total number of elements for mask calculation.
    """
    s0, s1, s2, s3 = read_shape(meta)
    return s0 * s1 * s2 * s3


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 256}),
        triton.Config(kwargs={"BLOCK_SIZE": 512}),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}),
    ],
    key=["BLOCK_SIZE"],
)
@triton.jit
def forward_kernel(
    input__ptr: torch.tensor,
    input__meta: torch.tensor,
    input_low_ptr: torch.tensor,
    input_low_meta: torch.tensor,
    input_range_ptr: torch.tensor,
    input_range_meta: torch.tensor,
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
    input__s0, input__s1, input__s2, input__s3 = read_shape(input__meta)
    input__elements = input__s0 * input__s1 * input__s2 * input__s3

    tmp = offsets
    i3 = tmp % input__s3
    tmp //= input__s3
    i2 = tmp % input__s2
    tmp //= input__s2
    i1 = tmp % input__s1
    tmp //= input__s1
    i0 = tmp % input__s0

    input_low_st0, input_low_st1, input_low_st2, input_low_st3 = read_stride(input_low_meta)
    input_low_offset = i0 * input_low_st0 + i1 * input_low_st1 + i2 * input_low_st2 + i3 * input_low_st3
    input_low_elements = calculate_total_elements(input_low_meta)

    input_range_st0, input_range_st1, input_range_st2, input_range_st3 = read_stride(input_range_meta)
    input_range_offset = i0 * input_range_st0 + i1 * input_range_st1 + i2 * input_range_st2 + i3 * input_range_st3
    input_range_elements = calculate_total_elements(input_range_meta)

    input_ = tl.load(input__ptr + offsets, mask=offsets < input__elements).to(tl.float32)
    input_low = tl.load(input_low_ptr + input_low_offset, mask=input_low_offset < input_low_elements).to(tl.float32)
    input_range = tl.load(input_range_ptr + input_range_offset, mask=input_range_offset < input_range_elements).to(
        tl.float32
    )

    scale = (levels - 1) / input_range

    output = tl.clamp(input_, min=input_low, max=input_low + input_range)

    zero_point = libdevice.nearbyint(-input_low * scale)
    output -= input_low
    output *= scale
    output -= zero_point
    output = libdevice.nearbyint(output)
    output = output / scale

    tl.store(output_ptr + offsets, output, mask=offsets < input__elements)


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 256}),
        triton.Config(kwargs={"BLOCK_SIZE": 512}),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}),
    ],
    key=["BLOCK_SIZE"],
)
@triton.jit
def backward_kernel(
    grad_output_ptr: torch.tensor,
    grad_output_meta: torch.tensor,
    input__ptr: torch.tensor,
    input__meta: torch.tensor,
    input_low_ptr: torch.tensor,
    input_low_meta: torch.tensor,
    input_range_ptr: torch.tensor,
    input_range_meta: torch.tensor,
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
    input__s0, input__s1, input__s2, input__s3 = read_shape(input__meta)
    input__elements = input__s0 * input__s1 * input__s2 * input__s3

    tmp = offsets
    i3 = tmp % input__s3
    tmp //= input__s3
    i2 = tmp % input__s2
    tmp //= input__s2
    i1 = tmp % input__s1
    tmp //= input__s1
    i0 = tmp % input__s0

    input_low_st0, input_low_st1, input_low_st2, input_low_st3 = read_stride(input_low_meta)
    input_low_offset = i0 * input_low_st0 + i1 * input_low_st1 + i2 * input_low_st2 + i3 * input_low_st3
    input_low_elements = calculate_total_elements(input_low_meta)

    input_range_st0, input_range_st1, input_range_st2, input_range_st3 = read_stride(input_range_meta)
    input_range_offset = i0 * input_range_st0 + i1 * input_range_st1 + i2 * input_range_st2 + i3 * input_range_st3
    input_range_elements = calculate_total_elements(input_range_meta)

    grad_output = tl.load(grad_output_ptr + offsets, mask=offsets < input__elements).to(tl.float32)
    input_ = tl.load(input__ptr + offsets, mask=offsets < input__elements).to(tl.float32)
    input_low = tl.load(input_low_ptr + input_low_offset, mask=input_low_offset < input_low_elements).to(tl.float32)
    input_range = tl.load(input_range_ptr + input_range_offset, mask=input_range_offset < input_range_elements).to(
        tl.float32
    )

    mask_hi = input_ > (input_low + input_range)
    mask_hi = mask_hi.to(tl.float32)
    mask_lo = input_ < input_low
    mask_lo = mask_lo.to(tl.float32)

    mask_in = 1 - mask_hi - mask_lo

    scale = (levels - 1) / input_range
    output = tl.clamp(input_, min=input_low, max=input_low + input_range)
    zero_point = libdevice.nearbyint(-input_low * scale)
    output -= input_low
    output *= scale
    output -= zero_point
    output = libdevice.nearbyint(output)
    output = output / scale

    # Signed range calculation
    input_range_above_zero = input_range > 0
    input_range_below_zero = input_range < 0
    range_sign = input_range_above_zero - input_range_below_zero
    # Reciprocal calculation
    reciprocal = 1 / (input_range * range_sign)
    err = (output - input_) * reciprocal
    grad_range = grad_output * (err * mask_in + range_sign * (level_low / level_high) * mask_lo + mask_hi)
    grad_range = tl.sum(grad_range, keep_dims=True)

    grad_input = grad_output * mask_in

    grad_low = grad_output * (mask_hi + mask_lo)
    grad_low = tl.sum(grad_low, keep_dims=True)

    tl.store(grad_input_ptr + offsets, grad_input, mask=offsets < input__elements)
    tl.store(grad_low_ptr + input_low_offset, grad_low, mask=input_low_offset < input_low_elements)
    tl.store(grad_range_ptr + input_range_offset, grad_range, mask=input_range_offset < input_range_elements)


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

    input__meta = get_4d_tensor_meta(input_)
    input_low_meta = get_4d_tensor_meta(input_low)
    input_range_meta = get_4d_tensor_meta(input_range)

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

    grad_output_meta = get_4d_tensor_meta(grad_output)
    input__meta = get_4d_tensor_meta(input_)
    input_low_meta = get_4d_tensor_meta(input_low)
    input_range_meta = get_4d_tensor_meta(input_range)

    with torch.cuda.device(input_.device):
        grid = lambda meta: (
            triton.cdiv(input_.numel(), meta["BLOCK_SIZE"])
            * triton.cdiv(input_low.numel(), meta["BLOCK_SIZE"])
            * triton.cdiv(input_range.numel(), meta["BLOCK_SIZE"]),
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
