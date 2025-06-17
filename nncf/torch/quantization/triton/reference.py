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
    input_low_ptr: torch.tensor,
    input_range_ptr: torch.tensor,
    levels: int,
    output_ptr: torch.tensor,
    diff: int,
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
    offsets_d = offsets // diff

    mask = tl.full([BLOCK_SIZE], True, tl.int1)

    input_ = tl.load(input__ptr + offsets, mask).to(tl.float32)
    input_low = tl.load(input_low_ptr + offsets_d, mask, eviction_policy="evict_last").to(tl.float32)
    input_range = tl.load(input_range_ptr + offsets_d, mask, eviction_policy="evict_last").to(tl.float32)

    scale = (levels - 1) / input_range

    output = tl.clamp(input_, min=input_low, max=input_low + input_range)

    zero_point = libdevice.nearbyint(-input_low * scale)
    output -= input_low
    output *= scale
    output -= zero_point
    output = libdevice.nearbyint(output)
    output = output / scale

    tl.store(output_ptr + offsets, output, mask)


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
    input__ptr: torch.tensor,
    input_low_ptr: torch.tensor,
    input_range_ptr: torch.tensor,
    levels: int,
    level_low: int,
    level_high: int,
    grad_input_ptr: torch.tensor,
    grad_low_ptr: torch.tensor,
    grad_range_ptr: torch.tensor,
    diff: int,
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
    offsets_d = offsets // diff

    mask = tl.full([BLOCK_SIZE], True, tl.int1)

    grad_output = tl.load(grad_output_ptr + offsets, mask).to(tl.float32)
    input_ = tl.load(input__ptr + offsets, mask).to(tl.float32)
    input_low = tl.load(input_low_ptr + offsets_d, mask, eviction_policy="evict_last").to(tl.float32)
    input_range = tl.load(input_range_ptr + offsets_d, mask, eviction_policy="evict_last").to(tl.float32)

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

    grad_input = grad_output * mask_in

    grad_low = grad_output * (mask_hi + mask_lo)

    tl.store(grad_input_ptr + offsets, grad_input, mask)
    tl.store(grad_low_ptr + offsets_d, grad_low, mask)
    tl.store(grad_range_ptr + offsets_d, grad_range, mask)


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

    diff = input_.numel() // input_low.numel()

    with torch.cuda.device(input_.device):
        grid = lambda meta: (triton.cdiv(input_.numel(), meta["BLOCK_SIZE"]),)
        forward_kernel[grid](
            input_,
            input_low,
            input_range,
            levels,
            output,
            diff,
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

    diff = input_.numel() // input_low.numel()

    with torch.cuda.device(input_.device):
        grid = lambda meta: (
            triton.cdiv(input_.numel(), meta["BLOCK_SIZE"])
            * triton.cdiv(input_low.numel(), meta["BLOCK_SIZE"])
            * triton.cdiv(grad_range.numel(), meta["BLOCK_SIZE"]),
        )

        backward_kernel[grid](
            grad_output,
            input_,
            input_low,
            input_range,
            levels,
            level_low,
            level_high,
            grad_input,
            grad_low,
            grad_range,
            diff,
        )

    return grad_input, grad_low, grad_range
