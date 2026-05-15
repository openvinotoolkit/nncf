# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any

import torch
from torch.overrides import handle_torch_function
from torch.overrides import has_torch_function_unary

from nncf.common.logging import nncf_logger
from nncf.errors import ValidationError
from nncf.torch.quantization.extensions import QuantizedFunctionsCPU
from nncf.torch.quantization.extensions import QuantizedFunctionsCUDA
from nncf.torch.quantization.reference import ReferenceQuantizedFunctions as RQ
from nncf.torch.utils import add_ov_domain


class QuantizeSymmetric(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, scale, level_low, level_high, levels):
        # Required to support both torch.amp.autocast and models that perform explicit type casting
        # inside their forward calls.
        if input_.dtype in [torch.bfloat16, torch.float16]:
            scale = scale.type(input_.dtype)

        input_low = scale * (level_low / level_high)
        input_range = scale - input_low

        if input_.is_cuda:
            if not input_.is_contiguous():
                nncf_logger.debug("input_ is not contiguous!")
                input_ = input_.contiguous()
            output = QuantizedFunctionsCUDA.get("Quantize_forward")(input_, input_low, input_range, levels)
        else:
            output = QuantizedFunctionsCPU.get("Quantize_forward")(input_, input_low, input_range, levels)

        ctx.save_for_backward(input_, input_low, input_range)
        ctx.levels = levels
        ctx.level_low = level_low
        ctx.level_high = level_high

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]
        input_, input_low, input_range = ctx.saved_tensors
        levels = ctx.levels
        level_low = ctx.level_low
        level_high = ctx.level_high

        if grad_output.is_cuda:
            if not grad_output.is_contiguous():
                nncf_logger.debug("grad_output is not contiguous!")
                grad_output = grad_output.contiguous()

            grad_input, _, grad_scale = QuantizedFunctionsCUDA.get("Quantize_backward")(
                grad_output, input_, input_low, input_range, levels, level_low, level_high
            )
        else:
            grad_input, _, grad_scale = QuantizedFunctionsCPU.get("Quantize_backward")(
                grad_output, input_, input_low, input_range, levels, level_low, level_high, False
            )

        return grad_input, grad_scale, None, None, None


class QuantizeAsymmetric(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, input_low, input_range, level_low, level_high, levels):
        # Required to support both torch.amp.autocast and models that perform explicit type casting
        # inside their forward calls.
        if input_.dtype in [torch.bfloat16, torch.float16]:
            input_low = input_low.type(input_.dtype)
            input_range = input_range.type(input_.dtype)

        if input_.is_cuda:
            if not input_.is_contiguous():
                nncf_logger.debug("input_ is not contiguous!")
                input_ = input_.contiguous()
            output = QuantizedFunctionsCUDA.get("Quantize_forward")(input_, input_low, input_range, levels)
        else:
            output = QuantizedFunctionsCPU.get("Quantize_forward")(input_, input_low, input_range, levels)

        ctx.save_for_backward(input_, input_low, input_range)
        ctx.levels = levels
        ctx.level_low = level_low
        ctx.level_high = level_high
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]
        input_, input_low, input_range = ctx.saved_tensors
        levels = ctx.levels
        level_low = ctx.level_low
        level_high = ctx.level_high

        if grad_output.is_cuda:
            if not grad_output.is_contiguous():
                nncf_logger.debug("grad_output is not contiguous!")
                grad_output = grad_output.contiguous()

            grad_input, grad_input_low, grad_input_range = QuantizedFunctionsCUDA.get("Quantize_backward")(
                grad_output, input_, input_low, input_range, levels, level_low, level_high
            )
        else:
            grad_input, grad_input_low, grad_input_range = QuantizedFunctionsCPU.get("Quantize_backward")(
                grad_output, input_, input_low, input_range, levels, level_low, level_high, True
            )
        return grad_input, grad_input_low, grad_input_range, None, None, None


class QuantizeSymmetricTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, input_shape, scale, level_low, level_high, levels):
        # range: [-scale, 7/8 * scale] if scale > 0 else [7/8 * scale, -scale]
        input_low = torch.where(scale > 0, -scale, -scale / level_low * level_high)
        # 15/8 * scale or (2-1/8) * scale
        input_range = torch.abs((2 + 1 / level_low) * scale)
        dtype = input_.dtype
        original_shape = input_.shape
        input_ = input_.reshape(input_shape)

        output = RQ.Quantize_forward(input_.type(torch.float32), input_low, input_range, levels)

        ctx.save_for_backward(input_, input_low, input_range)
        ctx.level_low = level_low
        ctx.level_high = level_high
        ctx.levels = levels

        output = output.reshape(original_shape)
        return output.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input_, input_low, input_range = ctx.saved_tensors
        levels = ctx.levels
        level_low = ctx.level_low
        level_high = ctx.level_high

        input_shape = input_.shape
        orig_shape = grad_output.shape
        grad_output = grad_output.reshape(input_shape)

        grad_input, _, grad_scale = RQ.Quantize_backward(
            grad_output, input_, input_low, input_range, levels, level_low, level_high
        )

        grad_input = grad_input.reshape(orig_shape)
        grad_scale = grad_scale.float()
        # input, input_shape, scale, level_low, level_high, levels
        return grad_input, None, grad_scale, None, None, None


class QuantizeAsymmetricTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, input_shape, input_low, input_range, level_low, level_high, levels):
        dtype = input_.dtype
        original_shape = input_.shape
        input_ = input_.reshape(input_shape)

        output = RQ.Quantize_forward(input_.type(torch.float32), input_low, input_range, levels)

        # Save tensors for backward pass
        ctx.save_for_backward(input_, input_low, input_range)
        ctx.level_low = level_low
        ctx.level_high = level_high
        ctx.levels = levels

        output = output.reshape(original_shape)
        return output.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input_, input_low, input_range = ctx.saved_tensors
        levels = ctx.levels
        level_low = ctx.level_low
        level_high = ctx.level_high
        input_shape = input_.shape
        orig_shape = grad_output.shape
        grad_output = grad_output.reshape(input_shape)

        grad_input, grad_low, grad_range = RQ.Quantize_backward(
            grad_output, input_, input_low, input_range, levels, level_low, level_high
        )

        grad_input = grad_input.reshape(orig_shape)
        grad_low = grad_low.float()
        grad_range = grad_range.float()
        # input, input_size, input_low, input_range, level_low, level_high, levels
        return grad_input, None, grad_low, grad_range, None, None, None


class ExportQuantizeToFakeQuantize(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g, input_, levels, input_low, input_high, output_low, output_high, scale, zero_point, q_min, q_max, ch_axis
    ):
        output = g.op(
            add_ov_domain("FakeQuantize"), input_, input_low, input_high, output_low, output_high, levels_i=levels
        )
        # setType is needed for proper shape inference of custom op on ONNX export. Should work for torch >= 1.14
        output.setType(input_.type())
        return output

    @staticmethod
    def forward(
        ctx, input_, levels, input_low, input_high, output_low, output_high, scale, zero_point, q_min, q_max, ch_axis
    ):
        if ch_axis is not None:
            return torch.fake_quantize_per_channel_affine(input_, scale, zero_point, ch_axis, q_min, q_max)
        return torch.fake_quantize_per_tensor_affine(input_, scale, zero_point, q_min, q_max)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        # backward is not used during export
        return grad_outputs[0]


class ExportQuantizeToONNXQuantDequant(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input_, y_scale, y_zero_point, axis):
        quantized = g.op("QuantizeLinear", input_, y_scale, y_zero_point, axis_i=axis)
        dequantized = g.op("DequantizeLinear", quantized, y_scale, y_zero_point, axis_i=axis)
        return dequantized

    @staticmethod
    def forward(ctx, input_, y_scale, y_zero_point, axis):
        return torch.clone(input_)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        # backward is not used during export
        return grad_outputs[0]


def get_scale_zp_from_input_low_input_high(level_low, level_high, input_low, input_high):
    y_scale = (input_high - input_low) / (level_high - level_low)
    y_zero_point = (level_low * input_high - level_high * input_low) / (input_high - input_low)

    type_ = torch.int8 if level_low < 0 else torch.uint8
    level_low *= torch.ones_like(y_zero_point).to(type_)
    level_high *= torch.ones_like(y_zero_point).to(type_)
    level_low = level_low.to(y_zero_point.device)
    level_high = level_high.to(y_zero_point.device)
    y_zero_point = torch.min(torch.max(level_low, torch.round(y_zero_point).to(type_)), level_high)

    y_scale = torch.squeeze(y_scale)
    y_zero_point = torch.squeeze(y_zero_point)
    return y_scale, y_zero_point


def symmetric_quantize(input_, levels, level_low, level_high, scale, eps, skip: bool = False):
    if has_torch_function_unary(input_):
        return handle_torch_function(
            symmetric_quantize, (input_,), input_, levels, level_low, level_high, scale, eps, skip
        )
    if skip:
        return input_
    scale_safe = abs(scale) + eps
    return QuantizeSymmetric.apply(input_, scale_safe, level_low, level_high, levels)


def asymmetric_quantize(input_, levels, level_low, level_high, input_low, input_range, eps, skip: bool = False):
    if has_torch_function_unary(input_):
        return handle_torch_function(
            asymmetric_quantize, (input_,), input_, levels, level_low, level_high, input_low, input_range, eps, skip
        )
    if skip:
        return input_
    input_range_safe = abs(input_range) + eps
    input_low_tuned, input_range_tuned = TuneRange.apply(input_low, input_range_safe, levels)
    return QuantizeAsymmetric.apply(input_, input_low_tuned, input_range_tuned, level_low, level_high, levels)


def asymmetric_quantize_lora(
    input_, input_shape, A, B, input_low_, input_range_, level_low, level_high, levels, eps, skip: bool = False
):
    if has_torch_function_unary(input_):
        return handle_torch_function(
            asymmetric_quantize_lora,
            (input_,),
            input_,
            input_shape,
            A,
            B,
            input_low_,
            input_range_,
            level_low,
            level_high,
            levels,
            eps,
            skip,
        )
    if skip:
        return input_
    input_range_safe = abs(input_range_) + eps
    input_low, input_range = TuneRange.apply(input_low_, input_range_safe, levels)
    input_ = (input_ + B @ A).type(input_.dtype)  # input(float16) + lora(bfloat16) = float32, need a cast to float16
    return QuantizeAsymmetricTorch.apply(
        input_,
        input_shape,
        input_low,
        input_range,
        level_low,
        level_high,
        levels,
    )


def symmetric_quantize_lora(input_, input_shape, A, B, scale, level_low, level_high, levels, eps, skip: bool = False):
    if has_torch_function_unary(input_):
        return handle_torch_function(
            symmetric_quantize_lora,
            (input_,),
            input_,
            input_shape,
            A,
            B,
            scale,
            level_low,
            level_high,
            levels,
            eps,
            skip,
        )
    if skip:
        return input_
    scale_safe = torch.where(torch.abs(scale) < eps, eps, scale)
    input_ = (input_ + B @ A).type(input_.dtype)  # input(float16) + lora(bfloat16) = float32, need a cast to float16
    return QuantizeSymmetricTorch.apply(
        input_,
        input_shape,
        scale_safe,
        level_low,
        level_high,
        levels,
    )


class TuneRange(torch.autograd.Function):
    """
    Makes sure that the zero-point quantum in the quantized domain points exactly to floating point zero,
    e.g. that the input floating point zeroes to the fake quantization operation are translated to output
    floating point zeroes even if we don't use rounding.
    See [docs](../../../docs/compression_algorithms/Quantization.md#asymmetric-quantization) for details.
    """

    @staticmethod
    def forward(ctx, input_low, input_range, levels):
        input_high = input_range + input_low
        input_low_copy = input_low.clone()
        input_low_copy[input_low_copy > 0] = 0
        input_high[input_high < 0] = 0
        n = levels - 1
        # Need a cast here because fp16 division yields fp32 results sometimes
        scale = (n / (input_high - input_low_copy)).to(dtype=input_high.dtype)
        zp = torch.round(-input_low_copy * scale)

        new_input_low = torch.where(zp < n, zp / (zp - n) * input_high, input_low_copy)
        new_input_high = torch.where(zp > 0.0, (zp - n) / zp * input_low_copy, input_high)

        range_1 = input_high - new_input_low
        range_2 = new_input_high - input_low_copy

        mask = (range_1 > range_2).to(input_high.dtype)
        inv_mask = (1 - mask).abs()

        new_input_low = mask * new_input_low + inv_mask * input_low_copy
        new_input_range = inv_mask * new_input_high + mask * input_high - new_input_low

        return new_input_low, new_input_range

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_input_low = grad_outputs[0]
        grad_input_range = grad_outputs[1]
        return grad_input_low, grad_input_range, None


def decompress_asymmetric(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """
    Decompress the asymmetrically quantized input tensor.

    :param input: An input tensor
    :param scale: A scale tensor
    :param zero_point: A zero point tensor
    :return: The decompressed tensor
    """
    input = input.type(dtype=scale.dtype)
    zero_point = zero_point.type(dtype=scale.dtype)
    decompressed_input = (input - zero_point) * scale
    return decompressed_input


def decompress_symmetric(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Decompress the symmetrically quantized input tensor.

    :param input: An input tensor
    :param scale: A scale tensor
    :return: The decompressed tensor
    """
    input = input.type(dtype=scale.dtype)
    decompressed_input = input * scale
    return decompressed_input


def pack_uint4(tensor: torch.Tensor) -> torch.Tensor:
    """
    Packs a tensor containing uint4 values (in the range [0, 15]) into a tensor with uint8 values,
    where each element stores two uint4 values.

    :param tensor: A tensor of dtype `torch.uint8` where each element represents a uint4 value.
        The tensor should contain values in the range [0, 15].
    :return: A packed tensor of dtype `torch.uint8` where each element packs two uint4 values.
    :raises nncf.errors.ValidationError: If the input tensor is not of type `torch.uint8`.
    """
    if tensor.dtype != torch.uint8:
        msg = f"Invalid tensor dtype {tensor.type}. torch.uint8 type is supported."
        raise ValidationError(msg)
    packed_tensor = tensor.contiguous()
    packed_tensor = packed_tensor.reshape(-1, 2)
    packed_tensor = torch.bitwise_and(packed_tensor[..., ::2], 15) | packed_tensor[..., 1::2] << 4
    return packed_tensor


def unpack_uint4(packed_tensor: torch.Tensor) -> torch.Tensor:
    """
    Unpacks a tensor, where each uint8 element stores two uint4 values, back into a tensor with
    individual uint4 values.

    :param packed_tensor: A tensor of dtype `torch.uint8` where each element packs two uint4 values.
    :return: A tensor of dtype `torch.uint8` where each element represents a uint4 value.
    """
    return torch.stack((torch.bitwise_and(packed_tensor, 15), torch.bitwise_right_shift(packed_tensor, 4)), dim=-1)


def pack_int4(tensor: torch.Tensor) -> torch.Tensor:
    """
    Packs a tensor containing int4 values (in the range [-8, 7]) into a tensor with uint8 values,
    where each element stores two int4 values.

    :param tensor: A tensor of dtype `torch.int8` where each element represents an int4 value.
        The tensor should contain values in the range [-8, 7].
    :return: A packed tensor of dtype `torch.uint8` where each element packs two int4 values.
    :raises nncf.errors.ValidationError: If the input tensor is not of type `torch.int8`.
    """
    if tensor.dtype != torch.int8:
        msg = f"Invalid tensor dtype {tensor.type}. torch.int8 type is supported."
        raise ValidationError(msg)
    tensor = tensor + 8
    return pack_uint4(tensor.type(torch.uint8))


def unpack_int4(packed_tensor: torch.Tensor) -> torch.Tensor:
    """
    Unpacks a tensor, where each uint8 element stores two int4 values, back into a tensor with
    individual int4 values.

    :param packed_tensor: A tensor of dtype `torch.uint8` where each element packs two int4 values.
    :return: A tensor of dtype `torch.int8` where each element represents an int4 value.
    """
    t = unpack_uint4(packed_tensor)
    return t.type(torch.int8) - 8
