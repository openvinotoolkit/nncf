"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from typing import Any

import torch

from nncf.torch.utils import add_domain
from nncf.common.logging import nncf_logger

from nncf.torch.quantization.extensions import QuantizedFunctionsCPU, QuantizedFunctionsCUDA
from nncf.torch.dynamic_graph.patch_pytorch import register_operator
from nncf.torch.functions import STRound, clamp


# pylint:disable=abstract-method
class QuantizeSymmetric(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, scale, level_low, level_high, levels):
        input_low = scale * (level_low / level_high)
        input_range = scale - input_low

        if input_.is_cuda:
            if not input_.is_contiguous():
                nncf_logger.debug("input_ is not contiguous!")
                input_ = input_.contiguous()

            # Required to support both torch.amp.autocast and models that perform explicit type casting
            # inside their forward calls.
            if input_.dtype == torch.float16:
                input_low = input_low.type(torch.float16)
                input_range = input_range.type(torch.float16)
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


# pylint:disable=abstract-method
class QuantizeAsymmetric(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, input_low, input_range, level_low, level_high, levels):
        if input_.is_cuda:
            if not input_.is_contiguous():
                nncf_logger.debug("input_ is not contiguous!")
                input_ = input_.contiguous()

            # Required to support both torch.amp.autocast and models that perform explicit type casting
            # inside their forward calls.
            if input_.dtype == torch.float16:
                input_low = input_low.type(torch.float16)
                input_range = input_range.type(torch.float16)
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


def _quantize_autograd_to_range(input_, input_low, input_high, levels):
    input_ = input_ - input_low
    input_range = (input_high - input_low)
    scale = (levels - 1) / input_range
    output = clamp(input_, low=torch.zeros_like(input_), high=input_range)
    output = output * scale
    output = STRound.apply(output)
    output = output * input_range / (levels - 1) + input_low
    return output


# pylint:disable=abstract-method
class ExportQuantizeToFakeQuantize(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input_, levels, input_low, input_high, output_low, output_high):
        output = g.op(
            add_domain("FakeQuantize"), input_, input_low, input_high, output_low, output_high, levels_i=levels
        )
        # setType is needed for proper shape inference of custom op on ONNX export. Should work for torch >= 1.14
        output.setType(input_.type())
        return output

    @staticmethod
    def forward(ctx, input_, levels, input_low, input_high, output_low, output_high):
        return torch.clone(input_)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        # backward is not used during export
        return grad_outputs[0]


# pylint:disable=abstract-method
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
    levels = level_high - level_low + 1
    assert levels in [255, 256], "Can only export to INT8 256-level ONNX Quantize/Dequantize pairs"

    y_scale = (input_high - input_low) / (level_high - level_low)
    y_zero_point = (level_low * input_high - level_high * input_low) / (input_high - input_low)

    type_ = torch.int8 if level_low < 0 else torch.uint8
    level_low *= torch.ones_like(y_zero_point).to(type_)
    level_high *= torch.ones_like(y_zero_point).to(type_)
    level_low = level_low.to(y_zero_point.device)
    level_high = level_high.to(y_zero_point.device)
    y_zero_point = torch.min(torch.max(level_low, y_zero_point.to(type_)), level_high)

    y_scale = torch.squeeze(y_scale)
    y_zero_point = torch.squeeze(y_zero_point)
    return y_scale, y_zero_point


@register_operator()
def symmetric_quantize(input_, levels, level_low, level_high, scale, eps, skip: bool = False):
    if skip:
        return input_
    scale = scale.to(dtype=input_.dtype)
    scale_safe = abs(scale) + eps
    return QuantizeSymmetric.apply(input_, scale_safe, level_low, level_high, levels)


@register_operator()
def asymmetric_quantize(input_, levels, level_low, level_high, input_low, input_range, eps, skip: bool = False):
    if skip:
        return input_
    input_range_safe = abs(input_range) + eps
    input_low_tuned, input_range_tuned = TuneRange.apply(input_low, input_range_safe, levels)
    return QuantizeAsymmetric.apply(input_, input_low_tuned, input_range_tuned, level_low, level_high, levels)


# pylint:disable=abstract-method
class TuneRange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_low, input_range, levels):
        input_high = input_range + input_low
        input_low_copy = input_low.clone()
        input_low_copy[input_low_copy > 0] = 0
        input_high[input_high < 0] = 0
        n = levels - 1
        # Need a cast here because fp16 division yileds fp32 results sometimes
        scale = (levels / (input_high - input_low_copy)).to(dtype=input_high.dtype)
        zp = torch.round(-input_low_copy * scale)

        new_input_low = torch.where(zp < n, zp / (zp - n) * input_high, input_low_copy)
        new_input_high = torch.where(zp > 0., (zp - n) / zp * input_low_copy, input_high)

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
