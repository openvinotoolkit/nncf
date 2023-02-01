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

from nncf.common.logging import nncf_logger
from nncf.torch.utils import add_domain
from nncf.torch.binarization.extensions import BinarizedFunctionsCUDA

from torch import _C  # pylint:disable=protected-access


def _is_value(x: Any) -> bool:
    return isinstance(x, _C.Value)

# Implementation is copy-pasted from torch.onnx.symbolic_helper.
# It's need to support torch < 1.9, since there's no such function in such versions of torch.
def _is_constant(value: Any) -> bool:
    return not _is_value(value) or value.node().kind() in {
        "onnx::Constant",
        "prim::Constant",
    }


def _unsqueeze_helper(g, input_, axes_i):
    # Unsqueeze handling for different opsets inspired by torch.onnx.symbolic_helper._unsqueeze_helper
    # The original unsqueeze_helper cannot be used in 1.13 since it references
    # an `.opset` attribute on the `g` argument which is not there. The original intent
    # of this in pytorch was to allow accessing opset info in the symbolic fn's
    # code (see PR #84728 in the pytorch repo), but somehow the `symbolic` functions
    # that we define are called from the C++ code, and they pass the old torch._C.Graph
    # object as `g` instead of the torch.onnx._internal.jit_utils.GraphContext as it
    # should be.
    # The effect is that we can only do opset>=13-style unsqueeze here and hope that
    # the user did not request an older opset.
    if _is_constant(axes_i[0]):
        axes = g.op("Constant", value_t=torch.tensor(axes_i, dtype=torch.long))
        return g.op("Unsqueeze", input_, axes)
    return g.op("Unsqueeze", input_, axes_i=axes_i[0])


# pylint:disable=abstract-method
class XNORBinarizeFn(torch.autograd.Function):
    """ Binarizes x into `scale` * { +1; -1}, where +1 or -1 are chosen based
        on whether the x element value is >0 or <0. `scale` is determined as mean of absolute
        values, per input channel (0-th dimension of x). """
    @staticmethod
    def symbolic(g, x):
        zero = g.constant(0, [1], 'float')
        zero = _unsqueeze_helper(g, zero, [1, 2, 3])
        scale = g.op("Abs", x)
        scale = g.op("ReduceMean", scale, axes_i=[1, 2, 3])
        scale_neg = g.op("Neg", scale)
        return g.op(add_domain("FakeQuantize"), x, zero, zero, scale_neg, scale, levels_i=2)

    @staticmethod
    def forward(ctx, x):
        if x.is_cuda:
            output = BinarizedFunctionsCUDA.get("WeightBinarize_forward")(x, True)
        else:
            # Current CPU kernel implementations do not improve performance
            norm = x.abs().mean([1, 2, 3], keepdim=True)
            sign = ((x > 0).type(x.dtype) * 2 - 1)
            output = sign * norm
            return output
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return grad_outputs[0]


# pylint:disable=abstract-method
class DOREFABinarizeFn(torch.autograd.Function):
    """ Binarizes x into `scale` * { +1; -1}, where +1 or -1 are chosen based
        on whether the x element value is >0 or <0. `scale` is determined as mean of absolute
        values of the entire x tensor. """
    @staticmethod
    def symbolic(g, x):
        zero = g.constant(0, [1], 'float')
        zero = _unsqueeze_helper(g, zero, [1, 2, 3])
        scale = g.op("Abs", x)
        scale = g.op("ReduceMean", scale, axes_i=[0, 1, 2, 3])
        scale_neg = g.op("Neg", scale)
        return g.op(add_domain("FakeQuantize"), x, zero, zero, scale_neg, scale, levels_i=2)

    @staticmethod
    def forward(ctx, x):
        if x.is_cuda:
            output = BinarizedFunctionsCUDA.get("WeightBinarize_forward")(x, False)
        else:
            # Current CPU kernel implementations do not improve performance
            norm = x.abs().mean()
            sign = ((x > 0).type(x.dtype) * 2 - 1)
            output_flat = sign * norm
            return output_flat.view_as(x)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return grad_outputs[0]


# Activation binarization function
# pylint:disable=abstract-method
class ActivationBinarizationScaleThresholdFn(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, scale, threshold):
        zero = g.constant(0, [1], 'float')
        zero = _unsqueeze_helper(g, zero, [0, 2, 3])
        threshold = g.op("Mul", threshold, scale)
        scale = _unsqueeze_helper(g, scale, [0, 2, 3])
        return g.op(add_domain("FakeQuantize"), x, threshold, threshold, zero, scale, levels_i=2)

    @staticmethod
    def forward(ctx, input_, scale, threshold):
        if input_.is_cuda:
            output = BinarizedFunctionsCUDA.get("ActivationBinarize_forward")(input_, scale, threshold)
        else:
            # Current CPU kernel implementations do not improve performance
            shape = [1 for s in input_.shape]
            shape[1] = input_.shape[1]
            t = (threshold * scale).view(shape)
            output = (input_ > t).type(input_.dtype) * scale
            ctx.save_for_backward(input_, scale, output)
        ctx.save_for_backward(input_, scale, output)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]
        if grad_output.is_cuda:
            if not grad_output.is_contiguous():
                nncf_logger.debug("grad_output is not contiguous!")
                grad_output = grad_output.contiguous()

        input_, scale, output = ctx.saved_variables

        if input_.is_cuda:
            grad_input, grad_scale, grad_threshold = BinarizedFunctionsCUDA.get("ActivationBinarize_backward")(
                grad_output,
                input_,
                scale, output)
        else:
            # Current CPU kernel implementations do not improve performance
            # grad_input, grad_scale, grad_threshold = BinarizedFunctionsCPU.ActivationBinarize_backward(grad_output,
            #                                                                                           input_,
            #                                                                                           scale, output)
            # calc gradient for input
            mask_lower = (input_ <= scale).type(input_.dtype)
            grad_input = grad_output * (input_ >= 0).type(input_.dtype) * mask_lower

            # calc gradient for scale
            err = (output - input_) * scale.reciprocal()
            grad_scale = grad_output * (mask_lower * err + (1 - mask_lower))
            grad_scale = grad_scale.sum().view(1)

            # calc gradient for threshold
            grad_threshold = -grad_output * (input_ > 0).type(input_.dtype) * (input_ < scale).type(input_.dtype)

            for idx, _ in enumerate(input_.shape):
                if idx != 1:  # sum over all dims except activations channel
                    grad_threshold = grad_threshold.sum(idx, keepdim=True)

        return grad_input, grad_scale, grad_threshold
