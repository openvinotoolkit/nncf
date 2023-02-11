from enum import Enum
from typing import List
from typing import Tuple
from typing import TypeVar

import numpy as np
import torch

from nncf.torch.utils import sum_like

GeneralizedTensor = TypeVar('GeneralizedTensor', torch.Tensor, np.ndarray)


class ReferenceBackendType(Enum):
    NUMPY = 'numpy'
    TORCH = 'torch'


class ReferenceQuantize:
    def __init__(self, backend_type: ReferenceBackendType):
        if backend_type is ReferenceBackendType.NUMPY:
            self.backend = np
        elif backend_type is ReferenceBackendType.TORCH:
            self.backend = torch
        else:
            raise RuntimeError("Unknown backend for ReferenceQuantize")

    def forward(self,
                input_: GeneralizedTensor,
                input_low: GeneralizedTensor,
                input_range: GeneralizedTensor,
                levels: int) -> GeneralizedTensor:
        scale = (levels - 1) / input_range
        output = input_.clip(min=input_low, max=input_low + input_range)
        zero_point = (- input_low * scale).round()
        output -= input_low
        output *= scale
        output -= zero_point
        output = output.round()
        output = output / scale
        return output

    def backward(self,
                 grad_output: GeneralizedTensor,
                 input_: GeneralizedTensor,
                 input_low: GeneralizedTensor,
                 input_range: GeneralizedTensor,
                 output: GeneralizedTensor,
                 level_low: int,
                 level_high: int,
                 range_sign: int) -> List[GeneralizedTensor]:
        mask_hi = (input_ > (input_low + input_range)).astype(input_.dtype)
        mask_lo = (input_ < input_low).astype(input_.dtype)

        mask_in = 1 - mask_hi - mask_lo
        err = (output - input_) * np.reciprocal(input_range * range_sign)
        grad_range = grad_output * (err * mask_in + range_sign * (level_low / level_high) * mask_lo + mask_hi)
        grad_range = sum_like(grad_range, input_range)

        grad_input = grad_output * mask_in

        grad_low = grad_output * (mask_hi + mask_lo)
        grad_low = sum_like(grad_low, input_low)
        return [grad_input, grad_low, grad_range]

    def tune_range(self, input_low: GeneralizedTensor, input_range: GeneralizedTensor, levels: int) \
            -> Tuple[GeneralizedTensor, GeneralizedTensor]:
        input_high = input_range + input_low
        input_low[input_low > 0] = 0
        input_high[input_high < 0] = 0
        n = levels - 1
        scale = levels / (input_high - input_low)
        scale = scale.astype(dtype=input_high.dtype)
        zp = self.backend.round(-input_low * scale)

        new_input_low = self.backend.where(zp < n, zp / (zp - n) * input_high, input_low)
        new_input_high = self.backend.where(zp > 0., (zp - n) / zp * input_low, input_high)

        range_1 = input_high - new_input_low
        range_2 = new_input_high - input_low

        mask = (range_1 > range_2).astype(input_high.dtype)
        inv_mask = abs(1 - mask)

        new_input_low = mask * new_input_low + inv_mask * input_low
        new_input_range = inv_mask * new_input_high + mask * input_high - new_input_low

        return new_input_low, new_input_range


class ReferenceQuantizedFunctions:
    _executor = ReferenceQuantize(backend_type=ReferenceBackendType.TORCH)
    Quantize_forward = _executor.forward
    Quantize_backward = _executor.backward
