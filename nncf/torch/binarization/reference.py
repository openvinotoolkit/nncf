"""
 Copyright (c) 2022 Intel Corporation
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
from enum import Enum
from typing import TypeVar

import numpy as np
import torch

GeneralizedTensor = TypeVar('GeneralizedTensor', torch.Tensor, np.ndarray)


class ReferenceBackendType(Enum):
    NUMPY = 'numpy'
    TORCH = 'torch'


class ReferenceBase:
    def __init__(self, backend_type: ReferenceBackendType):
        if backend_type is ReferenceBackendType.NUMPY:
            self.backend = np
        elif backend_type is ReferenceBackendType.TORCH:
            self.backend = torch
        else:
            raise RuntimeError("Unknown backend for ReferenceQuantize")


class ReferenceXNORBinarize(ReferenceBase):
    def forward(self, x: GeneralizedTensor) -> GeneralizedTensor:
        norm = self.backend.abs(x).mean((1, 2, 3), keepdims=True)
        sign = ((x > 0).astype(x.dtype) * 2 - 1)
        output = sign * norm
        return output

    @staticmethod
    def backward(grad_output: GeneralizedTensor) -> GeneralizedTensor:
        return grad_output


class ReferenceDOREFABinarize(ReferenceBase):
    def forward(self, x: GeneralizedTensor) -> GeneralizedTensor:
        norm = self.backend.abs(x).mean()
        sign = ((x > 0).astype(x.dtype) * 2 - 1)
        return sign * norm

    @staticmethod
    def backward(grad_output: GeneralizedTensor) -> GeneralizedTensor:
        return grad_output


class ReferenceActivationBinarize(ReferenceBase):
    @staticmethod
    def forward(x: GeneralizedTensor, scale: GeneralizedTensor, threshold: GeneralizedTensor) -> GeneralizedTensor:
        shape = [1 for s in x.shape]
        shape[1] = x.shape[1]
        t = threshold * scale
        output = (x > t).astype(x.dtype) * scale
        return output

    @staticmethod
    def backward(grad_output, x, scale, output):

        # calc gradient for input
        mask_lower = (x <= scale).astype(x.dtype)
        grad_input = grad_output * (x >= 0).astype(x.dtype) * mask_lower

        # calc gradient for scale
        err = (output - x) / scale
        grad_scale = grad_output * (mask_lower * err + (1 - mask_lower))
        grad_scale = grad_scale.sum()

        # calc gradient for threshold
        grad_threshold = -grad_output * (x > 0).astype(x.dtype) * (x < scale).astype(x.dtype)

        for idx, _ in enumerate(x.shape):
            if idx != 1:  # activation channel dimension
                grad_threshold = grad_threshold.sum(idx, keepdims=True)

        return [grad_input, grad_scale, grad_threshold]


class ReferenceBinarizedFunctions:
    _xnor = ReferenceXNORBinarize(backend_type=ReferenceBackendType.TORCH)
    _dorefa = ReferenceDOREFABinarize(backend_type=ReferenceBackendType.TORCH)
    _act = ReferenceActivationBinarize(backend_type=ReferenceBackendType.TORCH)

    @staticmethod
    def WeightBinarize_forward(x: torch.Tensor, is_xnor: bool) -> torch.Tensor:
        if is_xnor:
            return ReferenceBinarizedFunctions._xnor.forward(x)
        return ReferenceBinarizedFunctions._dorefa.forward(x)

    ActivationBinarize_forward = _act.forward
