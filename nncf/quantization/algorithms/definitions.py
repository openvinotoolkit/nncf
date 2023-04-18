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

from enum import Enum


class Granularity(Enum):
    PERTENSOR = 'pertensor'
    PERCHANNEL = 'perchannel'


class RangeType(Enum):
    MINMAX = 'min_max'
    MEAN_MINMAX = 'mean_min_max'


class OverflowFix(Enum):
    """
    This option controls whether to apply the overflow issue fix for the 8-bit quantization.

    8-bit instructions of older Intel CPU generations (based on SSE, AVX-2, and AVX-512 instruction sets)
    suffer from the so-called saturation (overflow) issue: in some configurations,
    the output does not fit into an intermediate buffer and has to be clamped.
    This can lead to an accuracy drop on the aforementioned architectures.
    The fix set to use only half a quantization range to avoid overflow for specific operations.

    If you are going to infer the quantized model on the architectures with AVX-2, and AVX-512 instruction sets,
    we recommend using FIRST_LAYER option as lower aggressive fix of the overflow issue.
    If you still face significant accuracy drop, try using ENABLE, but this may get worse the accuracy.

    :param ENABLE: All weights of all types of Convolutions and MatMul operations
        are be quantized using a half of the 8-bit quantization range.
    :param FIRST_LAYER: Weights of the first Convolutions of each model inputs
        are quantized using a half of the 8-bit quantization range.
    :param DISABLE: All weights are quantized using the full 8-bit quantization range.
    """
    ENABLE = 'enable'
    FIRST_LAYER = 'first_layer_only'
    DISABLE = 'disable'
