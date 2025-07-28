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

from dataclasses import dataclass

import torch


@dataclass
class TorchQDQParameters:
    """
    Stores the quantization parameters required for
    creation of a PyTorch quantize-dequantize pair.

    :param quant_min: Minimum quant value.
    :type quant_min: int
    :param quant_max: Maximum quant value.
    :type quant_max: int
    :param scale: Defines the scale factor used for quantization.
    :type scale: torch.Tensor
    :param zero_point: Specifies the quantized value to which 0 in floating point maps to.
    :type zero_point: torch.Tensor
    :param is_per_channel: Whether quantization is applied per channel.
    :type is_per_channel: bool
    :param ch_axis: Channel axis used for per-channel quantization.
    :type ch_axis: int
    """

    quant_min: int
    quant_max: int
    scale: torch.Tensor
    zero_point: torch.Tensor
    is_per_channel: bool
    ch_axis: int
