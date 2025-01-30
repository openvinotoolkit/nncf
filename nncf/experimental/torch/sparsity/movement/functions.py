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

from nncf.torch.dynamic_graph.patch_pytorch import register_operator
from nncf.torch.functions import STThreshold


@register_operator()
def binary_mask_by_threshold(
    input_tensor: torch.Tensor, threshold: float = 0.5, max_percentile: float = 0.98
) -> torch.Tensor:
    """
    Conduct straight-through thresholding function while limiting the maximum threshold.

    :param input_tensor: The tensor to conduct thresholding.
    :param threshold: The criterion for thresholding.
    :param max_percentile: Specifies the `q`-th quantiles of the input tensor.
    :return: The mask for the input tensor with the threshold limited by `max_percentile`.
    """
    with torch.no_grad():
        max_threshold = torch.quantile(input_tensor, q=max_percentile).item()
    return STThreshold.apply(input_tensor, min(threshold, max_threshold))
