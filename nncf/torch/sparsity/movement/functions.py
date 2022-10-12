"""
 Copyright (c) 2019 Intel Corporation
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

import torch

from nncf.torch.dynamic_graph.patch_pytorch import register_operator
from nncf.torch.functions import STThreshold

@register_operator()
def binary_mask_by_threshold(importance, threshold=0.5, sigmoid=True, max_percentile=0.98):
    with torch.no_grad():
        if sigmoid is True:
            max_threshold = torch.quantile(torch.sigmoid(importance), q=max_percentile).item()
        else:
            max_threshold = torch.quantile(importance, q=max_percentile).item()
    
    if sigmoid is True:
        return STThreshold.apply(torch.sigmoid(importance), min(threshold, max_threshold))
    return STThreshold.apply(importance, min(threshold, max_threshold))