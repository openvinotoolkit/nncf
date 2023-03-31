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
    This option controls whether to apply the overflow issue fix.
    If set to `disable`, there is no effect. 
    If set to `enable`, the fix will be applied to all weight quantizers. 
    If set to `first_layer_only` the fix will be applied to the first weight quantizers.
    
    The fix itself pushes weights FakeQuantizers effectively use only a half quantization range.
    """
    ENABLE = 'enable'
    FIRST_LAYER = 'first_layer_only'
    DISABLE = 'disable'
