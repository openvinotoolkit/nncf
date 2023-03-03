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

class TargetDevice(Enum):
    """
    Describes the target device the specificity of which will be taken
    into account while compressing in order to obtain the best performance
    for this type of device.

    :param ANY:
    :param CPU:
    :param GPU:
    :param VPU:
    """

    ANY = 'ANY'
    CPU = 'CPU'
    GPU = 'GPU'
    VPU = 'VPU'
    CPU_SPR = 'CPU_SPR'


class ModelType(Enum):
    """
    Describes the model type the specificity of which will be taken into
    account during compression.

    :param TRANSFORMER: Transformer-based models
        (https://arxiv.org/pdf/1706.03762.pdf)
    """

    TRANSFORMER = 'transformer'
