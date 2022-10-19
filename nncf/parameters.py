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
from typing import List


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


class IgnoredScope:
    """
    Dataclass that contains description of the ignored scope
    """

    def __init__(self,
                 node_names: List[str] = None,
                 node_name_regexps: List[str] = None,
                 node_types: List[str] = None):
        """
        :param node_names: list of ignored node names
        :param node_name_regexps: list of regular expressions applied
            to node names
        :param node_types: list of ignored node types
        """
        self.node_names = node_names
        self.node_name_regexps = node_name_regexps
        self.node_types = node_types
