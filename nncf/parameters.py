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
from typing import List, Optional


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
    Dataclass that contains description of the ignored scope.

    The ignored scope defines model sub-graphs that should be excluded from
    the optimization process such as quantization, pruning and etc.

    Examples:

    ```
    import nncf

    # Exclude by node name:
    node_names = ['node_1', 'node_2', 'node_3']
    ignored_scope = nncf.IgnoredScope(node_names=node_names)

    # Exclude using regular expressions:
    pattern = ['node_\\d']
    ignored_scope = nncf.IgnoredScope(node_name_patterns=pattern)

    # Exclude by node type:

    # OpenVINO opset https://docs.openvino.ai/latest/openvino_docs_ops_opset.html
    node_types = ['Multiply', 'GroupConvolution', 'Interpolate']
    ignored_scope = nncf.IgnoredScope(node_types=pattern)

    # ONNX opset https://github.com/onnx/onnx/blob/main/docs/Operators.md
    node_types = ['Mul', 'Conv', 'Resize']
    ignored_scope = nncf.IgnoredScope(node_types=pattern)

    ...

    ```

    **Note** Node types must be specified according to the model framework.
    """

    def __init__(self,
                 node_names: Optional[List[str]] = None,
                 node_name_patterns: Optional[List[str]] = None,
                 node_types: Optional[List[str]] = None):
        """
        :param node_names: List of ignored node names
        :param node_name_patterns: List of regular expressions specifying
            patterns for names of ignored nodes
        :param node_types: List of ignored node types
        """
        self.node_names = node_names
        self.node_name_patterns = node_name_patterns
        self.node_types = node_types
