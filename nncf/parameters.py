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

import re
from enum import Enum
from typing import List, Optional

from nncf.common.logging import nncf_logger
from nncf.common.graph.graph import NNCFGraph

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


class IgnoredScope:
    """
    Dataclass that contains description of the ignored scope.

    The ignored scope defines model sub-graphs that should be excluded from
    the compression process such as quantization, pruning and etc.

    Examples:

    ```
    import nncf

    # Exclude by node name:
    node_names = ['node_1', 'node_2', 'node_3']
    ignored_scope = nncf.IgnoredScope(names=node_names)

    # Exclude using regular expressions:
    patterns = ['node_\\d']
    ignored_scope = nncf.IgnoredScope(patterns=patterns)

    # Exclude by operation type:

    # OpenVINO opset https://docs.openvino.ai/latest/openvino_docs_ops_opset.html
    operation_types = ['Multiply', 'GroupConvolution', 'Interpolate']
    ignored_scope = nncf.IgnoredScope(types=operation_types)

    # ONNX opset https://github.com/onnx/onnx/blob/main/docs/Operators.md
    operation_types = ['Mul', 'Conv', 'Resize']
    ignored_scope = nncf.IgnoredScope(types=operation_types)

    ...

    ```

    **Note** Operation types must be specified according to the model framework.
    """

    def __init__(self,
                 names: Optional[List[str]] = None,
                 patterns: Optional[List[str]] = None,
                 types: Optional[List[str]] = None):
        """
        :param names: List of ignored node names
        :param patterns: List of regular expressions that define patterns for
            names of ignored nodes
        :param types: List of ignored operation types
        """
        self.names = names
        self.patterns = patterns
        self.types = types


def convert_ignored_scope_to_list(ignored_scope: Optional[IgnoredScope]) -> List[str]:
    """
    Convert the contents of the `IgnoredScope` class to the legacy ignored
    scope format.

    :param ignored_scope: The ignored scope
    :return: An ignored scope in the legacy format as list
    """
    results = []
    if ignored_scope is None:
        return results
    if ignored_scope.names is not None:
        results.extend(ignored_scope.names)
    if ignored_scope.patterns is not None:
        for p in ignored_scope.patterns:
            results.append('{re}' + p)
    if ignored_scope.types is not None:
        raise RuntimeError('Legacy ignored scope format does not support '
                           'operation types')
    return results


def get_ignored_node_names_from_ignored_scope(ignored_scope: IgnoredScope,
                                              nncf_graph: NNCFGraph) -> List[str]:
    """
    Returns list of ignored names according to ignored scope and NNCFGraph.

    :param ignored_scope: Given ignored scope instance.
    :param nncf_grpah: Given NNCFGrpah.
    :returns: List of NNCF node names from given NNCFGraph specified in given ignored scope.
    """
    node_names = [node.node_name for node in nncf_graph.get_all_nodes()]
    matched_by_names = []
    if ignored_scope.names:
        for ignored_node_name in ignored_scope.names:
            if ignored_node_name in node_names:
                matched_by_names.append(ignored_node_name)
        nncf_logger.info(f'{len(matched_by_names)} out of {len(ignored_scope.names)}'
                          ' ignored nodes was found by name in the NNCFGraph')

    matched_by_patterns = []
    if ignored_scope.patterns:
        for str_pattern in ignored_scope.patterns:
            pattern = re.compile(str_pattern)
            matches = list(filter(pattern.match, node_names))
            matched_by_patterns.extend(matches)
        nncf_logger.info(f'{matched_by_patterns}'
                          ' ignored nodes was found by patterns in the NNCFGraph')

    matched_by_types = []
    if ignored_scope.types:
        for node in nncf_graph.get_all_nodes():
            if node.node_type in ignored_scope.types:
                matched_by_types.append(node.node_name)
        nncf_logger.info(f'{matched_by_types}'
                          ' ignored nodes was found by types in the NNCFGraph')

    results = list(set(matched_by_names + matched_by_types + matched_by_patterns))
    nncf_logger.info(f'{len(results)} ignored nodes was found in total')
    return results
