# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from dataclasses import dataclass
from dataclasses import field
from typing import List, Optional

from nncf.common.graph.graph import NNCFGraph
from nncf.common.logging import nncf_logger
from nncf.common.utils.api_marker import api


@api(canonical_alias="nncf.IgnoredScope")
@dataclass
class IgnoredScope:
    """
    Provides an option to specify portions of model to be excluded from compression.

    The ignored scope defines model sub-graphs that should be excluded from the compression process such as
    quantization, pruning and etc.

    Example:

    ..  code-block:: python

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

    **Note:** Operation types must be specified according to the model framework.

    :param names: List of ignored node names.
    :type names: List[str]
    :param patterns: List of regular expressions that define patterns for names of ignored nodes.
    :type patterns: List[str]
    :param types: List of ignored operation types.
    :type types: List[str]
    """

    names: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    types: List[str] = field(default_factory=list)


def convert_ignored_scope_to_list(ignored_scope: Optional[IgnoredScope]) -> List[str]:
    """
    Convert the contents of the `IgnoredScope` class to the legacy ignored
    scope format.

    :param ignored_scope: The ignored scope.
    :return: An ignored scope in the legacy format as list.
    """
    results = []
    if ignored_scope is None:
        return results
    results.extend(ignored_scope.names)
    for p in ignored_scope.patterns:
        results.append("{re}" + p)
    if ignored_scope.types:
        raise RuntimeError("Legacy ignored scope format does not support operation types")
    return results


def get_ignored_node_names_from_ignored_scope(
    ignored_scope: IgnoredScope, nncf_graph: NNCFGraph, strict: bool = True
) -> List[str]:
    """
    Returns list of ignored names according to ignored scope and NNCFGraph.
    If strict is True, raises RuntimeError if any ignored name is not found in the NNCFGraph or
    any ignored pattern or any ignored type match 0 nodes in the NNCFGraph.
    If strict is False, returns all possible matches.

    :param ignored_scope: Given ignored scope instance.
    :param nncf_grpah: Given NNCFGrpah.
    :param strict: Whether all ignored_scopes must match at least one node or not.
    :returns: List of NNCF node names from given NNCFGraph specified in given ignored scope.
    """
    error_msg = (
        "Refer to the original_graph.dot to discover the operations"
        "in the model currently visible to NNCF and specify the ignored/target"
        " scopes in terms of the names there."
    )

    node_names = [node.node_name for node in nncf_graph.get_all_nodes()]
    matched_by_names = []
    if ignored_scope.names:
        for ignored_node_name in ignored_scope.names:
            if ignored_node_name in node_names:
                matched_by_names.append(ignored_node_name)
        if strict and len(ignored_scope.names) != len(matched_by_names):
            skipped_names = set(ignored_scope.names) - set(matched_by_names)
            raise RuntimeError(
                f"Ignored nodes with name {list(skipped_names)} were not found in the NNCFGraph. " + error_msg
            )
        nncf_logger.info(f"{len(matched_by_names)} ignored nodes was found by name in the NNCFGraph")

    matched_by_patterns = []
    if ignored_scope.patterns:
        not_matched_patterns = []
        for str_pattern in ignored_scope.patterns:
            pattern = re.compile(str_pattern)
            matches = list(filter(pattern.match, node_names))
            if not matches:
                not_matched_patterns.append(str_pattern)
            matched_by_patterns.extend(matches)
        if strict and not_matched_patterns:
            raise RuntimeError(f"No mathes for ignored patterns {not_matched_patterns} in the NNCFGraph. " + error_msg)
        nncf_logger.info(f"{len(matched_by_patterns)} ignored nodes was found by patterns in the NNCFGraph")

    matched_by_types = []
    if ignored_scope.types:
        types_found = set()
        for node in nncf_graph.get_all_nodes():
            if node.node_type in ignored_scope.types:
                types_found.add(node.node_type)
                matched_by_types.append(node.node_name)
        not_matched_types = set(ignored_scope.types) - types_found
        if strict and not_matched_types:
            raise RuntimeError(
                f"Nodes with ignored types {list(not_matched_types)} were not found in the NNCFGraph. " + error_msg
            )
        nncf_logger.info(f"{len(matched_by_types)} ignored nodes was found by types in the NNCFGraph")

    return list(set(matched_by_names + matched_by_types + matched_by_patterns))
