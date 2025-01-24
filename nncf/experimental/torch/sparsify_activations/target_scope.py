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
from typing import Set

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.scopes import IgnoredScope
from nncf.scopes import get_difference_ignored_scope
from nncf.scopes import get_matched_ignored_scope_info


@dataclass
class TargetScope(IgnoredScope):
    """
    Specifies the target portions in a model graph.

    Example:

    ..  code-block:: python
        # Specified by node names:
        node_names = ['node_1', 'node_2', 'node_3']
        target_scope = TargetScope(names=node_names)

        # Specified using regular expressions:
        patterns = ['.*node_\\d']
        target_scope = TargetScope(patterns=patterns)

        # Specified by operation types, e.g.,

        # OpenVINO opset https://docs.openvino.ai/latest/openvino_docs_ops_opset.html
        operation_types = ['Multiply', 'GroupConvolution', 'Interpolate']
        target_scope = TargetScope(types=operation_types)

        # ONNX opset https://github.com/onnx/onnx/blob/main/docs/Operators.md
        operation_types = ['Mul', 'Conv', 'Resize']
        target_scope = TargetScope(types=operation_types)

        # Specifies by subgraphs:
        from nncf import Subgraph
        target_scope = TargetScope(subgraphs=[
            Subgraph(inputs=["node_1"], outputs=["node_3"])
        ])

    **Note:** Operation types must be specified according to the model framework.

    :param names: List of target node names.
    :param patterns: List of regular expressions that define patterns for names of target nodes.
    :param types: List of target operation types.
    :param subgraphs: List of target subgraphs.
    :param validate: If set to True, then a RuntimeError will be raised if any target scope does not match
      in the model graph.
    """

    def __hash__(self) -> int:
        return hash(
            (
                frozenset(self.names),
                frozenset(self.patterns),
                frozenset(self.types),
                frozenset((frozenset(subgraph.inputs), frozenset(subgraph.outputs)) for subgraph in self.subgraphs),
                self.validate,
            )
        )


def get_target_node_names_from_target_scope(
    target_scope: TargetScope, nncf_graph: NNCFGraph, strict: bool = True
) -> Set[str]:
    """
    Returns NNCF node names from the graph that are matched by target scope.
    If strict is True, raises nncf.ValidationError if no rule is matched.

    :param target_scope: Target scope specifying the matching rules.
    :param nncf_graph: The graph.
    :param strict: Whether target_scope must match at least one node or not.
    :return: NNCF node names from the given graph matched by target scope.
    """
    matched_target_scope, matches = get_matched_ignored_scope_info(target_scope, [nncf_graph])
    if strict:
        _check_target_scope_strictly_matched(target_scope, matched_target_scope)
    return set().union(*matches.values())


def _check_target_scope_strictly_matched(target_scope: TargetScope, matched_target_scope: TargetScope):
    """
    Passes when target_scope and matched_target_scope are equal, otherwise raises ValidationError.

    :param target_scope: The given target scope.
    :param matched_target_scope: The actual target scope matched in a graph.
    """
    unmatched_scope = get_difference_ignored_scope(target_scope, matched_target_scope)
    error_messages = []
    for match_type in ("names", "types", "patterns", "subgraphs"):
        unmatched_rules = getattr(unmatched_scope, match_type)
        if unmatched_rules:
            error_messages.append(f"The following {match_type} are not found in the graph: {unmatched_rules}.")
    if error_messages:
        raise nncf.ValidationError("\n".join(error_messages))
