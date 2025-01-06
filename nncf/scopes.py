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

import re
from dataclasses import dataclass
from dataclasses import field
from typing import Dict, List, Optional, Set, Tuple

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.logging import nncf_logger
from nncf.common.utils.api_marker import api


@api(canonical_alias="nncf.Subgraph")
@dataclass
class Subgraph:
    """
    Defines the ignored subgraph as follows: A subgraph comprises all nodes along
    all simple paths in the graph from input to output nodes.

    :param inputs: Input node names.
    :type inputs: List[str]
    :param outputs: Output node names.
    :type outputs: List[str]
    """

    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


def get_ignored_node_names_from_subgraph(graph: NNCFGraph, subgraph: Subgraph) -> List[str]:
    """
    Returns all names that should be ignored according to given subgraph.

    :param graph: Given NNCFGraph.
    :param subgraph: Given subgraph instance.
    :return: All names that should be ignored according to given subgraph.
    """
    ignored_names = set()
    for start_node_name in subgraph.inputs:
        for end_node_name in subgraph.outputs:
            if start_node_name == end_node_name:
                # For networkx<3.3 nx.get_all_simple_paths returns empty path for this case
                node = graph.get_node_by_name(start_node_name)
                ignored_names.add(node.node_name)
                continue
            for path in graph.get_all_simple_paths(start_node_name, end_node_name):
                for node_key in path:
                    node = graph.get_node_by_key(node_key)
                    ignored_names.add(node.node_name)

    return list(sorted(ignored_names))


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
    :param subgraphs: List of ignored subgraphs.
    :type subgraphs: List[Subgraph]
    :param validate: If set to True, then a RuntimeError will be raised if any ignored scope does not match
      in the model graph.
    :type types: bool
    """

    names: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    types: List[str] = field(default_factory=list)
    subgraphs: List[Subgraph] = field(default_factory=list)
    validate: bool = True


def get_difference_ignored_scope(ignored_scope_1: IgnoredScope, ignored_scope_2: IgnoredScope) -> IgnoredScope:
    """
    Returns ignored scope with rules from 'ignored_scope_1' not presented at 'ignored_scope_2'

    :param ignored_scope_1: First ignored scope.
    :param ignored_scope_2: Second ignored scope.
    :return: Ignored scope.
    """
    return IgnoredScope(
        names=list(set(ignored_scope_1.names) - set(ignored_scope_2.names)),
        patterns=list(set(ignored_scope_1.patterns) - set(ignored_scope_2.patterns)),
        types=list(set(ignored_scope_1.types) - set(ignored_scope_2.types)),
        subgraphs=[subgraph for subgraph in ignored_scope_1.subgraphs if subgraph not in ignored_scope_2.subgraphs],
        validate=ignored_scope_1.validate,
    )


def convert_ignored_scope_to_list(ignored_scope: Optional[IgnoredScope]) -> List[str]:
    """
    Convert the contents of the `IgnoredScope` class to the legacy ignored
    scope format.

    :param ignored_scope: The ignored scope.
    :return: An ignored scope in the legacy format as list.
    """
    results: List[str] = []
    if ignored_scope is None:
        return results
    results.extend(ignored_scope.names)
    for p in ignored_scope.patterns:
        results.append("{re}" + p)
    if ignored_scope.types:
        raise nncf.InternalError("Legacy ignored scope format does not support operation types")
    return results


def get_matched_ignored_scope_info(
    ignored_scope: IgnoredScope, nncf_graphs: List[NNCFGraph]
) -> Tuple[IgnoredScope, Dict[str, Set[str]]]:
    """
    Returns matched ignored scope for provided graphs along with all found matches.
    The resulted ignored scope consist of all matched rules.
    The found matches consist of a dictionary with a rule name as a key and matched node names as a value.

    :param ignored_scope: Ignored scope instance.
    :param nncf_graphs: Graphs.
    :returns: Matched ignored scope along with all matches.
    """
    names, patterns, types, subgraphs_numbers = set(), set(), set(), set()  # type: ignore
    matches = {"names": names, "patterns": set(), "types": set(), "subgraphs": set()}

    for graph in nncf_graphs:
        if ignored_scope.names or ignored_scope.patterns:
            node_names = set(node.node_name for node in graph.nodes.values())

            for ignored_node_name in filter(lambda name: name in node_names, ignored_scope.names):
                names.add(ignored_node_name)

            for str_pattern in ignored_scope.patterns:
                pattern = re.compile(str_pattern)
                pattern_matched_names = set(filter(pattern.match, node_names))
                if pattern_matched_names:
                    matches["patterns"].update(pattern_matched_names)
                    patterns.add(str_pattern)

        for node in graph.get_nodes_by_types(ignored_scope.types):
            matches["types"].add(node.node_name)
            types.add(node.node_type)

        for i, subgraph in enumerate(ignored_scope.subgraphs):
            names_from_subgraph = get_ignored_node_names_from_subgraph(graph, subgraph)
            if names_from_subgraph:
                matches["subgraphs"].update(names_from_subgraph)
                subgraphs_numbers.add(i)

    matched_ignored_scope = IgnoredScope(
        names=list(names),
        patterns=list(patterns),
        types=list(types),
        subgraphs=[subgraph for i, subgraph in enumerate(ignored_scope.subgraphs) if i in subgraphs_numbers],
        validate=ignored_scope.validate,
    )
    return matched_ignored_scope, matches


def _info_matched_ignored_scope(matches: Dict[str, Set[str]]) -> None:
    """
    Log matches.

    :param matches: Matches.
    """
    for rule_type, rules in matches.items():
        if rules:
            nncf_logger.info(f"{len(rules)} ignored nodes were found by {rule_type} in the NNCFGraph")


def _error_unmatched_ignored_scope(unmatched_ignored_scope: IgnoredScope) -> str:
    """
    Returns an error message for unmatched ignored scope.

    :param unmatched_ignored_scope: Unmatched ignored scope.
    :return str: Error message.
    """
    err_msg = "\n"
    for ignored_type in ("names", "types", "patterns"):
        unmatched_rules = getattr(unmatched_ignored_scope, ignored_type)
        if unmatched_rules:
            err_msg += f"Ignored nodes that matches {ignored_type} {unmatched_rules} were not found in the NNCFGraph.\n"
    for subgraph in unmatched_ignored_scope.subgraphs:
        err_msg += (
            f"Ignored nodes that matches subgraph with input names {subgraph.inputs} "
            f"and output names {subgraph.outputs} were not found in the NNCFGraph.\n"
        )
    return err_msg


def _check_ignored_scope_strictly_matched(ignored_scope: IgnoredScope, matched_ignored_scope: IgnoredScope) -> None:
    """
    Passes when ignored_scope and matched_ignored_scope are equal, otherwise - raises ValidationError.

    :param ignored_scope: Ignored scope.
    :param matched_ignored_scope: Matched ignored scope.
    """
    unmatched_ignored_scope = get_difference_ignored_scope(ignored_scope, matched_ignored_scope)
    if (
        unmatched_ignored_scope.names
        or unmatched_ignored_scope.types
        or unmatched_ignored_scope.patterns
        or unmatched_ignored_scope.subgraphs
    ):
        raise nncf.ValidationError(_error_unmatched_ignored_scope(unmatched_ignored_scope))


def get_ignored_node_names_from_ignored_scope(
    ignored_scope: IgnoredScope, nncf_graph: NNCFGraph, strict: bool = True
) -> Set[str]:
    """
    Returns ignored names according to ignored scope and NNCFGraph.
    If strict is True, raises nncf.ValidationError if any ignored rule was not matched.
    If strict is False, returns all possible matches.

    :param ignored_scope: Ignored scope.
    :param nncf_graph: Graph.
    :param strict: Whether all ignored_scopes must match at least one node or not.
    :return: NNCF node names from given NNCFGraph specified in given ignored scope.
    """
    matched_ignored_scope, matches = get_matched_ignored_scope_info(ignored_scope, [nncf_graph])
    if strict:
        _check_ignored_scope_strictly_matched(ignored_scope, matched_ignored_scope)
    _info_matched_ignored_scope(matches)
    return {name for match in matches.values() for name in match}


def validate_ignored_scope(ignored_scope: IgnoredScope, nncf_graphs: List[NNCFGraph]) -> None:
    """
    Passes whether all rules at 'ignored_scope' have matches at provided graphs, otherwise - raises ValidationError.

    :param ignored_scope: Ignored scope.
    :param nncf_graphs: Graphs.
    """
    matched_ignored_scope, _ = get_matched_ignored_scope_info(ignored_scope, nncf_graphs)
    _check_ignored_scope_strictly_matched(ignored_scope, matched_ignored_scope)
