# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import re
from dataclasses import dataclass
from dataclasses import field
from typing import List, Optional, OrderedDict, Set

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
        raise nncf.InternalError("Legacy ignored scope format does not support operation types")
    return results


@dataclass
class IgnoredScopeMatch:
    matched_ignored_scope: IgnoredScope = IgnoredScope()
    matches: OrderedDict[str, Set[str]] = field(default_factory=lambda: OrderedDict(dict))


def get_ignored_scope_match(ignored_scope: IgnoredScope, nncf_graphs: List[NNCFGraph]) -> IgnoredScopeMatch:
    """
    Returns ignored scope match for provided NNCFGraphs.
    The resulted match is a union of all matches across graphs.

    :param ignored_scope: Ignored scope instance.
    :param nncf_graphs: NNCFGraphs.
    :returns: ignored scope match united all mathces across graphs
    """
    names, patterns, types, subgraphs_numbers = set(), set(), set(), set()
    for graph in nncf_graphs:
        matches = collections.OrderedDict({"name": set(), "patterns": set(), "types": set(), "subgraphs": set()})
        node_names = set(node.node_name for node in graph.nodes.values())

        for ignored_node_name in filter(lambda name: name in node_names, ignored_scope.names):
            matches["name"].add(ignored_node_name)
            names.add(ignored_node_name)

        for str_pattern in ignored_scope.patterns:
            pattern = re.compile(str_pattern)
            matches = list(filter(pattern.match, node_names))
            matches["patterns"].add(matches)
            patterns.add(str_pattern)

        ignored_scope_types = set(ignored_scope.types)
        for node in graph.get_nodes_by_types(ignored_scope_types):
            matches["types"].add(node.node_name)
            types.add(node.node_type)

        for i, subgraph in enumerate(ignored_scope.subgraphs):
            names_from_subgraph = get_ignored_node_names_from_subgraph(graph, subgraph)
            matches["subgraphs"].update(names_from_subgraph)
            subgraphs_numbers.add(i)
    matched_ignored_scope = IgnoredScope(
        names=list(names),
        patterns=list(patterns),
        types=list(types),
        subgraphs=[subgraph for i, subgraph in enumerate(ignored_scope.subgraphs) if i in subgraphs_numbers],
        validate=ignored_scope.validate,
    )
    return IgnoredScopeMatch(matched_ignored_scope, matches)


def get_unmatched_ignored_scope(matched_ignored_scope: IgnoredScope, ignored_scope: IgnoredScope) -> IgnoredScope:
    """
    Returns unmatched ignored scope rules from full ignored scope and matched ignored scope.

    :param matched_ignored_scope: Matched ingored scope.
    :param ignored_scope: Full ignored scope.
    :return: Unmatched ignored scope.
    """
    assert matched_ignored_scope.validate == ignored_scope.validate
    return IgnoredScope(
        names=[name for name in ignored_scope.names if name not in matched_ignored_scope.names],
        patterns=[pattern for pattern in ignored_scope.patterns if pattern not in matched_ignored_scope.patterns],
        types=[type for type in ignored_scope.types if type not in matched_ignored_scope.types],
        subgraphs=[subgraph for subgraph in ignored_scope.subgraphs if subgraph not in matched_ignored_scope.subgraphs],
        validate=matched_ignored_scope.validate,
    )


def info_matched_ignored_scope(matches) -> None:
    """
    Log matches.

    :param matches: Matches.
    """
    for rule_type, rules in matches.items():
        if rules:
            nncf_logger.info(f"{len(rules)} ignored nodes were found by {rule_type} in the NNCFGraph")


def error_unmatched_ignored_scope(unmatched_ignored_scope: IgnoredScope) -> str:
    """
    Returns an error message for unmatched ignored scope.

    :param unmatched_ignored_scope: Unmatched ignored scope.
    :return str: Error message.
    """
    err_msg = ""
    if unmatched_ignored_scope.names:
        err_msg += f"Ignored nodes with name {unmatched_ignored_scope.names} were not found in the NNCFGraph. "
    if unmatched_ignored_scope.patterns:
        err_msg += f"No matches for ignored patterns {unmatched_ignored_scope.patterns} in the NNCFGraph. "
    if unmatched_ignored_scope.types:
        err_msg += f"Nodes with ignored types {unmatched_ignored_scope.types} were not found in the NNCFGraph. "
    for subgraph in unmatched_ignored_scope.subgraphs:
        err_msg += (
            f"Ignored subgraph with input names {subgraph.inputs} and output names {subgraph.outputs} "
            "was not found in the NNCFGraph. "
        )
    return err_msg + (
        "Refer to the original_graph.dot to discover the operations"
        "in the model currently visible to NNCF and specify the ignored/target"
        " scopes in terms of the names there."
    )


def get_ignored_node_names_from_ignored_scope(
    ignored_scope: IgnoredScope, nncf_graph: NNCFGraph, strict: bool = True
) -> Set[str]:
    """
    Returns ignored names according to ignored scope and NNCFGraph.
    If strict is True, raises RuntimeError if any ignored rule was not matched.
    If strict is False, returns all possible matches.

    :param ignored_scope: Ignored scope.
    :param nncf_graph: Graph.
    :param strict: Whether all ignored_scopes must match at least one node or not.
    :return: NNCF node names from given NNCFGraph specified in given ignored scope.
    """
    match = get_ignored_scope_match(ignored_scope, [nncf_graph])
    if strict:
        validate_ignored_scope(ignored_scope, match.matched_ignored_scope)
    info_matched_ignored_scope(match.matches)
    return {name for match in match.matches.values() for name in match}


def validate_ignored_scope(ignored_scope: IgnoredScope, matched_ignored_scope: IgnoredScope):
    """
    Checks whether the every rule in ignored scope has a match.

    :param ignored_scope: Ignored scope.
    :param matched_ignored_scope: Matched Ignored scope.
    """
    unmatched_ignored_scope = get_unmatched_ignored_scope(matched_ignored_scope, ignored_scope)
    if (
        any(unmatched_ignored_scope.names)
        or any(unmatched_ignored_scope.types)
        or any(unmatched_ignored_scope.patterns)
        or any(unmatched_ignored_scope.subgraphs)
    ):
        raise nncf.ValidationError(error_unmatched_ignored_scope(unmatched_ignored_scope))
