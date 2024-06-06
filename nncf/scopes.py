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
import itertools
import re
from dataclasses import asdict
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


def merge_ignored_scopes(ignored_scope_1: IgnoredScope, ignored_scope_2: IgnoredScope):
    assert ignored_scope_1.validate == ignored_scope_2.validate
    d1 = asdict(ignored_scope_1)
    for attr, val in asdict(ignored_scope_2).items():
        if attr == "validate":
            continue
        d1[attr] = list(set(itertools.chain(d1[attr], val)))
    return IgnoredScope(**d1, validate=ignored_scope_1.validate)


@dataclass
class IgnoredScopeMatch:
    matched_ignored_scope: IgnoredScope = IgnoredScope()
    matches: OrderedDict[str, Set[str]] = field(default_factory=lambda: OrderedDict(dict))


def get_matched_ignored_scope(ignored_scope: IgnoredScope, nncf_graph: NNCFGraph) -> IgnoredScopeMatch:
    """
    Returns ignored names according to ignored scope and NNCFGraph.
    If strict is True, raises RuntimeError if any ignored name is not found in the NNCFGraph or
    any ignored pattern or any ignored type match 0 nodes in the NNCFGraph.
    If strict is False, returns all possible matches.

    :param ignored_scope: Given ignored scope instance.
    :param nncf_graph: Given NNCFGraph.
    :param strict: Whether all ignored_scopes must match at least one node or not.
    :returns: NNCF node names from given NNCFGraph specified in given ignored scope.
    """

    matched_ignored_scope = IgnoredScope()
    matches = collections.OrderedDict({"name": set(), "patterns": set(), "types": set(), "subgraphs": set()})
    node_names = set(node.node_name for node in nncf_graph.nodes.values())

    for ignored_node_name in ignored_scope.names:
        if ignored_node_name in node_names:
            matches["name"].add(ignored_node_name)
            matched_ignored_scope.names.append(ignored_node_name)

    for str_pattern in ignored_scope.patterns:
        pattern = re.compile(str_pattern)
        matches = list(filter(pattern.match, node_names))
        matches["patterns"].add(matches)
        matched_ignored_scope.patterns.append(str_pattern)

    ignored_scope_types = set(ignored_scope.types)
    for node in nncf_graph.get_nodes_by_types(ignored_scope_types):
        matches["types"].add(node.node_name)
        matched_ignored_scope.types.append(node.node_type)

    for subgraph in ignored_scope.subgraphs:
        names_from_subgraph = get_ignored_node_names_from_subgraph(nncf_graph, subgraph)
        matches["subgraphs"].update(names_from_subgraph)
        matched_ignored_scope.subgraphs.append(subgraph)
    return IgnoredScopeMatch(matched_ignored_scope, matches)


def get_unmatched_ignored_scope(matched_ignored_scope: IgnoredScope, ignored_scope: IgnoredScope) -> IgnoredScope:
    return IgnoredScope(
        names=[name for name in ignored_scope.names if name not in matched_ignored_scope.names],
        patterns=[pattern for pattern in ignored_scope.patterns if pattern not in matched_ignored_scope.patterns],
        types=[type for type in ignored_scope.types if type not in matched_ignored_scope.types],
        subgraphs=[subgraph for subgraph in ignored_scope.subgraphs if subgraph not in matched_ignored_scope.subgraphs],
    )


def _info_matched_ignored_scope(matches):
    info_msg = ""
    for rule_type, rules in matches.items():
        if rules:
            info_msg += f"{len(rules)} ignored nodes were found by {rule_type} in the NNCFGraph"
    return info_msg


def _error_unmatched_ignored_scope(unmatched_ignored_scope: IgnoredScope):
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


def get_ignored_node_names_from_ignored_scope(ignored_scope: IgnoredScope, nncf_graph: NNCFGraph, strict: bool = True):
    match = get_matched_ignored_scope(ignored_scope, nncf_graph)
    unmatched_ignored_scope = get_unmatched_ignored_scope(match.matched_ignored_scope, ignored_scope)
    if strict and (
        any(unmatched_ignored_scope.names)
        or any(unmatched_ignored_scope.types)
        or any(unmatched_ignored_scope.patterns)
        or any(unmatched_ignored_scope.subgraphs)
    ):
        raise nncf.ValidationError(_error_unmatched_ignored_scope(unmatched_ignored_scope, ignored_scope))
    nncf_logger.info(_info_matched_ignored_scope(match.matches))
    return {name for match in match.matches.values() for name in match}
