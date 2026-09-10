# Copyright (c) 2026 Intel Corporation
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

    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


def get_node_names_from_subgraph(graph: NNCFGraph, subgraph: Subgraph) -> list[str]:
    """
    Returns all names that are matched by the given subgraph.

    :param graph: Given NNCFGraph.
    :param subgraph: Given subgraph instance.
    :return: All names that are matched by the given subgraph.
    """
    matched_names = set()
    for start_node_name in subgraph.inputs:
        for end_node_name in subgraph.outputs:
            if start_node_name == end_node_name:
                # For networkx<3.3 nx.get_all_simple_paths returns empty path for this case
                node = graph.get_node_by_name(start_node_name)
                matched_names.add(node.node_name)
                continue
            for path in graph.get_all_simple_paths(start_node_name, end_node_name):
                for node_key in path:
                    node = graph.get_node_by_key(node_key)
                    matched_names.add(node.node_name)

    return list(sorted(matched_names))


@dataclass
class BaseScope:
    """
    Defines a portion of a model by a set of matching rules against the model graph.

    :param names: List of node names.
    :param patterns: List of regular expressions that define patterns for names of nodes.
    :param types: List of operation types.
    :param subgraphs: List of subgraphs.
    :param validate: If set to True, then an error will be raised if any rule does not match
      in the model graph.
    """

    names: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    subgraphs: list[Subgraph] = field(default_factory=list)
    validate: bool = True


@api(canonical_alias="nncf.IgnoredScope")
@dataclass
class IgnoredScope(BaseScope):
    r"""
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


@api(canonical_alias="nncf.CustomAnnotationScope")
@dataclass
class CustomAnnotationScope(BaseScope):
    r"""
    Provides an option to specify portions of model to be annotated with a custom compression configuration.

    The custom annotation scope defines model nodes for which the user-defined compression configuration is
    used instead of the one assigned by the algorithm. Matching rules are the same as for
    :class:`nncf.IgnoredScope`.

    Example:

    ..  code-block:: python

            import nncf

            # Annotate by node name:
            node_names = ['node_1', 'node_2', 'node_3']
            scope = nncf.CustomAnnotationScope(names=node_names)

            # Annotate using regular expressions:
            patterns = ['.*self_attn.*', '.*router.*']
            scope = nncf.CustomAnnotationScope(patterns=patterns)

    **Note:** Operation types must be specified according to the model framework.

    :param names: List of annotated node names.
    :type names: List[str]
    :param patterns: List of regular expressions that define patterns for names of annotated nodes.
    :type patterns: List[str]
    :param types: List of annotated operation types.
    :type types: List[str]
    :param subgraphs: List of annotated subgraphs.
    :type subgraphs: List[Subgraph]
    :param validate: If set to True, then an error will be raised if any annotation rule does not match
      in the model graph.
    :type validate: bool
    """


def get_difference_scope(scope_1: BaseScope, scope_2: BaseScope) -> BaseScope:
    """
    Returns scope with rules from 'scope_1' not presented at 'scope_2'.
    The returned scope has the same type as 'scope_1'.

    :param scope_1: First scope.
    :param scope_2: Second scope.
    :return: Scope with the rules difference.
    """
    cls = scope_1.__class__
    return cls(
        names=list(set(scope_1.names) - set(scope_2.names)),
        patterns=list(set(scope_1.patterns) - set(scope_2.patterns)),
        types=list(set(scope_1.types) - set(scope_2.types)),
        subgraphs=[subgraph for subgraph in scope_1.subgraphs if subgraph not in scope_2.subgraphs],
        validate=scope_1.validate,
    )


def convert_ignored_scope_to_list(ignored_scope: IgnoredScope | None) -> list[str]:
    """
    Convert the contents of the `IgnoredScope` class to the legacy ignored
    scope format.

    :param ignored_scope: The ignored scope.
    :return: An ignored scope in the legacy format as list.
    """
    results: list[str] = []
    if ignored_scope is None:
        return results
    results.extend(ignored_scope.names)
    for p in ignored_scope.patterns:
        results.append("{re}" + p)
    if ignored_scope.types:
        msg = "Legacy ignored scope format does not support operation types"
        raise nncf.InternalError(msg)
    return results


def get_matched_scope_info(scope: BaseScope, nncf_graphs: list[NNCFGraph]) -> tuple[BaseScope, dict[str, set[str]]]:
    """
    Returns matched scope for provided graphs along with all found matches.
    The resulted scope consist of all matched rules and has the same type as the given scope.
    The found matches consist of a dictionary with a rule name as a key and matched node names as a value.

    :param scope: Scope instance.
    :param nncf_graphs: Graphs.
    :returns: Matched scope along with all matches.
    """
    names, patterns, types, subgraphs_numbers = set(), set(), set(), set()  # type: ignore
    matches = {"names": names, "patterns": set(), "types": set(), "subgraphs": set()}

    for graph in nncf_graphs:
        if scope.names or scope.patterns:
            node_names = set(node.node_name for node in graph.nodes.values())

            for matched_node_name in filter(lambda name: name in node_names, scope.names):
                names.add(matched_node_name)

            for str_pattern in scope.patterns:
                pattern = re.compile(str_pattern)
                pattern_matched_names = set(filter(pattern.match, node_names))
                if pattern_matched_names:
                    matches["patterns"].update(pattern_matched_names)
                    patterns.add(str_pattern)

        for node in graph.get_nodes_by_types(scope.types):
            matches["types"].add(node.node_name)
            types.add(node.node_type)

        for i, subgraph in enumerate(scope.subgraphs):
            names_from_subgraph = get_node_names_from_subgraph(graph, subgraph)
            if names_from_subgraph:
                matches["subgraphs"].update(names_from_subgraph)
                subgraphs_numbers.add(i)

    cls = scope.__class__
    matched_scope = cls(
        names=list(names),
        patterns=list(patterns),
        types=list(types),
        subgraphs=[subgraph for i, subgraph in enumerate(scope.subgraphs) if i in subgraphs_numbers],
        validate=scope.validate,
    )
    return matched_scope, matches


def _info_matched_scope(matches: dict[str, set[str]], scope_kind: str = "Ignored") -> None:
    """
    Log matches.

    :param matches: Matches.
    :param scope_kind: Human-readable kind of the scope to mention in the message.
    """
    for rule_type, rules in matches.items():
        if rules:
            nncf_logger.info(f"{len(rules)} {scope_kind.lower()} nodes were found by {rule_type} in the NNCFGraph")


def _error_unmatched_scope(unmatched_scope: BaseScope, scope_kind: str = "Ignored") -> str:
    """
    Returns an error message for unmatched scope.

    :param unmatched_scope: Unmatched scope.
    :param scope_kind: Human-readable kind of the scope to mention in the message.
    :return str: Error message.
    """
    err_msg = "\n"
    for rule_type in ("names", "types", "patterns"):
        unmatched_rules = getattr(unmatched_scope, rule_type)
        if unmatched_rules:
            err_msg += (
                f"{scope_kind} nodes that matches {rule_type} {unmatched_rules} were not found in the NNCFGraph.\n"
            )
    for subgraph in unmatched_scope.subgraphs:
        err_msg += (
            f"{scope_kind} nodes that matches subgraph with input names {subgraph.inputs} "
            f"and output names {subgraph.outputs} were not found in the NNCFGraph.\n"
        )
    return err_msg


def _check_scope_strictly_matched(scope: BaseScope, matched_scope: BaseScope, scope_kind: str = "Ignored") -> None:
    """
    Passes when scope and matched_scope are equal, otherwise - raises ValidationError.

    :param scope: Scope.
    :param matched_scope: Matched scope.
    :param scope_kind: Human-readable kind of the scope to mention in the error message.
    """
    unmatched_scope = get_difference_scope(scope, matched_scope)
    if unmatched_scope.names or unmatched_scope.types or unmatched_scope.patterns or unmatched_scope.subgraphs:
        raise nncf.ValidationError(_error_unmatched_scope(unmatched_scope, scope_kind))


def get_node_names_from_scope(
    scope: BaseScope, nncf_graph: NNCFGraph, strict: bool = True, scope_kind: str = "Ignored"
) -> set[str]:
    """
    Returns matched names according to the scope and NNCFGraph.
    If strict is True, raises nncf.ValidationError if any rule was not matched.
    If strict is False, returns all possible matches.

    :param scope: Scope.
    :param nncf_graph: Graph.
    :param strict: Whether all scope rules must match at least one node or not.
    :param scope_kind: Human-readable kind of the scope to mention in the error message.
    :return: NNCF node names from given NNCFGraph specified in given scope.
    """
    matched_scope, matches = get_matched_scope_info(scope, [nncf_graph])
    if strict:
        _check_scope_strictly_matched(scope, matched_scope, scope_kind)
    _info_matched_scope(matches, scope_kind)
    return {name for match in matches.values() for name in match}


def validate_scope(scope: BaseScope, nncf_graphs: list[NNCFGraph]) -> None:
    """
    Passes whether all rules at 'scope' have matches at provided graphs, otherwise - raises ValidationError.

    :param scope: Scope.
    :param nncf_graphs: Graphs.
    """
    matched_scope, _ = get_matched_scope_info(scope, nncf_graphs)
    _check_scope_strictly_matched(scope, matched_scope)
