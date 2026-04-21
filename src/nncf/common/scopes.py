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
from collections import deque
from typing import Container, Iterable

import nncf
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.logging import nncf_logger
from nncf.common.quantization.structs import QuantizerId
from nncf.scopes import IgnoredScope
from nncf.scopes import convert_ignored_scope_to_list


def matches_any(tested_str: str, strs_to_match_to: Iterable[str] | str | None) -> bool:
    """
    Return True if tested_str matches at least one element in strs_to_match_to.

    :param tested_str: One of the supported entity types to be matched - currently possible to pass either
        NNCFNodeName (to refer to the original model operations) or QuantizerId (to refer to specific quantizers).
    :param strs_to_match_to: A list or set of strings specifying for the serializable_id. Entries of the strings
        may be prefixed with `{re}` to enable regex matching.

    :return: A boolean value specifying whether a tested_str should matches at least one element
        in strs_to_match_to.
    """
    if strs_to_match_to is None:
        return False

    str_list = [strs_to_match_to] if isinstance(strs_to_match_to, str) else strs_to_match_to
    for item in str_list:
        if "{re}" in item:
            regex = item.replace("{re}", "")
            if re.search(regex, tested_str):
                return True
        else:
            if tested_str == item:
                return True
    return False


def should_consider_scope(
    serializable_id: QuantizerId | NNCFNodeName,
    ignored_scopes: Iterable[str] | None,
    target_scopes: Iterable[str] | None = None,
) -> bool:
    """
    Used when an entity arising during compression has to be compared to an allowlist or a denylist of strings.

    :param serializable_id: One of the supported entity types to be matched - currently possible to pass either
        NNCFNodeName (to refer to the original model operations) or QuantizerId (to refer to specific quantizers)
    :param ignored_scopes: A list or set of strings specifying a denylist for the serializable_id. Entries of the list
        may be prefixed with `{re}` to enable regex matching.
    :param target_scopes: A list of strings specifying an allowlist for the serializable_id. Entries of the list
        may be prefixed with `{re}` to enable regex matching.

    :return: A boolean value specifying whether a serializable_id should be considered (i.e. "not ignored", "targeted")
    """
    string_id = str(serializable_id)
    return (target_scopes is None or matches_any(string_id, target_scopes)) and not matches_any(
        string_id, ignored_scopes
    )


def get_not_matched_scopes(scope: list[str] | str | IgnoredScope | None, nodes: list[NNCFNode]) -> list[str]:
    """
    Return list of scope that do not match node list.

    :param scope: List of ignored/target scope or instance of IgnoredScope.
    :param graph: The model graph.

    :return : List of not matched scopes.
    """
    if scope is None:
        return []

    if isinstance(scope, str):
        patterns = [scope]
    elif isinstance(scope, IgnoredScope):
        patterns = convert_ignored_scope_to_list(scope)
    else:
        patterns = list(scope)

    if not patterns:
        return []

    matched_patterns = set()
    for node in nodes:
        for pattern in patterns:
            if matches_any(node.node_name, pattern):
                matched_patterns.add(pattern)
    return list(set(patterns) - matched_patterns)


def propagate_ignored_constants_to_weighted_consumers(
    ignored_names: set[str],
    nncf_graph: NNCFGraph,
    weighted_metatypes: Container[type[OperatorMetatype]],
    constant_metatypes: Container[type[OperatorMetatype]],
    passthrough_metatypes: Container[type[OperatorMetatype]] = (),
    passthrough_node_types: Container[str] = (),
) -> set[str]:
    """
    Expand a resolved IgnoredScope name set to include weighted operations reachable
    from any ignored Constant node.

    IgnoredScope name matching runs against every node in the NNCFGraph, including
    Constants. But algorithms such as ``nncf.compress_weights`` and ``nncf.prune``
    iterate only over weighted operation nodes (MatMul, Embedding, Convolution, ...).
    A user who names a weight Constant in IgnoredScope therefore sees the node
    matched but the weight is still modified, because the consuming operation is
    never compared against the ignored set. This helper rewrites such cases so that
    the consuming operation is also skipped, matching user intent.

    For each ignored name whose node metatype is in ``constant_metatypes``, the
    helper performs a forward BFS through the graph. When a node with a metatype
    in ``weighted_metatypes`` is reached, its name is added to the ignored set;
    traversal does not continue past a weighted op. When a node with a metatype in
    ``passthrough_metatypes`` or a node_type in ``passthrough_node_types`` is
    reached, BFS continues through its successors - this mirrors NNCF's existing
    reverse walk used by ``get_operation_const_op`` for patterns like
    ``Constant -> Convert -> FakeQuantize -> Reshape -> Operation``. Any other node
    halts that branch of the walk.

    Callers that do not have any "passthrough" nodes between Constants and their
    consumers (e.g. ``nncf.prune`` on a raw PyTorch model) may leave the passthrough
    parameters empty; in that case the walk resolves in a single hop.

    :param ignored_names: Names resolved from an IgnoredScope against ``nncf_graph``.
        All names are expected to belong to ``nncf_graph``.
    :param nncf_graph: Graph containing the nodes referenced by ``ignored_names``.
    :param weighted_metatypes: Metatypes that the calling algorithm iterates over as
        candidates for compression/pruning (e.g. ``matmul_metatypes +
        embedding_metatypes + convolution_metatypes`` on a weight-compression
        backend). Reaching one of these stops traversal along that branch and adds
        the node's name to the ignored set.
    :param constant_metatypes: Metatypes representing weight Constants on the
        caller's backend (e.g. ``[OVConstantMetatype]`` or ``[PTConstNoopMetatype]``).
        Only ignored nodes whose metatype is in this set trigger propagation.
    :param passthrough_metatypes: Optional metatypes of nodes that should be walked
        through during propagation without halting - typically Convert, Reshape,
        FakeQuantize and similar dtype/shape/quantization passthroughs. Matches
        NNCF's existing convention in ``get_operation_const_op``.
    :param passthrough_node_types: Optional node_type strings to walk through, for
        backends that identify passthrough nodes by node_type rather than metatype
        (e.g. PyTorch's ``symmetric_quantize`` / ``apply_magnitude_binary_mask``).
    :return: A superset of ``ignored_names`` including consuming weighted ops.
    """
    expanded = set(ignored_names)
    added: set[str] = set()
    for name in ignored_names:
        node = nncf_graph.get_node_by_name(name)
        if node.metatype not in constant_metatypes:
            continue
        queue = deque(nncf_graph.get_next_nodes(node))
        visited: set[int] = set()
        while queue:
            current = queue.popleft()
            if current.node_id in visited:
                continue
            visited.add(current.node_id)
            if current.metatype in weighted_metatypes:
                if current.node_name not in expanded:
                    added.add(current.node_name)
                    expanded.add(current.node_name)
                continue
            if current.metatype in passthrough_metatypes or current.node_type in passthrough_node_types:
                queue.extend(nncf_graph.get_next_nodes(current))
    if added:
        nncf_logger.info(
            f"IgnoredScope propagation: {len(added)} weighted consumer(s) of ignored Constant node(s) "
            f"were also excluded: {sorted(added)}"
        )
    return expanded


def check_scopes_in_graph(
    graph: NNCFGraph,
    ignored_scopes: IgnoredScope | list[str],
    target_scopes: list[str] | None = None,
    validate_scopes: bool = True,
) -> None:
    """
    Raise RuntimeError in case if ignored/target scope names do not match model graph.

    :param graph: The model graph.
    :param ignored_scopes: The instance of IgnoredScope or a list of strings specifying a denylist
        for the serializable_id.
    :param target_scopes: A list of strings specifying an allowlist for the serializable_id.
    :param validate_scopes: If set to True, then a RuntimeError will be raised if the names of the
      ignored/target scopes do not match the names of the scopes in the model graph.
    """
    node_list = graph.get_all_nodes()
    not_matched_ignored_scopes = get_not_matched_scopes(ignored_scopes, node_list)
    not_matched_target_scopes = get_not_matched_scopes(target_scopes, node_list)

    if not_matched_ignored_scopes or not_matched_target_scopes:
        err_message = (
            "No match has been found among the model operations for the following ignored/target scope definitions:\n"
        )
        if not_matched_ignored_scopes:
            err_message += f" - ignored_scope: {not_matched_ignored_scopes}\n"
        if not_matched_target_scopes:
            err_message += f" - target_scope: {not_matched_target_scopes}\n"

        err_message += (
            "Refer to the original_graph.dot to discover the operations "
            "in the model currently visible to NNCF and specify the ignored/target "
            "scopes in terms of the names there."
        )

        if validate_scopes:
            raise nncf.ValidationError(err_message)
        nncf_logger.info(err_message)
