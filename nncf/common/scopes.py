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
from typing import List, Optional, Set, Union

import nncf
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.common.logging import nncf_logger
from nncf.common.quantization.structs import QuantizerId
from nncf.scopes import IgnoredScope
from nncf.scopes import convert_ignored_scope_to_list


def matches_any(tested_str: str, strs_to_match_to: Union[List[str], Set[str], str]) -> bool:
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
    serializable_id: Union[QuantizerId, NNCFNodeName],
    ignored_scopes: Union[List[str], Set[str]],
    target_scopes: Optional[List[str]] = None,
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


def get_not_matched_scopes(scope: Union[List[str], str, IgnoredScope, None], nodes: List[NNCFNode]) -> List[str]:
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
        patterns = scope

    if not patterns:
        return []

    matched_patterns = set()
    for node in nodes:
        for pattern in patterns:
            if matches_any(node.node_name, pattern):
                matched_patterns.add(pattern)
    return list(set(patterns) - matched_patterns)


def check_scopes_in_graph(
    graph: NNCFGraph,
    ignored_scopes: Union[IgnoredScope, List[str]],
    target_scopes: Optional[List[str]] = None,
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
            "No match has been found among the model operations "
            "for the following ignored/target scope definitions:\n"
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
