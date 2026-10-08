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
from typing import Iterable

from nncf.common.graph import NNCFNodeName
from nncf.common.quantization.structs import QuantizerId


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
