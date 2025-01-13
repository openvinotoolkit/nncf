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
import pytest

from nncf.common.graph import NNCFNode
from nncf.common.scopes import get_not_matched_scopes
from nncf.scopes import IgnoredScope
from nncf.scopes import Subgraph
from nncf.scopes import get_difference_ignored_scope


@pytest.mark.parametrize(
    "scope, ref",
    [
        ("A", []),
        ("1", ["1"]),
        (["A", "B"], []),
        (["1", "2"], ["1", "2"]),
        ([r"{re}\d"], [r"{re}\d"]),
        ([r"{re}\w"], []),
        (["A", "B", "{re}.*", "1"], ["1"]),
        (IgnoredScope(names=["A", "B"]), []),
        (IgnoredScope(names=["1", "2"]), ["1", "2"]),
        (IgnoredScope(patterns=[r"\d"]), [r"{re}\d"]),
        (IgnoredScope(patterns=[r"\w"]), []),
    ],
)
def test_get_not_matched_scopes(scope, ref):
    node_lists = [
        NNCFNode({NNCFNode.ID_NODE_ATTR: 1, NNCFNode.NODE_NAME_ATTR: "A"}),
        NNCFNode({NNCFNode.ID_NODE_ATTR: 2, NNCFNode.NODE_NAME_ATTR: "B"}),
    ]
    not_matched = get_not_matched_scopes(scope, node_lists)
    assert not set(not_matched) - set(ref)


@pytest.mark.parametrize(
    "scope_1, scope_2, ref",
    (
        (
            IgnoredScope(
                names=["A_name", "B_name"],
                patterns=["A_pattern", "B_pattern"],
                types=["A_type", "B_type"],
                subgraphs=[
                    Subgraph(inputs=["A_input"], outputs=["A_output"]),
                    Subgraph(inputs=["B_input"], outputs=["B_output"]),
                ],
            ),
            IgnoredScope(
                names=["B_name", "C_name"],
                patterns=["B_pattern", "C_pattern"],
                types=["B_type", "C_type"],
                subgraphs=[
                    Subgraph(inputs=["B_input"], outputs=["B_output"]),
                    Subgraph(inputs=["C_input"], outputs=["C_output"]),
                ],
            ),
            IgnoredScope(
                names=["A_name"],
                patterns=["A_pattern"],
                types=["A_type"],
                subgraphs=[Subgraph(inputs=["A_input"], outputs=["A_output"])],
            ),
        ),
    ),
)
def test_ignored_scope_diff(scope_1, scope_2, ref):
    assert get_difference_ignored_scope(scope_1, scope_2) == ref
