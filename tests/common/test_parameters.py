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

import nncf
from nncf.scopes import IgnoredScope
from nncf.scopes import convert_ignored_scope_to_list


def test_convert_ignored_scope_to_list():
    ignored_names = ["name1", "name2"]
    ignored_patterns = [".*1", ".*2"]

    ignored_scope = IgnoredScope(
        names=ignored_names,
        patterns=ignored_patterns,
    )
    ignored_list = convert_ignored_scope_to_list(ignored_scope)

    assert len(ignored_list) == len(ignored_names) + len(ignored_patterns)

    for name in ignored_names:
        assert name in ignored_list
    for p in ignored_patterns:
        assert "{re}" + p in ignored_list


def test_create_ignored_scope_config_raise_exception():
    ignored_scope = IgnoredScope(types=["type1"])
    with pytest.raises(nncf.InternalError):
        _ = convert_ignored_scope_to_list(ignored_scope)
