# Copyright (c) 2023 Intel Corporation
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

from nncf.openvino.pot.quantization.quantize_model import _create_ignored_scope_config
from nncf.scopes import IgnoredScope


def test_create_ignored_scope_config():
    ignored_names = ["name1", "name2"]
    ignored_types = ["type1", "type2"]

    ignored_scope = IgnoredScope(
        names=ignored_names,
        types=ignored_types,
    )
    ignored_config = _create_ignored_scope_config(ignored_scope)

    assert ignored_config["scope"] == ignored_names

    actual_types = [a["type"] for a in ignored_config["operations"]]
    assert actual_types.sort() == ignored_types.sort()


def test_create_ignored_scope_config_raise_exception():
    ignored_scope = IgnoredScope(patterns=[".*"])
    with pytest.raises(Exception):
        _ = _create_ignored_scope_config(ignored_scope)
