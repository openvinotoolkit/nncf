"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import pytest

from nncf.parameters import IgnoredScope
from nncf.openvino.quantization.helpers import _create_ignored_scope_config

IGNORED_NODE_NAMES = [['name1', 'name2'], 'name1']
IGNORED_NODE_TYPES = [['type1', 'type2'], 'type1']


@pytest.mark.parametrize(('ignored_node_names, ignored_node_types'),
                         zip(IGNORED_NODE_NAMES, IGNORED_NODE_TYPES),
                         ids=['list', 'str'])
def test_create_ignored_scope_config(ignored_node_names, ignored_node_types):
    ignored_scope = IgnoredScope(
        node_names=ignored_node_names,
        node_types=ignored_node_types,
    )
    ignored_config = _create_ignored_scope_config(ignored_scope)

    expected_node_names = ignored_node_names
    if isinstance(expected_node_names, str):
        expected_node_names = [expected_node_names]

    assert ignored_config['scope'] == expected_node_names

    expected_node_types = ignored_node_types
    if isinstance(expected_node_types, str):
        expected_node_types = [expected_node_types]

    actual_node_types = [a['type'] for a in ignored_config['operations']]
    actual_node_types = actual_node_types.sort()
    expected_node_types = expected_node_types.sort()
    assert actual_node_types == expected_node_types


def test_create_ignored_scope_config_raise_exception():
    ignored_scope = IgnoredScope(
        node_name_regexps='.*'
    )
    with pytest.raises(Exception):
        _ = _create_ignored_scope_config(ignored_scope)
