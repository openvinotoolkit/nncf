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

IGNORED_NAMES = [['name1', 'name2'], 'name1']
IGNORED_TYPES = [['type1', 'type2'], 'type1']


@pytest.mark.parametrize(('ignored_names, ignored_types'),
                         zip(IGNORED_NAMES, IGNORED_TYPES),
                         ids=['list', 'str'])
def test_create_ignored_scope_config(ignored_names, ignored_types):
    ignored_scope = IgnoredScope(
        names=ignored_names,
        types=ignored_types,
    )
    ignored_config = _create_ignored_scope_config(ignored_scope)

    expected_names = ignored_names
    if isinstance(expected_names, str):
        expected_names = [expected_names]

    assert ignored_config['scope'] == expected_names

    expected_types = ignored_types
    if isinstance(expected_types, str):
        expected_types = [expected_types]

    actual_types = [a['type'] for a in ignored_config['operations']]
    actual_types = actual_types.sort()
    expected_types = expected_types.sort()
    assert actual_types == expected_types


def test_create_ignored_scope_config_raise_exception():
    ignored_scope = IgnoredScope(
        patterns='.*'
    )
    with pytest.raises(Exception):
        _ = _create_ignored_scope_config(ignored_scope)
