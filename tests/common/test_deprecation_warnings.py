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

from nncf.common.deprecation import _generate_deprecation_message
from nncf.common.deprecation import deprecated
from nncf.common.logging.logger import NNCFDeprecationWarning

EXAMPLE_MSG = "foo"
START_VERSION = "1.2.3"
END_VERSION = "4.5.6"


@deprecated(msg=EXAMPLE_MSG, start_version=START_VERSION, end_version=END_VERSION)
def deprecated_function():
    pass


@deprecated(msg=EXAMPLE_MSG, start_version=START_VERSION, end_version=END_VERSION)
def deprecated_function_with_versions():
    pass


@deprecated(msg=EXAMPLE_MSG)
class DeprecatedClass:
    pass


def test_warnings_are_shown_for_deprecated_function_call():
    with pytest.warns(NNCFDeprecationWarning, match=EXAMPLE_MSG):
        deprecated_function()


def test_warnings_are_shown_for_deprecated_function_call_with_versions():
    with pytest.warns(NNCFDeprecationWarning) as record:
        deprecated_function_with_versions()
        text = record[0].message.args[0]
        assert EXAMPLE_MSG in text
        assert START_VERSION in text
        assert END_VERSION in text


def test_warnings_are_shown_for_deprecated_class_instantiation():
    with pytest.warns(NNCFDeprecationWarning, match=EXAMPLE_MSG):
        DeprecatedClass()


def test_generate_deprecation_message():
    ret = _generate_deprecation_message("foo", "text", "1.2.3", "4.5.6")
    assert ret == "Usage of foo is deprecated starting from NNCF v1.2.3 and will be removed in NNCF v4.5.6.\ntext"

    ret = _generate_deprecation_message("foo", "text", "1.2.3", None)
    assert (
        ret
        == "Usage of foo is deprecated starting from NNCF v1.2.3 and will be removed in a future NNCF version.\ntext"
    )

    ret = _generate_deprecation_message("foo", "text", None, None)
    assert ret == "Usage of foo is deprecated and will be removed in a future NNCF version.\ntext"

    ret = _generate_deprecation_message("foo", "text", None, "4.5.6")
    assert ret == "Usage of foo is deprecated and will be removed in NNCF v4.5.6.\ntext"

    ret = _generate_deprecation_message("foo", None, None, None)
    assert ret == "Usage of foo is deprecated and will be removed in a future NNCF version."
