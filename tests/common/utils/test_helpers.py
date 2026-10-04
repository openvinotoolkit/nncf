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

import os

from nncf.common.utils.helpers import create_table
from nncf.common.utils.helpers import set_env_variable


def test_create_table_ascii():
    table_fmt = "grid"
    table = create_table(["Name", "Ratio"], [["layer", 0.125], ["total", 1.0]], table_fmt=table_fmt)
    assert table == (
        "+---------------+\n"
        "| Name  | Ratio |\n"
        "|-------+-------|\n"
        "| layer | 0.125 |\n"
        "|-------+-------|\n"
        "| total | 1.000 |\n"
        "+---------------+"
    )


def test_set_env_variable():
    # Test the case when the variable is not set
    assert os.environ.get("TEST_VAR") is None
    with set_env_variable("TEST_VAR", "test_value"):
        assert os.environ.get("TEST_VAR") == "test_value"
    assert os.environ.get("TEST_VAR") is None

    # Test the case when the variable is already set
    os.environ["TEST_VAR"] = "original_value"
    assert os.environ.get("TEST_VAR") == "original_value"
    with set_env_variable("TEST_VAR", "test_value"):
        assert os.environ.get("TEST_VAR") == "test_value"
    assert os.environ.get("TEST_VAR") == "original_value"
