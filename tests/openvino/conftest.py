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

#pylint:disable=unused-import
import pytest
from tests.shared.install_fixtures import tmp_venv_with_nncf
from tests.shared.case_collection import COMMON_SCOPE_MARKS_VS_OPTIONS
from tests.shared.case_collection import skip_marked_cases_if_options_not_specified


def pytest_addoption(parser):
    pass


# Custom markers specifying tests to be run only if a specific option
# is present on the pytest command line must be registered here.
MARKS_VS_OPTIONS = {
    **COMMON_SCOPE_MARKS_VS_OPTIONS
}

def pytest_collection_modifyitems(config, items):
    skip_marked_cases_if_options_not_specified(config, items, MARKS_VS_OPTIONS)
