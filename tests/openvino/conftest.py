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
from tests.common.helpers import create_venv_with_nncf


def pytest_addoption(parser):
    parser.addoption(
        "--run-install-tests", action="store_true", default=False, help="To run installation tests"
    )


@pytest.fixture(scope="module")
def install_tests(request):
    return request.config.getoption("--run-install-tests")


@pytest.fixture(scope="function")
def tmp_venv_with_nncf(tmp_path, package_type, venv_type, install_tests):  # pylint:disable=redefined-outer-name
    if not install_tests:
        pytest.skip('To test the installation, use --run-install-tests option.')
    venv_path = create_venv_with_nncf(tmp_path, package_type, venv_type, extra_reqs='openvino')
    return venv_path
