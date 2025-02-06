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

TESTED_BACKENDS = ["torch", "tf", "onnx", "openvino"]


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        type=str,
        help="Backend to test installation for.",
        choices=[*TESTED_BACKENDS, "all"],
        nargs="+",
        default=["all"],
    )
    parser.addoption(
        "--check_performance",
        default=False,
        help="If the parameter is set then the performance metrics will be tested as well",
    )
    parser.addoption(
        "--ov_version_override", default=None, help="Parameter to set OpenVINO into the env with the version from PyPI"
    )
    parser.addoption("--data", type=str, default=None, help="Path to test datasets")
    parser.addoption("--reuse-venv", action="store_true", help="Use venv from example directory")


@pytest.fixture(scope="module")
def backends_list(request):
    return request.config.getoption("--backend")


@pytest.fixture(scope="module")
def is_check_performance(request):
    return request.config.getoption("--check_performance")


@pytest.fixture(scope="module")
def ov_version_override(request):
    return request.config.getoption("--ov_version_override")


@pytest.fixture(scope="module")
def data(request):
    return request.config.getoption("--data")


@pytest.fixture(scope="module")
def reuse_venv(request) -> bool:
    return request.config.getoption("--reuse-venv")
