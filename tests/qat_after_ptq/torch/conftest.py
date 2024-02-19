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

from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--data", action="store", help="Path to omz training datasets director.")
    parser.addoption("--weights", action="store", help="Directory with nncf checkpoints for baseline checkopints.")


@pytest.fixture(scope="session", name="data_dir")
def fixture_data(pytestconfig):
    if pytestconfig.getoption("data") is None:
        raise ValueError("This test requires the --data argument to be specified.")
    return Path(pytestconfig.getoption("data"))


@pytest.fixture(scope="session", name="weights_dir")
def fixture_weights_dir(pytestconfig):
    if pytestconfig.getoption("weights") is None:
        raise ValueError("This test requires the --weights argument to be specified.")
    return Path(pytestconfig.getoption("weights"))
