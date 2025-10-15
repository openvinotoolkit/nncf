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

from pathlib import Path

import pytest

from tests.cross_fw.shared.case_collection import COMMON_SCOPE_MARKS_VS_OPTIONS
from tests.cross_fw.shared.case_collection import skip_marked_cases_if_options_not_specified
from tests.cross_fw.shared.install_fixtures import tmp_venv_with_nncf  # noqa: F401
from tests.cross_fw.shared.paths import TEST_ROOT


def pytest_addoption(parser):
    parser.addoption("--data", type=str, default=None, help="Directory path to cached data.")


@pytest.fixture(name="data_dir")
def data(request):
    option = request.config.getoption("--data")
    if option is None:
        return Path(DATASET_PATH)
    return Path(option)


@pytest.fixture(name="omz_cache_dir")
def models(request):
    option = request.config.getoption("--data")
    if option is None:
        return Path(MODELS_PATH)
    return Path(option)


@pytest.fixture(name="zero_seed")
def _seed():
    """
    Fixture to ensure deterministic randomness across tests.
    """
    import random

    import numpy as np

    np.random.seed(0)
    random.seed(0)


# Custom markers specifying tests to be run only if a specific option
# is present on the pytest command line must be registered here.
MARKS_VS_OPTIONS = {**COMMON_SCOPE_MARKS_VS_OPTIONS}


def pytest_collection_modifyitems(config, items):
    skip_marked_cases_if_options_not_specified(config, items, MARKS_VS_OPTIONS)


OPENVINO_TEST_ROOT = TEST_ROOT / "openvino"
OPENVINO_POT_TEST_ROOT = OPENVINO_TEST_ROOT / "pot"
OPENVINO_NATIVE_TEST_ROOT = OPENVINO_TEST_ROOT / "native"
AC_CONFIGS_DIR = OPENVINO_TEST_ROOT / "data" / "ac_configs"
OPENVINO_DATASET_DEFINITIONS_PATH = OPENVINO_TEST_ROOT / "data" / "ov_dataset_definitions.yml"
NNCF_CACHE_PATH = Path("~/.cache/nncf")
DATASET_PATH = NNCF_CACHE_PATH / "datasets"
MODELS_PATH = NNCF_CACHE_PATH / "models"
