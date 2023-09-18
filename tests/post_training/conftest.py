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

import collections
from pathlib import Path

import pandas as pd
import pytest

from tests.shared.paths import TEST_ROOT

PTQ_TEST_ROOT = TEST_ROOT / "post_training"
FQ_CALCULATED_PARAMETERS_PATH = PTQ_TEST_ROOT / "data" / "fq_params" / "fq_params.json"


def pytest_addoption(parser):
    parser.addoption("--data", action="store")
    parser.addoption("--output", action="store", default="./tmp/")


def pytest_configure(config):
    config.test_results = {}


PTQ_TEST_ROOT = TEST_ROOT / "post_training_quantization"


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    result = outcome.get_result()

    if result.when == "call":
        test_results = collections.OrderedDict(sorted(item.config.test_results.items()))
        df = pd.DataFrame()
        for _, test_result in test_results.items():
            df = df.append(test_result, ignore_index=True)

        output_folder = Path(item.config.getoption("--output"))
        output_folder.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_folder / "results.csv", index=False)
