"""
 Copyright (c) 2023 Intel Corporation
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
import numpy as np
from pathlib import Path

def pytest_addoption(parser):
    parser.addoption("--data", action="store")
    parser.addoption("--output", action="store", default="./tmp/")

def pytest_configure(config):
    config.test_results = []

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    result = outcome.get_result()

    if result.when == 'call':
        table = item.config.test_results
        header = ["Model name", "FP32 top-1", "Torch INT8 top-1", "ONNX INT8 top-1", "OV Native INT8 top-1", "OV INT8 top-1",
            "FP32 FPS", "Torch INT8 FPS", "ONNX INT8 FPS", "OV Native FPS", "OV INT8 FPS"]
        output_folder = Path(item.config.getoption("--output"))
        output_folder.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_folder / "results.csv", table, delimiter=",", fmt='%s', header=','.join(header))
