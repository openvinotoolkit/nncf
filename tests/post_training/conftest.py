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
from enum import Enum
from dataclasses import dataclass
from dataclasses import fields


def pytest_addoption(parser):
    parser.addoption("--data", action="store")
    parser.addoption("--output", action="store", default="./tmp/")
    parser.addoption("--backends", action="store", default="TORCH,TORCH_PTQ,ONNX,OV_NATIVE,OV")


def pytest_configure(config):
    config.test_results = {}


class PipelineType(Enum):
    FP32 = 'FP32'
    TORCH = 'Torch INT8'
    TORCH_PTQ = 'Torch PTQ INT8'
    ONNX = 'ONNX INT8'
    OV_NATIVE = 'OV Native INT8'
    OV = 'Openvino INT8'


@dataclass
class RunInfo:
    top_1: float
    FPS: float
    status: str


@pytest.fixture
def backends_list(request):
    return request.config.getoption('--backends')


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    result = outcome.get_result()

    if result.when == 'call':
        test_results = item.config.test_results
        header = ["Model name"]
        for info in fields(RunInfo):
            for pipeline_type in PipelineType:
                header.append(" ".join((pipeline_type.value, info.name)))

        table = []
        for model_name, run_infos in test_results.items():
            row = [model_name]
            for info in fields(RunInfo):
                for pipeline_type in PipelineType:
                    data = '-'
                    if pipeline_type in run_infos:
                        data = getattr(run_infos[pipeline_type], info.name)
                    row.append(data)
            table.append(row)

        output_folder = Path(item.config.getoption("--output"))
        output_folder.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_folder / "results.csv", table, delimiter=",", fmt='%s', header=','.join(header))
