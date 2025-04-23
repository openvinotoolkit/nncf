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


def pytest_addoption(parser):
    parser.addoption("--data", action="store", help="Data directory")
    parser.addoption("--output", action="store", default="./tmp/", help="Directory to store artifacts")
    parser.addoption("--no-eval", action="store_true", help="Skip validation step")
    parser.addoption("--batch-size", action="store", default=None, type=int, help="Batch size of calibration dataset")
    parser.addoption("--subset-size", type=int, default=None, help="Set subset size")
    parser.addoption("--fp32", action="store_true", help="Test original model")
    parser.addoption("--cuda", action="store_true", help="Enable CUDA_TORCH backend")
    parser.addoption("--benchmark", action="store_true", help="Run benchmark_app")
    parser.addoption(
        "--torch-compile-validation",
        action="store_true",
        help='Validate TorchFX quantized models via torch.compile(..., backend="openvino")',
    )
    parser.addoption(
        "--extra-columns",
        action="store_true",
        help="Add additional columns to reports.csv",
    )
    parser.addoption(
        "--memory-monitor",
        action="store_true",
        help="Report memory using MemoryMonitor from tools/memory_monitor.py. "
        "Warning: currently, reported memory values are not always reproducible.",
    )


@pytest.fixture(scope="session", name="data_dir")
def fixture_data(pytestconfig):
    if pytestconfig.getoption("data") is None:
        msg = "This test requires the --data argument to be specified."
        raise ValueError(msg)
    return Path(pytestconfig.getoption("data"))


@pytest.fixture(scope="session", name="output_dir")
def fixture_output(pytestconfig):
    return Path(pytestconfig.getoption("output"))


@pytest.fixture(scope="session", name="no_eval")
def fixture_no_eval(pytestconfig):
    return pytestconfig.getoption("no_eval")


@pytest.fixture(scope="session", name="batch_size")
def fixture_batch_size(pytestconfig):
    return pytestconfig.getoption("batch_size")


@pytest.fixture(scope="session", name="subset_size")
def fixture_subset_size(pytestconfig):
    return pytestconfig.getoption("subset_size")


@pytest.fixture(scope="session", name="run_fp32_backend")
def fixture_run_fp32_backend(pytestconfig):
    return pytestconfig.getoption("fp32")


@pytest.fixture(scope="session", name="run_torch_cuda_backend")
def fixture_run_torch_cuda_backend(pytestconfig):
    return pytestconfig.getoption("cuda")


@pytest.fixture(scope="session", name="run_benchmark_app")
def fixture_run_benchmark_app(pytestconfig):
    return pytestconfig.getoption("benchmark")


@pytest.fixture(scope="session", name="extra_columns")
def fixture_extra_columns(pytestconfig):
    return pytestconfig.getoption("extra_columns")


@pytest.fixture(scope="session", name="memory_monitor")
def fixture_memory_monitor(pytestconfig):
    return pytestconfig.getoption("memory_monitor")


@pytest.fixture(scope="session", name="forked")
def fixture_forked(pytestconfig):
    return pytestconfig.getoption("forked")
