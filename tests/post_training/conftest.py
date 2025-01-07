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
