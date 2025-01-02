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

import os

from tests.cross_fw.shared.paths import TEST_ROOT


def pytest_addoption(parser):
    parser.addoption(
        "--model-dir",
        type=str,
        default=None,
        help="[e2e-test-onnx-model-zoo] Directory path to OMZ validation datasets",
    )
    parser.addoption(
        "--model-names",
        type=str,
        default=None,
        help="[e2e-test-onnx-model-zoo] String containing model names to test. "
        "Please, provide the model names using ' ' as a separator.",
    )
    parser.addoption(
        "--data-dir",
        type=str,
        default=None,
        help="[e2e-test-onnx-model-zoo] Directory path to OMZ validation datasets (Only for E2E)",
    )
    parser.addoption(
        "--output-dir", type=str, default=None, help="[e2e-test-onnx-model-zoo] Directory path to archieve outputs"
    )
    parser.addoption(
        "--anno-dir",
        type=str,
        default=None,
        help="[e2e-test-onnx-model-zoo] (Optional) Directory path for dataset annotations "
        "If it is not provided, tempfile.TemporaryDirectory will be used.",
    )
    parser.addoption(
        "--ptq-size", type=int, default=100, help="[e2e-test-onnx-model-zoo] Dataset subsample size for PTQ calibration"
    )
    parser.addoption(
        "--eval-size",
        type=int,
        default=None,
        help="[e2e-test-onnx-model-zoo] Dataset subsample size for evaluation. "
        "If not provided, full dataset is used for evaluation.",
    )
    parser.addoption(
        "--enable-ov-ep",
        action="store_true",
        default=False,
        help="[e2e-test-onnx-model-zoo] If the parameter is set then the accuracy validation of the quantized models "
        "will be enabled for ONNXRuntime-OpenVINOExecutionProvider.",
    )
    parser.addoption(
        "--enable-cpu-ep",
        action="store_true",
        default=False,
        help="[e2e-test-onnx-model-zoo] If the parameter is set then the accuracy validation of the quantized models "
        "will be enabled for ONNXRuntime-CPUExecutionProvider.",
    )
    parser.addoption(
        "--disable-ov",
        action="store_true",
        default=False,
        help="[e2e-test-onnx-model-zoo] If the parameter is set then the accuracy validation of the quantized models "
        "will be disable for OpenVINO.",
    )
    parser.addoption(
        "--regen-dot",
        action="store_true",
        default=False,
        help="If specified, the "
        "reference .dot files will be regenerated "
        "using the current state of the repository.",
    )
    parser.addoption("--data", type=str, default=None, help="Directory path to cached data.")


def pytest_configure(config):
    regen_dot = config.getoption("--regen-dot", False)
    if regen_dot:
        os.environ["NNCF_TEST_REGEN_DOT"] = "1"


ONNX_TEST_ROOT = TEST_ROOT / "onnx"
ONNX_MODEL_DIR = ONNX_TEST_ROOT / "data" / "models"
