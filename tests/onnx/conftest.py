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
import os

from tests.shared.paths import TEST_ROOT


def pytest_addoption(parser):
    parser.addoption(
        "--model-dir", type=str, default=None,
        help="[e2e-test-onnx-model-zoo] Directory path to OMZ validation datasets"
    )
    parser.addoption(
        "--data-dir", type=str, default=None,
        help="[e2e-test-onnx-model-zoo] Directory path to OMZ validation datasets"
    )
    parser.addoption(
        "--output-dir", type=str, default=None,
        help="[e2e-test-onnx-model-zoo] Directory path to archieve outputs"
    )
    parser.addoption(
        "--ckpt-dir", type=str, default=None,
        help="[e2e-test-onnx-model-zoo] (Optional) Directory path to save quantized models. "
        "If it is not provided, tempfile.TemporaryDirectory will be used."
    )
    parser.addoption(
        "--anno-dir", type=str, default=None,
        help="[e2e-test-onnx-model-zoo] (Optional) Directory path for dataset annotations "
        "If it is not provided, tempfile.TemporaryDirectory will be used."
    )
    parser.addoption(
        "--ptq-size", type=int, default=100,
        help="[e2e-test-onnx-model-zoo] Dataset subsample size for PTQ calibration"
    )
    parser.addoption(
        "--eval-size", type=int, default=None,
        help="[e2e-test-onnx-model-zoo] Dataset subsample size for evaluation. "
        "If not provided, full dataset is used for evaluation."
    )
    parser.addoption(
        "--regen-dot", action="store_true", default=False, help="If specified, the "
                                                                "reference .dot files will be regenerated "
                                                                "using the current state of the repository."

    )

def pytest_configure(config):
    regen_dot = config.getoption('--regen-dot', False)
    if regen_dot:
        os.environ["NNCF_TEST_REGEN_DOT"] = "1"


ONNX_TEST_ROOT = TEST_ROOT / 'onnx'
ONNX_MODEL_DIR = ONNX_TEST_ROOT / 'data' / 'models'
