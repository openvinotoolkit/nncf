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
from pathlib import Path

import pytest
import torch

from nncf.torch.extensions import EXTENSION_LOAD_TIMEOUT_ENV_VAR
from nncf.torch.extensions import ExtensionLoaderTimeoutException
from nncf.torch.quantization.extensions import QuantizedFunctionsCPU
from nncf.torch.quantization.extensions import QuantizedFunctionsCUDA
from tests.cross_fw.shared.isolation_runner import ISOLATION_RUN_ENV_VAR
from tests.cross_fw.shared.isolation_runner import run_pytest_case_function_in_separate_process


@pytest.mark.skipif(ISOLATION_RUN_ENV_VAR not in os.environ, reason="Should be run via isolation proxy")
def test_timeout_extension_loader_isolated(tmp_path, use_cuda):
    if not torch.cuda.is_available() and use_cuda is True:
        pytest.skip("Skipping CUDA test cases for CPU only setups")

    quant_func = QuantizedFunctionsCUDA if use_cuda else QuantizedFunctionsCPU

    os.environ[EXTENSION_LOAD_TIMEOUT_ENV_VAR] = "1"
    os.environ["TORCH_EXTENSIONS_DIR"] = tmp_path.as_posix()

    build_dir = Path(quant_func._loader.get_build_dir())
    lock_file = build_dir / "lock"
    lock_file.touch()
    with pytest.raises(ExtensionLoaderTimeoutException):
        quant_func.get("Quantize_forward")


def test_timeout_extension_loader():
    run_pytest_case_function_in_separate_process(test_timeout_extension_loader_isolated)
