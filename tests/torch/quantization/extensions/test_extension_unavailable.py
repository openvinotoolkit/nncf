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

import pytest

from tests.cross_fw.shared.isolation_runner import run_pytest_case_function_in_separate_process
from tests.torch.quantization.extensions.isolated_cases import (
    test_missing_cuda_compiler_fails_with_message_isolated_calledprocesserror,
)
from tests.torch.quantization.extensions.isolated_cases import (
    test_missing_cuda_compiler_fails_with_message_isolated_oserror,
)
from tests.torch.quantization.extensions.isolated_cases import test_reference_quantization_on_cpu_isolated


def test_reference_quantization_on_cpu():
    _, stdout, _ = run_pytest_case_function_in_separate_process(test_reference_quantization_on_cpu_isolated)
    print(stdout)
    assert "Could not compile CPU quantization extensions." in stdout


@pytest.mark.parametrize(
    "case",
    [
        test_missing_cuda_compiler_fails_with_message_isolated_calledprocesserror,
        test_missing_cuda_compiler_fails_with_message_isolated_oserror,
    ],
)
def test_missing_cuda_compiler_fails_with_message(case):
    _, stdout, _ = run_pytest_case_function_in_separate_process(case)
    print(stdout)
    assert "https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html" in stdout
