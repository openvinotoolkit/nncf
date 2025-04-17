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

# This file deliberately does not follow naming convention for pytest-discoverable test case files
# because the cases in this file are not to be called directly within the general pytest invocation;
# these are rather executed via proxy test cases that make sure that these test cases are launched in
# separate processes each.
import subprocess

import numpy as np
import pytest
import torch

import nncf
from tests.cross_fw.shared.isolation_runner import ISOLATION_RUN_ENV_VAR


@pytest.mark.skipif(ISOLATION_RUN_ENV_VAR not in os.environ, reason="Should be run via isolation proxy")
def test_reference_quantization_on_cpu_isolated(mocker):
    load_stub = mocker.patch("torch.utils.cpp_extension.load")

    def side_effect_fn(name, *args, **kwargs):
        # simulating failure to find a compiler during extension load
        if name in ["quantized_functions_cpu"]:
            raise subprocess.CalledProcessError(returncode=-1, cmd="where cl.exe")

    load_stub.side_effect = side_effect_fn

    from nncf.torch.quantization.quantize_functions import asymmetric_quantize
    from nncf.torch.quantization.reference import ReferenceBackendType
    from nncf.torch.quantization.reference import ReferenceQuantize
    from tests.torch.quantization.test_functions import check_outputs_for_quantization_functions
    from tests.torch.quantization.test_functions import generate_input

    shape = [2, 3, 32, 32]
    scale_shape = [1, 3, 1, 1]
    ref_input_low = -np.ones(scale_shape).astype(np.float32)
    ref_input_range = 2 * np.ones_like(ref_input_low).astype(np.float32)
    ref_input = generate_input(
        shape, ref_input_low, ref_input_range, 8, "per_channel_scale", is_weights=False, middle_points=False
    ).astype(np.float32)

    level_low = -128
    level_high = 127
    levels = 256
    test_input = torch.from_numpy(ref_input).float()
    test_input_low = torch.from_numpy(ref_input_low).float()
    test_input_range = torch.from_numpy(ref_input_range).float()

    RQ = ReferenceQuantize(backend_type=ReferenceBackendType.NUMPY)
    ref_input_low, ref_input_range = RQ.tune_range(ref_input_low, ref_input_range, levels)
    ref_value = RQ.forward(ref_input, ref_input_low, ref_input_range, levels)
    test_value = asymmetric_quantize(test_input, levels, level_low, level_high, test_input_low, test_input_range, eps=0)
    check_outputs_for_quantization_functions(test_value, ref_value, rtol=1e-3)


def _parametrized_missing_cuda_test_body(mocker, exception):
    load_stub = mocker.patch("torch.utils.cpp_extension.load")

    def side_effect_fn(name, *args, **kwargs):
        # simulating failure to find a compiler during extension load
        if name == "quantized_functions_cuda":
            raise exception

    load_stub.side_effect = side_effect_fn

    patched_fn = mocker.patch("torch.cuda.is_available")
    patched_fn.return_value = True
    # loading should trigger an exception and a message to the user
    with pytest.raises(nncf.InstallationError) as exc_info:
        from nncf.torch.quantization.extensions import QuantizedFunctionsCUDALoader

        QuantizedFunctionsCUDALoader.load()
    print(str(exc_info.getrepr()))


@pytest.mark.skipif(ISOLATION_RUN_ENV_VAR not in os.environ, reason="Should be run via isolation proxy")
def test_missing_cuda_compiler_fails_with_message_isolated_calledprocesserror(mocker):
    # simulates compiler not found
    _parametrized_missing_cuda_test_body(mocker, subprocess.CalledProcessError(returncode=-1, cmd="which nvcc"))


@pytest.mark.skipif(ISOLATION_RUN_ENV_VAR not in os.environ, reason="Should be run via isolation proxy")
def test_missing_cuda_compiler_fails_with_message_isolated_oserror(mocker):
    # simulates CUDA_HOME not set
    _parametrized_missing_cuda_test_body(mocker, OSError)
