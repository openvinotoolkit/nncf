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

import contextlib
import inspect
import os
import re

import pytest
import torch

from tests.cross_fw.shared.isolation_runner import ISOLATION_RUN_ENV_VAR


def clean_source_code(code_source):
    # clean source code from comments and annotation
    patterns = [
        r"\s*#.*",
        r": Callable\[P, R\]",
        r" -> Callable\[P, R\]",
        r": P.args",
        r": P.kwargs",
        r" -> R",
    ]
    for pattern in patterns:
        code_source = re.sub(pattern, "", code_source)
    # remove empty lines
    code_source = re.sub(r"\n\s*\n", "\n", code_source, flags=re.MULTILINE)
    return code_source


@pytest.mark.skipif(ISOLATION_RUN_ENV_VAR not in os.environ, reason="Should be run via isolation proxy")
def test_jit_if_tracing_script_source_equals():
    # Get original torch.jit._script_if_tracing source
    torch_source = clean_source_code(inspect.getsource(torch.jit._script_if_tracing))

    import nncf.torch  # noqa: F401

    # Get torch.jit._script_if_tracing source after patching was performed
    nncf_source = clean_source_code(inspect.getsource(torch.jit._script_if_tracing))

    # Check that the two versions are essentially the same
    nncf_source_corrected = nncf_source.replace("def torch_jit_script_if_tracing", "def _script_if_tracing").replace(
        "torch.jit.script", "script"
    )
    assert torch_source == nncf_source_corrected


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return x


@pytest.mark.skipif(ISOLATION_RUN_ENV_VAR not in os.environ, reason="Should be run via isolation proxy")
def test_jit_script_exception_preserves_patching_isolated():
    from nncf import NNCFConfig
    from nncf.torch import create_compressed_model

    _, compressed_model = create_compressed_model(
        DummyModel(),
        NNCFConfig.from_dict(
            {"input_info": {"sample_size": [1, 3, 32, 32]}, "compression": {"algorithm": "quantization"}}
        ),
    )

    with contextlib.suppress(Exception):
        torch.jit.script(compressed_model)  # supposed to fail since torch.jit.script does not support NNCF models

    # torch.nn.Module.__call__ is one of the fundamental patched functions, if the code object points to NNCF code,
    # then it means patching is still present
    assert "nncf" in torch.nn.Module.__call__.__code__.co_filename


def compile_and_run_test_model(compile_forward: bool) -> torch.Tensor:
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            return self.conv(x)

    model = TestModel()

    torch.manual_seed(0)
    state_dict = {}
    for k, v in model.state_dict().items():
        state_dict[k] = torch.rand(v.shape)
    model.load_state_dict(state_dict)

    if compile_forward:
        compiled_model = model
        compiled_model.forward = torch.compile(model.forward)
    else:
        compiled_model = torch.compile(model)
    assert "_torchdynamo_orig_callable" in compiled_model.forward.__dict__
    return compiled_model(torch.rand([1, 3, 5, 5]))


@pytest.mark.skipif(ISOLATION_RUN_ENV_VAR not in os.environ, reason="Should be run via isolation proxy")
def test_compile():
    compile_forward = os.environ.get("COMPILE_FORWARD", None) == "1"
    before_nncf = compile_and_run_test_model(compile_forward)
    import nncf.torch  # noqa: F401

    after_nncf = compile_and_run_test_model(compile_forward)
    assert torch.allclose(before_nncf, after_nncf)
