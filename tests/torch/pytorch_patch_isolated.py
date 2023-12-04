# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os

import pytest
import torch

from tests.shared.isolation_runner import ISOLATION_RUN_ENV_VAR


def remove_comments_from_source(source):
    lines = source.split("\n")
    processed_lines = []
    for line in lines:
        hash_position = line.find("#")
        if hash_position != -1:
            line = line[:hash_position]
        line = line.rstrip()
        if len(line) > 0:
            processed_lines.append(line)
    processed_source = "\n".join(processed_lines)
    return processed_source


@pytest.mark.skipif(ISOLATION_RUN_ENV_VAR not in os.environ, reason="Should be run via isolation proxy")
def test_jit_if_tracing_script_source_equals():
    # Get original torch.jit._script_if_tracing source
    torch_source = remove_comments_from_source(inspect.getsource(torch.jit._script_if_tracing))

    import nncf.torch  # noqa: F401

    # Get torch.jit._script_if_tracing source after patching was performed
    nncf_source = remove_comments_from_source(inspect.getsource(torch.jit._script_if_tracing))

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

    try:
        torch.jit.script(compressed_model)  # supposed to fail since torch.jit.script does not support NNCF models
    except:  # noqa: E722
        pass

    # torch.nn.Module.__call__ is one of the fundamental patched functions, if the code object points to NNCF code,
    # then it means patching is still present
    assert "nncf" in torch.nn.Module.__call__.__code__.co_filename
