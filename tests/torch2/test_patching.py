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
import torch

import nncf
from nncf.torch import create_compressed_model


def test_patching():
    # Check that patching torch functions is disabled
    import nncf.torch  # noqa: F401

    with pytest.raises(AttributeError):
        getattr(torch.relu, "_original_op")


def test_create_compressed_model_error():
    with pytest.raises(nncf.InternalError, match="NNCF_TORCH_LEGACY_TRACING=1"):
        create_compressed_model(None, {})
