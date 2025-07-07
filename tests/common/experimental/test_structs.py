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

import nncf
from nncf.experimental.quantization.structs import ExtendedQuantizerConfig
from nncf.tensor.definitions import TensorDataType


@pytest.mark.parametrize(
    "dest_dtype",
    [
        TensorDataType.float16,
        TensorDataType.bfloat16,
        TensorDataType.float32,
        TensorDataType.float64,
        TensorDataType.f8e4m3,
        TensorDataType.f8e5m2,
        TensorDataType.nf4,
        TensorDataType.int32,
        TensorDataType.int64,
        TensorDataType.uint4,
        TensorDataType.int4,
        None,
    ],
)
def test_extended_q_config_non_supported_dest_dtype(dest_dtype):
    with pytest.raises(nncf.ParameterNotSupportedError):
        ExtendedQuantizerConfig(dest_dtype=dest_dtype)
