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

from nncf.common.tensor_statistics.statistics_validator import validate_backend
from nncf.common.utils.backend import BackendType


@pytest.mark.parametrize("backend_value", [BackendType.TORCH, BackendType.ONNX])
def test_validate_backend(backend_value):
    # Test case where backend matches
    data = {"backend": backend_value.value}
    backend = backend_value

    validate_backend(data, backend)

    with pytest.raises(ValueError) as exc_info:
        # Test case where backend does not match
        validate_backend({"backend": BackendType.ONNX.value}, BackendType.TORCH)
    assert "Backend in loaded statistics" in str(exc_info)

    with pytest.raises(ValueError) as exc_info:
        # Test case where backend key is missing
        validate_backend({}, BackendType.TORCH)
    assert "The provided metadata has no information about backend." in str(exc_info)
