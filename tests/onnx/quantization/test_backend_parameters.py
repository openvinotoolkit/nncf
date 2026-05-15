# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nncf.onnx.quantization.backend_parameters import BackendParameters
from nncf.onnx.quantization.backend_parameters import get_external_data_dir
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters


def test_get_external_data_dir():
    assert get_external_data_dir(None) is None
    assert get_external_data_dir(AdvancedQuantizationParameters()) is None
    assert get_external_data_dir(AdvancedCompressionParameters()) is None
    assert get_external_data_dir(AdvancedQuantizationParameters(backend_params={})) is None
    assert get_external_data_dir(AdvancedCompressionParameters(backend_params={})) is None
    assert (
        get_external_data_dir(
            AdvancedQuantizationParameters(backend_params={BackendParameters.EXTERNAL_DATA_DIR: "path"})
        )
        == "path"
    )
    assert (
        get_external_data_dir(
            AdvancedCompressionParameters(backend_params={BackendParameters.EXTERNAL_DATA_DIR: "path"})
        )
        == "path"
    )
