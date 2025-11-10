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

import sys

import torch

import nncf
from tests.cross_fw.install.common import load_nncf_modules

if len(sys.argv) != 3:
    msg = "Must be run with an execution type as argument (either 'cpu' or 'gpu') and package type"
    raise nncf.ValidationError(msg)
execution_type = sys.argv[1]
package_type = sys.argv[2]

# Do not remove - these imports are for testing purposes.


import nncf  # noqa: F401, E402
from nncf.torch import create_compressed_model  # noqa: F401, E402

input_low_tensor = torch.zeros([1])
input_tensor = torch.ones([1, 1, 1, 1])
input_high_tensor = torch.ones([1])
scale_tensor = torch.ones([1])
threshold_tensor = torch.zeros([1, 1, 1, 1])
levels = 256

if execution_type == "cpu":
    from nncf.torch.quantization.extensions import QuantizedFunctionsCPU

    output_tensor = QuantizedFunctionsCPU.get("Quantize_forward")(
        input_tensor, input_low_tensor, input_high_tensor, levels
    )
elif execution_type == "gpu":
    input_tensor = input_tensor.cuda()
    input_low_tensor = input_low_tensor.cuda()
    input_high_tensor = input_high_tensor.cuda()
    scale_tensor = scale_tensor.cuda()
    threshold_tensor = threshold_tensor.cuda()
    from nncf.torch.quantization.extensions import QuantizedFunctionsCUDA

    output_tensor = QuantizedFunctionsCUDA.get("Quantize_forward")(
        input_tensor, input_low_tensor, input_high_tensor, levels
    )
else:
    msg = f"Invalid execution type {execution_type} (expected 'cpu' or 'gpu')!"
    raise nncf.ValidationError(msg)

EXCLUDED_MODULES_PATTERNS = (
    "nncf\\.openvino.*",
    "nncf\\.onnx.*",
    "nncf\\.experimental\\.openvino.*",
    "nncf\\.experimental\\.onnx.*",
    "^(?!nncf(?:\\.experimental)*\\.torch.*?\\.).*?openvino_[^\\.]*",
    "^(?!nncf(?:\\.experimental)*\\.torch.*?\\.).*?onnx_[^\\.]*",
    "^(?!nncf(?:\\.experimental)*\\.torch.*?\\.).*?tf_[^\\.]*",
)

load_nncf_modules(EXCLUDED_MODULES_PATTERNS)
