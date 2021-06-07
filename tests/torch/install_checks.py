"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

# Do not remove - these imports are for testing purposes.
#pylint:disable=unused-import
import nncf

import sys

import torch

if len(sys.argv) != 3:
    raise RuntimeError("Must be run with an execution type as argument (either 'cpu' or 'cuda') and package type")
execution_type = sys.argv[1]
package_type = sys.argv[2]

if package_type == 'pip_pypi':
    try:
        from nncf.torch import create_compressed_model
    except ImportError:
        from nncf import create_compressed_model
else:
    from nncf.torch import create_compressed_model

input_low_tensor = torch.zeros([1])
input_tensor = torch.ones([1, 1, 1, 1])
input_high_tensor = torch.ones([1])
scale_tensor = torch.ones([1])
threshold_tensor = torch.zeros([1, 1, 1, 1])
levels = 256

if execution_type == "cpu":
    if package_type == 'pip_pypi':
        try:
            from nncf.torch.binarization.extensions import BinarizedFunctionsCPU
            from nncf.torch.quantization.extensions import QuantizedFunctionsCPU
        except ImportError:
            from nncf.binarization.extensions import BinarizedFunctionsCPU
            from nncf.quantization.extensions import QuantizedFunctionsCPU
    else:
        from nncf.torch.binarization.extensions import BinarizedFunctionsCPU
        from nncf.torch.quantization.extensions import QuantizedFunctionsCPU
    output_tensor = QuantizedFunctionsCPU.Quantize_forward(input_tensor, input_low_tensor, input_high_tensor, levels)
    output_tensor = BinarizedFunctionsCPU.ActivationBinarize_forward(output_tensor, scale_tensor, threshold_tensor)
    output_tensor = BinarizedFunctionsCPU.WeightBinarize_forward(output_tensor, True)
elif execution_type == "cuda":
    input_tensor = input_tensor.cuda()
    input_low_tensor = input_low_tensor.cuda()
    input_high_tensor = input_high_tensor.cuda()
    scale_tensor = scale_tensor.cuda()
    threshold_tensor = threshold_tensor.cuda()
    if package_type == 'pip_pypi':
        try:
            from nncf.torch.binarization.extensions import BinarizedFunctionsCUDA
            from nncf.torch.quantization.extensions import QuantizedFunctionsCUDA
        except ImportError:
            from nncf.binarization.extensions import BinarizedFunctionsCUDA
            from nncf.quantization.extensions import QuantizedFunctionsCUDA
    else:
        from nncf.torch.binarization.extensions import BinarizedFunctionsCUDA
        from nncf.torch.quantization.extensions import QuantizedFunctionsCUDA
    output_tensor = QuantizedFunctionsCUDA.Quantize_forward(input_tensor, input_low_tensor, input_high_tensor, levels)
    output_tensor = BinarizedFunctionsCUDA.ActivationBinarize_forward(output_tensor, scale_tensor, threshold_tensor)
    output_tensor = BinarizedFunctionsCUDA.WeightBinarize_forward(output_tensor, True)
else:
    raise RuntimeError("Invalid execution type!")
