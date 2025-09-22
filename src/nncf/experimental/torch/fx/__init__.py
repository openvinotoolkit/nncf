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

from nncf.experimental.torch.fx.quantization.quantize_pt2e import quantize_pt2e as quantize_pt2e
from nncf.experimental.torch.fx.quantization.quantize_pt2e import compress_pt2e as compress_pt2e
from nncf.experimental.torch.fx.quantization.quantizer.openvino_quantizer import OpenVINOQuantizer as OpenVINOQuantizer
