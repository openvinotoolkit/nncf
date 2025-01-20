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

from nncf.openvino.optimized_functions.functions import astype as astype
from nncf.openvino.optimized_functions.functions import do_int_quantization as do_int_quantization
from nncf.openvino.optimized_functions.functions import quantize_dequantize_weight as quantize_dequantize_weight
from nncf.openvino.optimized_functions.models import OVModelParameters as OVModelParameters
from nncf.openvino.optimized_functions.models import clear_ov_model_cache as clear_ov_model_cache
