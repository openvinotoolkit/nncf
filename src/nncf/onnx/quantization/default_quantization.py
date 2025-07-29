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

from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.onnx.graph.metatypes import onnx_metatypes
from nncf.onnx.graph.metatypes.groups import INPUTS_QUANTIZABLE_OPERATIONS
from nncf.onnx.graph.metatypes.groups import QUANTIZE_AGNOSTIC_OPERATIONS

# If a metatype is not in this list, then it is considered to be QuantizationTrait.NON_QUANTIZABLE.

DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT = {
    QuantizationTrait.INPUTS_QUANTIZABLE: INPUTS_QUANTIZABLE_OPERATIONS,
    QuantizationTrait.QUANTIZATION_AGNOSTIC: QUANTIZE_AGNOSTIC_OPERATIONS,
    QuantizationTrait.CONCAT: [onnx_metatypes.ONNXConcatMetatype],
    QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS: [onnx_metatypes.ONNXEmbeddingMetatype],
}
