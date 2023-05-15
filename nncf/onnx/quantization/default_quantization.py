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

from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.onnx.graph.metatypes import onnx_metatypes

DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT = {
    QuantizationTrait.INPUTS_QUANTIZABLE: [
        onnx_metatypes.ONNXConvolutionMetatype,
        onnx_metatypes.ONNXDepthwiseConvolutionMetatype,
        onnx_metatypes.ONNXConvolutionTransposeMetatype,
        onnx_metatypes.ONNXLinearMetatype,
        onnx_metatypes.ONNXMatMulMetatype,
        onnx_metatypes.ONNXAveragePoolMetatype,
        onnx_metatypes.ONNXGlobalAveragePoolMetatype,
        onnx_metatypes.ONNXAddLayerMetatype,
        onnx_metatypes.ONNXSubMetatype,
        onnx_metatypes.ONNXMulLayerMetatype,
        onnx_metatypes.ONNXBatchNormMetatype,
        onnx_metatypes.ONNXHardSigmoidMetatype,
        onnx_metatypes.ONNXResizeMetatype,
        onnx_metatypes.ONNXPowMetatype,
        onnx_metatypes.ONNXReciprocalMetatype,
    ],
    QuantizationTrait.NON_QUANTIZABLE: [
        onnx_metatypes.ONNXSigmoidMetatype,
        onnx_metatypes.ONNXSoftmaxMetatype,
        onnx_metatypes.ONNXQuantizeLinearMetatype,
        onnx_metatypes.ONNXDequantizeLinearMetatype,
        onnx_metatypes.ONNXDeformableConvolutionMetatype,
        UnknownMetatype,
        # Ticket: 108478
        onnx_metatypes.ONNXReluMetatype,
        onnx_metatypes.ONNXExpMetatype,
        onnx_metatypes.ONNXLogMetatype,
        onnx_metatypes.ONNXAbsMetatype,
        onnx_metatypes.ONNXSqrtMetatype,
    ],
    QuantizationTrait.CONCAT: [onnx_metatypes.ONNXConcatLayerMetatype],
    QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS: [],
}
