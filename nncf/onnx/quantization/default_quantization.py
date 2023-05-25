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

from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.onnx.graph.metatypes import onnx_metatypes

# If a metatype is not in this list, then it is considered to be QuantizationTrait.NON_QUANTIZABLE.

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
    QuantizationTrait.QUANTIZATION_AGNOSTIC: [
        onnx_metatypes.ONNXMaxPoolMetatype,
        onnx_metatypes.ONNXReduceMaxMetatype,
        onnx_metatypes.ONNXReshapeMetatype,
        onnx_metatypes.ONNXTransposeMetatype,
        onnx_metatypes.ONNXSqueezeMetatype,
        onnx_metatypes.ONNXUnsqueezeMetatype,
        onnx_metatypes.ONNXSplitMetatype,
        onnx_metatypes.ONNXTileMetatype,
        onnx_metatypes.ONNXCenterCropPadMetatype,
        onnx_metatypes.ONNXSliceMetatype,
        onnx_metatypes.ONNXPadMetatype,
        onnx_metatypes.ONNXGatherMetatype,
        onnx_metatypes.ONNXGatherNDMetatype,
        onnx_metatypes.ONNXGatherElementsMetatype,
        onnx_metatypes.ONNXDepthToSpaceMetatype,
        onnx_metatypes.ONNXSpaceToDepthMetatype,
        onnx_metatypes.ONNXScatterElementsMetatype,
        onnx_metatypes.ONNXScatterNDMetatype,
        onnx_metatypes.ONNXScatterMetatype,
        onnx_metatypes.ONNXCastLikeMetatype,
        onnx_metatypes.ONNXDropoutMetatype,
        onnx_metatypes.ONNXFlattenMetatype,
        onnx_metatypes.ONNXExpandMetatype,
        onnx_metatypes.ONNXIdentityMetatype,
        # ONNXReluMetatype is not considered to be QUANTIZATION_AGNOSTIC, because:
        # 1. Runtime doesn't provide performance benefits by quantizing the stand-alone RELU's (ticket: 59548)
        # 2. It's frequently better for the end accuracy to have quantizers set up after the RELU
        # so that the input distribution to the quantizer is non-negative
        # and we can therefore have better quantization resolution while preserving the original dynamic range
    ],
    QuantizationTrait.CONCAT: [onnx_metatypes.ONNXConcatMetatype],
    QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS: [],
}
