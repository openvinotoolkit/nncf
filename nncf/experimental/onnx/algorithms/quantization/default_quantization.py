"""
 Copyright (c) 2022 Intel Corporation
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

from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionTransposeMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXLinearMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXSigmoidMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXHardSigmoidMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXAveragePoolMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXGlobalAveragePoolMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXAddLayerMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXSubMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXMulLayerMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXConcatLayerMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXBatchNormMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXResizeMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXSoftmaxMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXExpMetatype

from nncf.common.graph.operator_metatypes import UnknownMetatype

DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT = {
    QuantizationTrait.INPUTS_QUANTIZABLE: [
        ONNXConvolutionMetatype,
        ONNXConvolutionTransposeMetatype,
        ONNXLinearMetatype,
        ONNXAveragePoolMetatype,
        ONNXGlobalAveragePoolMetatype,
        ONNXAddLayerMetatype,
        ONNXSubMetatype,
        ONNXMulLayerMetatype,
        ONNXBatchNormMetatype,
        ONNXHardSigmoidMetatype,
        ONNXResizeMetatype,
    ],
    QuantizationTrait.NON_QUANTIZABLE: [ONNXSigmoidMetatype,
                                        ONNXSoftmaxMetatype,
                                        ONNXExpMetatype,
                                        UnknownMetatype],
    QuantizationTrait.CONCAT: [ONNXConcatLayerMetatype],
    QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS: []
}
