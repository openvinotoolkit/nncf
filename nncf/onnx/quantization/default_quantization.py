"""
 Copyright (c) 2023 Intel Corporation
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
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDepthwiseConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionTransposeMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXLinearMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXMatMulMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXSigmoidMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXHardSigmoidMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXAveragePoolMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXGlobalAveragePoolMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXAddLayerMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXSubMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXMulLayerMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConcatLayerMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXBatchNormMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXResizeMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXSoftmaxMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXExpMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXQuantizeLinearMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDequantizeLinearMetatype

from nncf.common.graph.operator_metatypes import UnknownMetatype

DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT = {
    QuantizationTrait.INPUTS_QUANTIZABLE: [
        ONNXConvolutionMetatype,
        ONNXDepthwiseConvolutionMetatype,
        ONNXConvolutionTransposeMetatype,
        ONNXLinearMetatype,
        ONNXMatMulMetatype,
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
                                        ONNXQuantizeLinearMetatype,
                                        ONNXDequantizeLinearMetatype,
                                        UnknownMetatype],
    QuantizationTrait.CONCAT: [ONNXConcatLayerMetatype],
    QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS: []
}
