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
from nncf.experimental.onnx.graph.metatypes.onnx_ops import ConvolutionMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import LinearMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import SigmoidMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import GlobalAveragePoolMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import AddLayerMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import MulLayerMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import ConcatLayerMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import BatchNormMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import ResizeMetatype

from nncf.common.graph.operator_metatypes import UnknownMetatype

DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT = {
    QuantizationTrait.INPUTS_QUANTIZABLE: [
        ConvolutionMetatype,
        LinearMetatype,
        GlobalAveragePoolMetatype,
        AddLayerMetatype,
        MulLayerMetatype,
        BatchNormMetatype,
        ResizeMetatype,
    ],
    QuantizationTrait.NON_QUANTIZABLE: [SigmoidMetatype,
                                        UnknownMetatype],
    QuantizationTrait.CONCAT: [ConcatLayerMetatype],
    QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS: []
}
