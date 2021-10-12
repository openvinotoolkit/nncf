"""
 Copyright (c) 2021 Intel Corporation
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
from nncf.experimental.tensorflow.graph.metatypes import tf_ops as op_metatypes
from nncf.experimental.tensorflow.graph.metatypes import common


DEFAULT_TF_QUANT_TRAIT_TO_OP_DICT = {
    QuantizationTrait.INPUTS_QUANTIZABLE: [
        *common.GENERAL_CONV_TF_OP_METATYPES,
        *common.DEPTHWISE_CONV_TF_OP_METATYPES,
        *common.LINEAR_TF_OP_METATYPES,
        *common.NORMALIZATION_TF_OP_METATYPES,
        *common.ELEMENTWISE_TF_OP_METATYPES,
        op_metatypes.TFAvgPoolOpMetatype,
        op_metatypes.TFAvgPool3DOpMetatype,
        op_metatypes.TFMeanOpMetatype,
        op_metatypes.TFResizeNearestNeighborOpMetatype,
        op_metatypes.TFEluOpMetatype,
        op_metatypes.TFLeakyReluOpMetatype,
        op_metatypes.TFRelu6OpMetatype,
        op_metatypes.TFReluOpMetatype,
    ],
    QuantizationTrait.NON_QUANTIZABLE: [
        op_metatypes.TFSoftmaxOpMetatype,
    ],
    QuantizationTrait.CONCAT: [
        op_metatypes.TFConcatOpMetatype,
    ],
    QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS: [
    ]
}
