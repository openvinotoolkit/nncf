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
from nncf.tensorflow.graph.metatypes import common
from nncf.tensorflow.graph.metatypes import keras_layers as layer_metatypes
from nncf.tensorflow.graph.metatypes import tf_ops as op_metatypes
from nncf.tensorflow.graph.metatypes.common import LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_MULTIPLE_INPUTS
from nncf.tensorflow.graph.metatypes.common import LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_MULTIPLE_OUTPUTS
from nncf.tensorflow.graph.metatypes.common import LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT

# If a metatype is not in this list, then it is considered to be QuantizationTrait.NON_QUANTIZABLE.

DEFAULT_TF_QUANT_TRAIT_TO_OP_DICT = {
    QuantizationTrait.INPUTS_QUANTIZABLE: [
        *common.GENERAL_CONV_LAYER_METATYPES,
        *common.DEPTHWISE_CONV_LAYER_METATYPES,
        *common.DECONV_LAYER_METATYPES,
        *common.LINEAR_LAYER_METATYPES,
        *common.NORMALIZATION_LAYER_METATYPES,
        *common.ELEMENTWISE_LAYER_METATYPES,
        layer_metatypes.TFLocallyConnected1DLayerMetatype,
        layer_metatypes.TFLocallyConnected2DLayerMetatype,
        layer_metatypes.TFAveragePooling1DLayerMetatype,
        layer_metatypes.TFAveragePooling2DLayerMetatype,
        layer_metatypes.TFAveragePooling3DLayerMetatype,
        layer_metatypes.TFGlobalAveragePooling1DLayerMetatype,
        layer_metatypes.TFGlobalAveragePooling2DLayerMetatype,
        layer_metatypes.TFGlobalAveragePooling3DLayerMetatype,
        layer_metatypes.TFUpSampling1DLayerMetatype,
        layer_metatypes.TFUpSampling2DLayerMetatype,
        layer_metatypes.TFUpSampling3DLayerMetatype,
        layer_metatypes.TFAverageLayerMetatype,
        layer_metatypes.TFThresholdedReLULayerMetatype,
        layer_metatypes.TFELULayerMetatype,
        layer_metatypes.TFPReLULayerMetatype,
        layer_metatypes.TFLeakyReLULayerMetatype,
        layer_metatypes.TFActivationLayerMetatype,
        op_metatypes.TFAvgPoolOpMetatype,
        op_metatypes.TFAvgPool3DOpMetatype,
        op_metatypes.TFMeanOpMetatype,
        op_metatypes.TFResizeNearestNeighborOpMetatype,
        op_metatypes.TFEluOpMetatype,
        op_metatypes.TFLeakyReluOpMetatype,
        op_metatypes.TFRelu6OpMetatype,
        op_metatypes.TFBatchMatMulV2OpMetatype,
    ],
    QuantizationTrait.QUANTIZATION_AGNOSTIC: LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT
    + LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_MULTIPLE_INPUTS
    + LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_MULTIPLE_OUTPUTS,
    QuantizationTrait.CONCAT: [
        layer_metatypes.TFConcatenateLayerMetatype,
        op_metatypes.TFConcatOpMetatype,
    ],
    QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS: [layer_metatypes.TFEmbeddingLayerMetatype],
}
