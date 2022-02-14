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
from nncf.tensorflow.graph.metatypes import keras_layers as layer_metatypes
from nncf.tensorflow.graph.metatypes import tf_ops as op_metatypes
from nncf.tensorflow.graph.metatypes import common
from nncf.common.graph.operator_metatypes import UnknownMetatype

# If there are no some metatypes it means that they are considered as QuantizationTrait.QuantizationAgnostic

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
    ],
    QuantizationTrait.NON_QUANTIZABLE: [layer_metatypes.TFSoftmaxLayerMetatype,
                                        op_metatypes.TFSigmoidOpMetatype,
                                        op_metatypes.TFExpOpMetatype,
                                        op_metatypes.TFLogOpMetatype,
                                        op_metatypes.TFSoftmaxOpMetatype,
                                        UnknownMetatype],
    QuantizationTrait.CONCAT: [
        layer_metatypes.TFConcatenateLayerMetatype,
        op_metatypes.TFConcatOpMetatype,
    ],
    QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS: [layer_metatypes.TFEmbeddingLayerMetatype]
}
