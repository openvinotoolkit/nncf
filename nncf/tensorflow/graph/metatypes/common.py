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

from typing import List
from typing import Type

from nncf.common.graph import INPUT_NOOP_METATYPES
from nncf.common.graph import OUTPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.tensorflow.graph.metatypes import keras_layers as layer_metatypes
from nncf.tensorflow.graph.metatypes import tf_ops as op_metatypes

ALL_LAYER_METATYPES_WITH_WEIGHTS = [
    layer_metatypes.TFConv1DLayerMetatype,
    layer_metatypes.TFConv2DLayerMetatype,
    layer_metatypes.TFConv3DLayerMetatype,
    layer_metatypes.TFDepthwiseConv1DSubLayerMetatype,
    layer_metatypes.TFDepthwiseConv2DSubLayerMetatype,
    layer_metatypes.TFDepthwiseConv3DSubLayerMetatype,
    layer_metatypes.TFDepthwiseConv2DLayerMetatype,
    layer_metatypes.TFConv1DTransposeLayerMetatype,
    layer_metatypes.TFConv2DTransposeLayerMetatype,
    layer_metatypes.TFConv3DTransposeLayerMetatype,
    layer_metatypes.TFDenseLayerMetatype,
    layer_metatypes.TFBatchNormalizationLayerMetatype,
    layer_metatypes.TFSeparableConv1DLayerMetatype,
    layer_metatypes.TFSeparableConv2DLayerMetatype,
    layer_metatypes.TFEmbeddingLayerMetatype,
    layer_metatypes.TFLocallyConnected1DLayerMetatype,
    layer_metatypes.TFLocallyConnected2DLayerMetatype,
    # ALL_TF_OP_METATYPES_WITH_WEIGHTS
    op_metatypes.TFConv2DOpMetatype,
    op_metatypes.TFConv3DOpMetatype,
    op_metatypes.TFMatMulOpMetatype,
    op_metatypes.TFDepthwiseConv2dNativeOpMetatype,
]

GENERAL_CONV_LAYER_METATYPES = [
    layer_metatypes.TFConv1DLayerMetatype,
    layer_metatypes.TFConv2DLayerMetatype,
    layer_metatypes.TFConv3DLayerMetatype,
    layer_metatypes.TFDepthwiseConv1DSubLayerMetatype,
    layer_metatypes.TFDepthwiseConv2DSubLayerMetatype,
    layer_metatypes.TFDepthwiseConv3DSubLayerMetatype,
    layer_metatypes.TFDepthwiseConv2DLayerMetatype,
    layer_metatypes.TFConv1DTransposeLayerMetatype,
    layer_metatypes.TFConv2DTransposeLayerMetatype,
    layer_metatypes.TFConv3DTransposeLayerMetatype,
    # GENERAL_CONV_TF_OP_METATYPES
    op_metatypes.TFConv2DOpMetatype,
    op_metatypes.TFConv3DOpMetatype,
    op_metatypes.TFDepthwiseConv2dNativeOpMetatype,
]

DEPTHWISE_CONV_LAYER_METATYPES = [
    layer_metatypes.TFDepthwiseConv1DSubLayerMetatype,
    layer_metatypes.TFDepthwiseConv2DSubLayerMetatype,
    layer_metatypes.TFDepthwiseConv3DSubLayerMetatype,
    layer_metatypes.TFDepthwiseConv2DLayerMetatype,
    # DEPTHWISE_CONV_TF_OP_METATYPES
    op_metatypes.TFDepthwiseConv2dNativeOpMetatype,
]

DECONV_LAYER_METATYPES = [
    layer_metatypes.TFConv1DTransposeLayerMetatype,
    layer_metatypes.TFConv2DTransposeLayerMetatype,
    layer_metatypes.TFConv3DTransposeLayerMetatype
]

LINEAR_LAYER_METATYPES = [
    layer_metatypes.TFDenseLayerMetatype,
    # LINEAR_TF_OP_METATYPES
    op_metatypes.TFMatMulOpMetatype,
]

NORMALIZATION_LAYER_METATYPES = [
    layer_metatypes.TFBatchNormalizationLayerMetatype,
    layer_metatypes.TFLayerNormalizationLayerMetatype,
    # NORMALIZATION_TF_OP_METATYPES
    op_metatypes.TFFusedBatchNormV3OpMetatype,
]

LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT = [
    layer_metatypes.TFCropping1DLayerMetatype,
    layer_metatypes.TFCropping2DLayerMetatype,
    layer_metatypes.TFCropping3DLayerMetatype,
    layer_metatypes.TFFlattenLayerMetatype,
    layer_metatypes.TFGlobalMaxPooling1DLayerMetatype,
    layer_metatypes.TFGlobalMaxPooling2DLayerMetatype,
    layer_metatypes.TFGlobalMaxPooling3DLayerMetatype,
    layer_metatypes.TFMaxPooling1DLayerMetatype,
    layer_metatypes.TFMaxPooling2DLayerMetatype,
    layer_metatypes.TFMaxPooling3DLayerMetatype,
    layer_metatypes.TFRepeatVectorLayerMetatype,
    layer_metatypes.TFReshapeLayerMetatype,
    layer_metatypes.TFZeroPadding1DLayerMetatype,
    layer_metatypes.TFZeroPadding2DLayerMetatype,
    layer_metatypes.TFZeroPadding3DLayerMetatype,
    # TF_OP_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT
    op_metatypes.TFIdentityOpMetatype,
    op_metatypes.TFPackOpMetatype,
    op_metatypes.TFPadOpMetatype,
    op_metatypes.TFStridedSliceOpMetatype,
    op_metatypes.TFReshapeOpMetatype,
    op_metatypes.TFShapeOpMetatype,
    op_metatypes.TFMaxOpMetatype,
    op_metatypes.TFMaxPoolOpMetatype,
    op_metatypes.TFExpandDimsOpMetatype,
    op_metatypes.TFSqueezeOpMetatype,
    op_metatypes.TFMaxPool3DOpMetatype,
    op_metatypes.TFTileOpMetatype,
]

LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_MULTIPLE_CONCAT_INPUTS = [
    layer_metatypes.TFConcatenateLayerMetatype,
    op_metatypes.TFConcatOpMetatype,
]

LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_MULTIPLE_INPUTS = [
    op_metatypes.TFMaximumOpMetatype,
    op_metatypes.TFMinimumOpMetatype,
]

LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION = \
    LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT + \
    LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_MULTIPLE_CONCAT_INPUTS + \
    LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_MULTIPLE_INPUTS

ELEMENTWISE_LAYER_METATYPES = [
    layer_metatypes.TFAddLayerMetatype,
    layer_metatypes.TFMultiplyLayerMetatype,
    layer_metatypes.TFRescalingLayerMetatype,
    # ELEMENTWISE_TF_OP_METATYPES
    op_metatypes.TFAddOpMetatype,
    op_metatypes.TFMulOpMetatype,
    op_metatypes.TFBiasAddOpMetatype,
    op_metatypes.TFGreaterOpMetatype,
    op_metatypes.TFNegOpMetatype,
    op_metatypes.TFSubOpMetatype,
    op_metatypes.TFFloorDivOpMetatype,
    op_metatypes.TFMaximumOpMetatype,
    op_metatypes.TFMinimumOpMetatype,
]

RESHAPE_METATYPES = [
    layer_metatypes.TFReshapeLayerMetatype,
    layer_metatypes.TFFlattenLayerMetatype,
    op_metatypes.TFReshapeOpMetatype
]

def get_operator_metatypes() -> List[Type[OperatorMetatype]]:
    keras_metatypes_list = list(layer_metatypes.KERAS_LAYER_METATYPES.registry_dict.values())
    tf_metatypes_list = list(op_metatypes.TF_OPERATION_METATYPES.registry_dict.values())
    return list(set(keras_metatypes_list + tf_metatypes_list + \
                    list(INPUT_NOOP_METATYPES.registry_dict.values()) +
                    list(OUTPUT_NOOP_METATYPES.registry_dict.values()) +
                    list(NOOP_METATYPES.registry_dict.values())))
