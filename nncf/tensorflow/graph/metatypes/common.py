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

from typing import List, Type

from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.tensorflow.graph.metatypes import keras_layers as layer_metatypes
from nncf.tensorflow.graph.metatypes import nncf_op as nncf_op_metatypes
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
    layer_metatypes.TFConv3DTransposeLayerMetatype
]

DEPTHWISE_CONV_LAYER_METATYPES = [
    layer_metatypes.TFDepthwiseConv1DSubLayerMetatype,
    layer_metatypes.TFDepthwiseConv2DSubLayerMetatype,
    layer_metatypes.TFDepthwiseConv3DSubLayerMetatype,
    layer_metatypes.TFDepthwiseConv2DLayerMetatype
]

DECONV_LAYER_METATYPES = [
    layer_metatypes.TFConv1DTransposeLayerMetatype,
    layer_metatypes.TFConv2DTransposeLayerMetatype,
    layer_metatypes.TFConv3DTransposeLayerMetatype
]

LINEAR_LAYER_METATYPES = [
    layer_metatypes.TFDenseLayerMetatype
]

NORMALIZATION_LAYER_METATYPES = [
    layer_metatypes.TFBatchNormalizationLayerMetatype,
    layer_metatypes.TFLayerNormalizationLayerMetatype
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
    op_metatypes.TFIdentityOpMetatype,
    op_metatypes.TFPackOpMetatype,
    op_metatypes.TFPadOpMetatype,
    op_metatypes.TFStridedSliceOpMetatype,
    op_metatypes.TFReshapeOpMetatype
]

LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_MULTIPLE_INPUTS = [
    layer_metatypes.TFConcatenateLayerMetatype,
    op_metatypes.TFConcatOpMetatype
]

LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION = \
    LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT + \
    LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_MULTIPLE_INPUTS

ELEMENTWISE_LAYER_METATYPES = [
    layer_metatypes.TFAddLayerMetatype,
    layer_metatypes.TFMultiplyLayerMetatype,
    layer_metatypes.TFRescalingLayerMetatype,
    op_metatypes.TFAddOpMetatype,
    op_metatypes.TFMulOpMetatype
]

INPUT_LAYER_METATYPES = [
    nncf_op_metatypes.InputNoopMetatype,
    layer_metatypes.TFInputLayerMetatype
]

OUTPUT_LAYER_METATYPES = [
    nncf_op_metatypes.OutputNoopMetatype,
]


def get_operator_metatypes() -> List[Type[OperatorMetatype]]:
    """
    Returns a list of the operator metatypes.

    :return: List of the operator metatypes .
    """
    keras_metatypes_list = list(layer_metatypes.KERAS_LAYER_METATYPES.registry_dict.values())
    tf_metatypes_list = list(op_metatypes.TF_OPERATION_METATYPES.registry_dict.values())
    return keras_metatypes_list + tf_metatypes_list


def get_input_metatypes() -> List[Type[OperatorMetatype]]:
    """
    Returns a list of the input operator metatypes.

    :return: List of the input operator metatypes .
    """
    return INPUT_LAYER_METATYPES


def get_output_metatypes() -> List[Type[OperatorMetatype]]:
    """
    Returns a list of the output operator metatypes.

    :return: List of the output operator metatypes .
    """
    return OUTPUT_LAYER_METATYPES
