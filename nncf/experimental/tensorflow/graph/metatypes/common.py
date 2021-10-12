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

from nncf.common.graph import INPUT_NOOP_METATYPES
from nncf.common.graph import OUTPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.experimental.tensorflow.graph.metatypes import tf_ops as op_metatypes


ALL_TF_OP_METATYPES_WITH_WEIGHTS = [
    op_metatypes.TFConv2DOpMetatype,
    op_metatypes.TFConv3DOpMetatype,
    op_metatypes.TFMatMulOpMetatype,
    op_metatypes.TFDepthwiseConv2dNativeOpMetatype,
]


GENERAL_CONV_TF_OP_METATYPES = [
    op_metatypes.TFConv2DOpMetatype,
    op_metatypes.TFConv3DOpMetatype,
    op_metatypes.TFDepthwiseConv2dNativeOpMetatype,
]


DEPTHWISE_CONV_TF_OP_METATYPES = [
    op_metatypes.TFDepthwiseConv2dNativeOpMetatype,
]


LINEAR_TF_OP_METATYPES = [
    op_metatypes.TFMatMulOpMetatype,
]


NORMALIZATION_TF_OP_METATYPES = [
    op_metatypes.TFFusedBatchNormV3OpMetatype,
]


TF_OP_METATYPES_AGNOSTIC_TO_DATA_PRECISION_ONE_INPUT = [
    op_metatypes.TFShapeOpMetatype,
    op_metatypes.TFIdentityOpMetatype,
    op_metatypes.TFPadOpMetatype,
    op_metatypes.TFStridedSliceOpMetatype,
    op_metatypes.TFReshapeOpMetatype,
    op_metatypes.TFMaxOpMetatype,
    op_metatypes.TFMaxPoolOpMetatype,
    op_metatypes.TFExpandDimsOpMetatype,
    op_metatypes.TFSqueezeOpMetatype,
    op_metatypes.TFMaxPool3DOpMetatype,
    op_metatypes.TFTileOpMetatype,
    op_metatypes.TFPackOpMetatype,
]


TF_OP_METATYPES_AGNOSTIC_TO_DATA_PRECISION_MULTIPLE_INPUTS = [
    op_metatypes.TFConcatOpMetatype,
]


TF_OP_METATYPES_AGNOSTIC_TO_DATA_PRECISION = [
    *TF_OP_METATYPES_AGNOSTIC_TO_DATA_PRECISION_ONE_INPUT,
    *TF_OP_METATYPES_AGNOSTIC_TO_DATA_PRECISION_MULTIPLE_INPUTS,
]


ELEMENTWISE_TF_OP_METATYPES = [
    op_metatypes.TFBiasAddOpMetatype,
    op_metatypes.TFAddOpMetatype,
    op_metatypes.TFGreaterOpMetatype,
    op_metatypes.TFMulOpMetatype,
    op_metatypes.TFNegOpMetatype,
    op_metatypes.TFSubOpMetatype,
]


def get_operator_metatypes() -> List[Type[OperatorMetatype]]:
    """
    Returns all registered meta types for the TensorFlow backend.

    :return: All registered meta types for the TensorFlow backend.
    """
    tf_op_metatypes = list(op_metatypes.TF_OPERATION_METATYPES.registry_dict.values())
    operator_metatypes = list(
        set(
            tf_op_metatypes +
            list(INPUT_NOOP_METATYPES.registry_dict.values()) +
            list(OUTPUT_NOOP_METATYPES.registry_dict.values()) +
            list(NOOP_METATYPES.registry_dict.values())
        )
    )

    return operator_metatypes
