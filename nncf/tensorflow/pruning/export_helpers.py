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
from typing import Dict
from typing import List

import tensorflow as tf

from nncf.tensorflow.graph.pattern_operations import KERAS_ACTIVATIONS_OPERATIONS
from nncf.tensorflow.graph.pattern_operations import ELEMENTWISE_OPERATIONS
from nncf.tensorflow.graph.pattern_operations import TF_ACTIVATIONS_OPERATIONS
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.common.pruning.export_helpers import (
    OpInput,
    OpOutput,
    OpIdentityMaskForwardOps,
    OpConvolution,
    OpTransposeConvolution,
    OpBatchNorm,
    OpConcat,
    OpElementwise,
    OpStopMaskForwardOps
)

TF_PRUNING_OPERATOR_METATYPES = PruningOperationsMetatypeRegistry("operator_metatypes")


def _get_types(operations_dict: Dict) -> List[str]:
    return operations_dict['type']


@TF_PRUNING_OPERATOR_METATYPES.register('model_input')
class TFInput(OpInput):
    additional_types = ['InputLayer', NNCFGraphNodeType.INPUT_NODE]


@TF_PRUNING_OPERATOR_METATYPES.register('model_output')
class TFOutput(OpOutput):
    additional_types = [NNCFGraphNodeType.OUTPUT_NODE]


@TF_PRUNING_OPERATOR_METATYPES.register('identity_mask_propagation')
class TFIdentityMaskForwardOps(OpIdentityMaskForwardOps):
    additional_types = _get_types(KERAS_ACTIVATIONS_OPERATIONS) + _get_types(TF_ACTIVATIONS_OPERATIONS) \
                       + ['AvgPool2D', 'GlobalAvgPool2D', 'AveragePooling2D', 'GlobalAveragePooling2D'] \
                       + ['MaxPooling2D', 'GlobalMaxPooling2D', 'MaxPool2D', 'GlobalMaxPool2D'] \
                       + ['Dropout', 'ZeroPadding2D', 'Identity', 'Pad', 'UpSampling2D']


@TF_PRUNING_OPERATOR_METATYPES.register('convolution')
class TFConvolution(OpConvolution):
    additional_types = ['Conv1D', 'Conv2D', 'Conv3D', 'DepthwiseConv2D']


@TF_PRUNING_OPERATOR_METATYPES.register('transpose_convolution')
class TFTransposeConvolution(OpTransposeConvolution):
    additional_types = ['Conv1DTranspose', 'Conv2DTranspose', 'Conv3DTranspose']


@TF_PRUNING_OPERATOR_METATYPES.register('batch_norm')
class TFBatchNorm(OpBatchNorm):
    additional_types = ['BatchNormalization', 'SyncBatchNormalization']


@TF_PRUNING_OPERATOR_METATYPES.register('elementwise')
class TFElementwise(OpElementwise):
    additional_types = _get_types(ELEMENTWISE_OPERATIONS)

    @classmethod
    def _assert_input_masks_close(cls, input_masks):
        for input_mask in input_masks[1:]:
            tf.debugging.assert_near(input_masks[0], input_mask)


@TF_PRUNING_OPERATOR_METATYPES.register('stop_propagation_ops')
class TFStopMaskForwardOps(OpStopMaskForwardOps):
    additional_types = ['Dense', 'MatMul']


@TF_PRUNING_OPERATOR_METATYPES.register('concat')
class TFConcat(OpConcat):
    additional_types = ['Concatenate', 'ConcatV2']

    ConvolutionOp = TFConvolution
    StopMaskForwardOp = TFStopMaskForwardOps
    InputOp = TFInput

    @classmethod
    def _get_unit_mask(cls, dim, device):
        with tf.device(device):
            mask = tf.ones(dim)
        return mask

    @classmethod
    def _get_masks_device(cls, input_masks):
        return [m for m in input_masks if m is not None][0].device

    @classmethod
    def _concat_masks(cls, filled_input_masks):
        return tf.concat(filled_input_masks, 0)
