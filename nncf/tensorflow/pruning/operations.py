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

from nncf.tensorflow.graph.pattern_operations import KERAS_ACTIVATIONS_OPERATIONS
from nncf.tensorflow.graph.pattern_operations import ELEMENTWISE_OPERATIONS
from nncf.tensorflow.graph.pattern_operations import TF_ACTIVATIONS_OPERATIONS
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.common.pruning.operations import (
    InputPruningOp,
    OutputPruningOp,
    IdentityMaskForwardPruningOp,
    ConvolutionPruningOp,
    TransposeConvolutionPruningOp,
    BatchNormPruningOp,
    ConcatPruningOp,
    ElementwisePruningOp,
    ReshapePruningOp,
    FlattenPruningOp,
    StopMaskForwardPruningOp
)

TF_PRUNING_OPERATOR_METATYPES = PruningOperationsMetatypeRegistry("operator_metatypes")


def _get_types(operations_dict: Dict) -> List[str]:
    return operations_dict['type']


@TF_PRUNING_OPERATOR_METATYPES.register('model_input')
class TFInputPruningOp(InputPruningOp):
    additional_types = ['InputLayer', NNCFGraphNodeType.INPUT_NODE]


@TF_PRUNING_OPERATOR_METATYPES.register('model_output')
class TFOutputPruningOp(OutputPruningOp):
    additional_types = [NNCFGraphNodeType.OUTPUT_NODE]


@TF_PRUNING_OPERATOR_METATYPES.register('identity_mask_propagation')
class TFIdentityMaskForwardPruningOp(IdentityMaskForwardPruningOp):
    additional_types = _get_types(KERAS_ACTIVATIONS_OPERATIONS) + _get_types(TF_ACTIVATIONS_OPERATIONS) \
                       + ['AvgPool2D', 'GlobalAvgPool2D', 'AveragePooling2D', 'GlobalAveragePooling2D'] \
                       + ['MaxPooling2D', 'GlobalMaxPooling2D', 'MaxPool2D', 'GlobalMaxPool2D'] \
                       + ['Dropout', 'ZeroPadding2D', 'Identity', 'Pad', 'UpSampling2D']


@TF_PRUNING_OPERATOR_METATYPES.register('convolution')
class TFConvolutionPruningOp(ConvolutionPruningOp):
    additional_types = ['Conv1D', 'Conv2D', 'Conv3D', 'DepthwiseConv2D']


@TF_PRUNING_OPERATOR_METATYPES.register('transpose_convolution')
class TFTransposeConvolutionPruningOp(TransposeConvolutionPruningOp):
    additional_types = ['Conv1DTranspose', 'Conv2DTranspose', 'Conv3DTranspose']


@TF_PRUNING_OPERATOR_METATYPES.register('batch_norm')
class TFBatchNormPruningOp(BatchNormPruningOp):
    additional_types = ['BatchNormalization', 'SyncBatchNormalization']


@TF_PRUNING_OPERATOR_METATYPES.register('elementwise')
class TFElementwisePruningOp(ElementwisePruningOp):
    additional_types = _get_types(ELEMENTWISE_OPERATIONS)


@TF_PRUNING_OPERATOR_METATYPES.register('reshape')
class TFReshapeOps(ReshapePruningOp):
    additional_types = ['Reshape']


@TF_PRUNING_OPERATOR_METATYPES.register('flatten')
class TFFlattenOps(FlattenPruningOp):
    additional_types = ['Flatten']


@TF_PRUNING_OPERATOR_METATYPES.register('stop_propagation_ops')
class TFStopMaskForwardPruningOp(StopMaskForwardPruningOp):
    additional_types = ['Dense', 'MatMul']


@TF_PRUNING_OPERATOR_METATYPES.register('concat')
class TFConcatPruningOp(ConcatPruningOp):
    additional_types = ['Concatenate', 'ConcatV2']
