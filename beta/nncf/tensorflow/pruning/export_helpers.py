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
from beta.nncf.tensorflow.pruning.utils import TFPruningOperationsMetatypeRegistry
from beta.nncf.tensorflow.pruning.utils import tf_is_depthwise_conv
from beta.nncf.tensorflow.graph.patterns import KERAS_ACTIVATIONS
from beta.nncf.tensorflow.graph.patterns import TF_ACTIVATIONS
from beta.nncf.tensorflow.layers.common import ELEMENTWISE_LAYERS
from nncf.common.graph.graph import NNCFNode
from nncf.common.pruning.utils import is_grouped_conv
from nncf.common.pruning.export_helpers import DefaultMetaOp

TF_PRUNING_OPERATOR_METATYPES = TFPruningOperationsMetatypeRegistry("operator_metatypes")


def _get_types(expression):
    try:
        return [expression.node_type]
    except AttributeError:
        types = []
        for expr in expression.expressions:
            types.extend(_get_types(expr))
        return types


@TF_PRUNING_OPERATOR_METATYPES.register('model_input')
class TFInput(DefaultMetaOp):
    additional_types = ['InputLayer']

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return False


@TF_PRUNING_OPERATOR_METATYPES.register('identity_mask_propagation')
class TFIdentityMaskForwardOps(DefaultMetaOp):
    # TODO: maybe Reshape should be in some other metaop
    additional_types = _get_types(KERAS_ACTIVATIONS | TF_ACTIVATIONS) \
                       + ['AvgPool2D', 'GlobalAvgPool2D', 'AveragePooling2D', 'GlobalAveragePooling2D'] \
                       + ['MaxPooling2D', 'GlobalMaxPooling2D', 'MaxPool2D', 'GlobalMaxPool2D'] \
                       + ['Dropout', 'Reshape', 'ZeroPadding2D', 'Identity', 'Pad', 'UpSampling2D']

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True


@TF_PRUNING_OPERATOR_METATYPES.register('convolution')
class TFConvolution(DefaultMetaOp):
    additional_types = ['Conv1D', 'Conv2D', 'Conv3D', 'DepthwiseConv2D']

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        accept_pruned_input = True
        if is_grouped_conv(node):
            if not tf_is_depthwise_conv(node):
                accept_pruned_input = False
        return accept_pruned_input


@TF_PRUNING_OPERATOR_METATYPES.register('transpose_convolution')
class TFTransposeConvolution(DefaultMetaOp):
    additional_types = ['Conv1DTranspose', 'Conv2DTranspose', 'Conv3DTranspose']

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True


@TF_PRUNING_OPERATOR_METATYPES.register('batch_norm')
class TFBatchNorm(DefaultMetaOp):
    additional_types = ['BatchNormalization']

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True


@TF_PRUNING_OPERATOR_METATYPES.register('concat')
class TFConcat(DefaultMetaOp):
    additional_types = ['Concatenate', 'ConcatV2']

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True


@TF_PRUNING_OPERATOR_METATYPES.register('elementwise')
class TFElementwise(DefaultMetaOp):
    additional_types = ELEMENTWISE_LAYERS

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True


@TF_PRUNING_OPERATOR_METATYPES.register('stop_propagation_ops')
class TFStopMaskForwardOps(DefaultMetaOp):
    additional_types = ['Dense', 'MatMul']

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return False
