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
import tensorflow as tf

from beta.nncf.tensorflow.pruning.utils import TFPruningOperationsMetatypeRegistry
from beta.nncf.tensorflow.pruning.utils import is_depthwise_conv
from beta.nncf.tensorflow.graph.patterns import KERAS_ACTIVATIONS
from beta.nncf.tensorflow.graph.patterns import TF_ACTIVATIONS
from beta.nncf.tensorflow.layers.common import ELEMENTWISE_LAYERS
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.graph import NNCFGraph
from nncf.common.pruning.utils import is_grouped_conv
from nncf.common.pruning.export_helpers import DefaultMetaOp
from nncf.common.pruning.mask_propagator import identity_mask_propagation
from nncf.common.pruning.mask_propagator import get_input_masks

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

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        node.data['input_masks'] = []
        node.data['output_mask'] = None


@TF_PRUNING_OPERATOR_METATYPES.register('identity_mask_propagation')
class TFIdentityMaskForwardOps(DefaultMetaOp):
    additional_types = _get_types(KERAS_ACTIVATIONS | TF_ACTIVATIONS) \
                       + ['AvgPool2D', 'GlobalAvgPool2D', 'AveragePooling2D', 'GlobalAveragePooling2D'] \
                       + ['MaxPooling2D', 'GlobalMaxPooling2D', 'MaxPool2D', 'GlobalMaxPool2D'] \
                       + ['Dropout', 'ZeroPadding2D', 'Identity', 'Pad', 'UpSampling2D']

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        identity_mask_propagation(node, graph)


@TF_PRUNING_OPERATOR_METATYPES.register('convolution')
class TFConvolution(DefaultMetaOp):
    additional_types = ['Conv1D', 'Conv2D', 'Conv3D', 'DepthwiseConv2D']

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        accept_pruned_input = True
        if is_grouped_conv(node):
            if not is_depthwise_conv(node):
                accept_pruned_input = False
        return accept_pruned_input

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        input_masks = get_input_masks(node, graph)
        output_mask = node.data.get('output_mask', None)

        if is_grouped_conv(node):
            if is_depthwise_conv(node):
                output_mask = input_masks[0]
            else:
                output_mask = None

        node.data['input_masks'] = input_masks
        node.data['output_mask'] = output_mask


@TF_PRUNING_OPERATOR_METATYPES.register('transpose_convolution')
class TFTransposeConvolution(DefaultMetaOp):
    additional_types = ['Conv1DTranspose', 'Conv2DTranspose', 'Conv3DTranspose']

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        input_masks = get_input_masks(node, graph)
        output_mask = node.data.get('output_mask', None)

        # In case of group convs we can't prune by output filters
        if is_grouped_conv(node):
            if is_depthwise_conv(node):
                output_mask = input_masks[0]
            else:
                output_mask = None

        node.data['input_masks'] = input_masks
        node.data['output_mask'] = output_mask


@TF_PRUNING_OPERATOR_METATYPES.register('batch_norm')
class TFBatchNorm(DefaultMetaOp):
    additional_types = ['BatchNormalization']

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        identity_mask_propagation(node, graph)


@TF_PRUNING_OPERATOR_METATYPES.register('concat')
class TFConcat(DefaultMetaOp):
    additional_types = ['Concatenate', 'ConcatV2']

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        input_masks = get_input_masks(node, graph)

        #TODO check and generate masks for None
        result_mask = tf.concat(input_masks, axis=0)

        node.data['input_masks'] = input_masks
        node.data['output_mask'] = result_mask


@TF_PRUNING_OPERATOR_METATYPES.register('elementwise')
class TFElementwise(DefaultMetaOp):
    additional_types = ELEMENTWISE_LAYERS

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        input_masks = get_input_masks(node, graph)

        node.data['input_masks'] = input_masks
        if input_masks[0] is not None:
            assert all(tf.debugging.assert_near(input_masks[0], mask) for mask in input_masks)
        node.data['output_mask'] = input_masks[0]


@TF_PRUNING_OPERATOR_METATYPES.register('stop_propagation_ops')
class TFStopMaskForwardOps(DefaultMetaOp):
    additional_types = ['Dense', 'MatMul']

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return False

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        input_masks = get_input_masks(node, graph)

        node.data['input_masks'] = input_masks
        node.data['output_mask'] = None
