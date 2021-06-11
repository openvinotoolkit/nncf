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

from typing import Union

import tensorflow as tf

from nncf.tensorflow.pruning.utils import is_depthwise_conv
from nncf.tensorflow.graph.patterns import KERAS_ACTIVATIONS
from nncf.tensorflow.graph.patterns import SET_ELEMENTWISE_LAYERS
from nncf.tensorflow.graph.patterns import TF_ACTIVATIONS
from nncf.common.graph import NNCFGraphNodeType
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.pruning.export_helpers import DefaultMetaOp
from nncf.common.pruning.mask_propagation import identity_mask_propagation
from nncf.common.pruning.mask_propagation import get_input_masks
from nncf.common.pruning.utils import get_sources_of_node
from nncf.common.pruning.utils import is_grouped_conv
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry

TF_PRUNING_OPERATOR_METATYPES = PruningOperationsMetatypeRegistry("operator_metatypes")


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
    additional_types = ['InputLayer', NNCFGraphNodeType.INPUT_NODE]

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return False

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        node.data['output_mask'] = None

@TF_PRUNING_OPERATOR_METATYPES.register('model_output')
class TFOutput(DefaultMetaOp):
    additional_types = [NNCFGraphNodeType.OUTPUT_NODE]

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
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
            output_mask = None
            if is_depthwise_conv(node):
                output_mask = input_masks[0]

        node.data['output_mask'] = output_mask


@TF_PRUNING_OPERATOR_METATYPES.register('transpose_convolution')
class TFTransposeConvolution(DefaultMetaOp):
    additional_types = ['Conv1DTranspose', 'Conv2DTranspose', 'Conv3DTranspose']

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

        # In case of group convs we can't prune by output filters
        if is_grouped_conv(node):
            output_mask = None
            if is_depthwise_conv(node):
                output_mask = input_masks[0]

        node.data['output_mask'] = output_mask


@TF_PRUNING_OPERATOR_METATYPES.register('batch_norm')
class TFBatchNorm(DefaultMetaOp):
    additional_types = ['BatchNormalization', 'SyncBatchNormalization']

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
    def check_concat(cls, node: NNCFNode, graph: NNCFGraph) -> bool:
        """
        Return whether all input sources of node is convolutions or not.

        :param node: Node to determine it's sources
        :param graph: NNCF graph to work with
        :return: True if all input sources of node is convolutions
        """

        for input_node in graph.get_previous_nodes(node):
            # If input has mask ->  it went from convolution (source of this node is a convolution)
            if input_node.data.get('output_mask', None) is None:
                continue

            source_nodes = get_sources_of_node(input_node, graph, TFConvolution.get_all_op_aliases() +
                                               TFStopMaskForwardOps.get_all_op_aliases() +
                                               TFInput.get_all_op_aliases())
            sources_types = [node.node_type for node in source_nodes]
            if any(t in sources_types for t in TFStopMaskForwardOps.get_all_op_aliases()):
                return False
        return True

    @classmethod
    def generate_output_mask(cls, node: NNCFNode, graph: NNCFGraph) -> Union[tf.Tensor, None]:
        """
        Generate output mask from input masks with all None replaced by identity masks.
        If all input masks is None return None.

        :param node: Node to determine it's sources
        :param graph: NNCF graph to work with
        :return: Output mask
        """
        input_edges = graph.get_input_edges(node)
        input_edges_desc = list(input_edges.values())
        previous_nodes = [graph.get_node_by_key(edge[0]) for edge in input_edges]
        input_masks = [input_node.data['output_mask'] for input_node in previous_nodes]

        if all(mask is None for mask in input_masks):
            return None

        device = [m for m in input_masks if m is not None][0].device

        filled_input_masks = []
        for i, mask in enumerate(input_masks):
            if mask is None:
                with tf.device(device):
                    mask = tf.ones(input_edges_desc[i][NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR][-1])
            filled_input_masks.append(mask)
        result_mask = tf.concat(filled_input_masks, 0)
        return result_mask

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        result_mask = None

        if cls.check_concat(node, graph):
            result_mask = cls.generate_output_mask(node, graph)

        node.data['output_mask'] = result_mask


@TF_PRUNING_OPERATOR_METATYPES.register('elementwise')
class TFElementwise(DefaultMetaOp):
    additional_types = list(SET_ELEMENTWISE_LAYERS)

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        input_masks = get_input_masks(node, graph)
        if input_masks[0] is not None:
            for input_mask in input_masks[1:]:
                tf.debugging.assert_near(input_masks[0], input_mask)
        node.data['output_mask'] = input_masks[0]


@TF_PRUNING_OPERATOR_METATYPES.register('stop_propagation_ops')
class TFStopMaskForwardOps(DefaultMetaOp):
    additional_types = ['Dense', 'MatMul']

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return False

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        node.data['output_mask'] = None
