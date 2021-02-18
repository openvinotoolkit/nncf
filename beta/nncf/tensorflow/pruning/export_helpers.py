"""
 Copyright (c) 2020 Intel Corporation
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
from beta.nncf.tensorflow.layers.common import ELEMENTWISE_LAYERS
from beta.nncf.tensorflow.graph.graph import NNCFNode
from nncf.common.pruning.utils import is_grouped_conv
from nncf.common.pruning.utils import get_sources_of_node
from nncf.common.pruning.export_helpers import DefaultMetaOp
from nncf.pruning.export_utils import get_input_masks

TF_PRUNING_OPERATOR_METATYPES = TFPruningOperationsMetatypeRegistry("operator_metatypes")


def _get_types(expression):
    try:
        return [expression.node_type]
    except:
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
    additional_types = _get_types(KERAS_ACTIVATIONS) \
                       + ['AvgPool2D', 'GlobalAvgPool2D', 'AveragePooling2D', 'GlobalAveragePooling2D']

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

    @classmethod
    def all_inputs_from_convs(cls, nx_node, nx_graph, graph):
        """
        Return whether all input sources of nx_node is convolutions or not
        :param nx_node: node to determine it's sources
        :param nx_graph:  networkx graph to work with
        :param graph:  NNCF graph to work with
        """
        inputs = [u for u, _ in nx_graph.in_edges(nx_node['key'])]
        input_masks = get_input_masks(nx_node, nx_graph)

        for i, inp in enumerate(inputs):
            # If input has mask ->  it went from convolution (source of this node is a convolution)
            if input_masks[i] is not None:
                continue
            nncf_input_node = graph._nx_node_to_nncf_node(nx_graph.nodes[inp])
            source_nodes = get_sources_of_node(nncf_input_node, graph, TFConvolution.get_all_op_aliases() +
                                               TFStopMaskForwardOps.get_all_op_aliases() +
                                               TFInput.get_all_op_aliases())
            sources_types = [node.node_type for node in source_nodes]
            if any([t in sources_types for t in TFStopMaskForwardOps.get_all_op_aliases()]):
                return False
        return True

    @classmethod
    def check_concat(cls, nx_node, nx_graph, graph):
        if cls.all_inputs_from_convs(nx_node, nx_graph, graph):
            return True
        return False


@TF_PRUNING_OPERATOR_METATYPES.register('elementwise')
class TFElementwise(DefaultMetaOp):
    additional_types = ELEMENTWISE_LAYERS

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True


@TF_PRUNING_OPERATOR_METATYPES.register('stop_propagation_ops')
class TFStopMaskForwardOps(DefaultMetaOp):
    additional_types = ['Average', 'Maximum', 'Minimum', 'Dense']

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return False
