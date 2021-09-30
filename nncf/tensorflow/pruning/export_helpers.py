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
from typing import Union

import tensorflow as tf

from nncf.tensorflow.graph.pattern_operations import KERAS_ACTIVATIONS_OPERATIONS
from nncf.tensorflow.graph.pattern_operations import ELEMENTWISE_OPERATIONS
from nncf.tensorflow.graph.pattern_operations import TF_ACTIVATIONS_OPERATIONS
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.pruning.mask_propagation import get_input_masks
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
    OpReshape,
    OpFlatten,
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
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        input_masks = get_input_masks(node, graph)
        if input_masks[0] is not None:
            for input_mask in input_masks[1:]:
                tf.debugging.assert_near(input_masks[0], input_mask)
        node.data['output_mask'] = input_masks[0]


@TF_PRUNING_OPERATOR_METATYPES.register('reshape')
class TFReshapeOps(OpReshape):
    additional_types = ['Reshape']


@TF_PRUNING_OPERATOR_METATYPES.register('flatten')
class TFFlattenOps(OpFlatten):
    additional_types = ['Flatten']


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
    def generate_output_mask(cls, node: NNCFNode, graph: NNCFGraph) -> Union[tf.Tensor, None]:
        """
        Generate output mask from input masks with all None replaced by identity masks.
        If all input masks is None return None.

        :param node: Node to determine it's sources
        :param graph: NNCF graph to work with
        :return: Output mask
        """
        input_edges = graph.get_input_edges(node)
        previous_nodes = [edge.from_node for edge in input_edges]
        input_masks = [input_node.data['output_mask'] for input_node in previous_nodes]

        if all(mask is None for mask in input_masks):
            return None

        device = [m for m in input_masks if m is not None][0].device

        filled_input_masks = []
        for i, mask in enumerate(input_masks):
            if mask is None:
                with tf.device(device):
                    mask = tf.ones(input_edges[i].tensor_shape[-1])
            filled_input_masks.append(mask)
        result_mask = tf.concat(filled_input_masks, 0)
        return result_mask

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        result_mask = None

        if cls.check_concat(node, graph):
            result_mask = cls.generate_output_mask(node, graph)

        node.data['output_mask'] = result_mask
