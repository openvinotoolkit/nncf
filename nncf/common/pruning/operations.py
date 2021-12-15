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

from typing import Optional, List, Type

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.pruning.tensor_processor import NNCFPruningBaseTensorProcessor
from nncf.common.pruning.utils import is_grouped_conv
from nncf.common.pruning.utils import get_input_masks
from nncf.common.pruning.utils import is_prunable_depthwise_conv
from nncf.common.pruning.utils import identity_mask_propagation
from nncf.common.tensor import NNCFTensor
from nncf.common.graph.layer_attributes import GroupNormLayerAttributes


class BasePruningOp:
    """
    Determines meta operations which aggregate operations having common
    properties of interaction with pruning masks
    """

    subtypes = []
    additional_types = []

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        """
        :return: accept_pruned_input - can this operation work with pruned input or not
        """
        raise NotImplementedError

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        """
        Propagates the pruning mask through a node using pruning masks of all inputs and the current node (if any).

        :param node: The graph node to propagate mask through it
        :param graph: The model graph to prune
        :param tensor_processor: Interface with tensor processing methods
        """
        raise NotImplementedError

    @classmethod
    def get_all_op_aliases(cls) -> List[str]:
        """
        :return: list of all aliases of types in metatype
        """
        op_types = []
        for subtype in cls.subtypes:
            op_types.extend(subtype.get_all_aliases())
        op_types = list(set(op_types)) + cls.additional_types
        return op_types


class InputPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        return False

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        node.data['output_mask'] = None


class OutputPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        node.data['output_mask'] = None


class IdentityMaskForwardPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        identity_mask_propagation(node, graph)


class ConvolutionPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        if is_grouped_conv(node) and not is_prunable_depthwise_conv(node):
            return False
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        input_masks = get_input_masks(node, graph)
        output_mask = node.data.get('output_mask', None)

        if is_grouped_conv(node):
            output_mask = None
            if is_prunable_depthwise_conv(node):
                output_mask = input_masks[0]

        node.data['output_mask'] = output_mask


class TransposeConvolutionPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        if is_grouped_conv(node) and not is_prunable_depthwise_conv(node):
            return False
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        input_masks = get_input_masks(node, graph)
        output_mask = node.data.get('output_mask', None)

        # In case of group convs we can't prune by output filters
        if is_grouped_conv(node):
            output_mask = None
            if is_prunable_depthwise_conv(node):
                output_mask = input_masks[0]

        node.data['output_mask'] = output_mask


class BatchNormPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        identity_mask_propagation(node, graph)


class GroupNormPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        # For Instance Normalization
        return isinstance(node.layer_attributes, GroupNormLayerAttributes) \
               and node.layer_attributes.num_groups == node.layer_attributes.num_channels

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        if cls.accept_pruned_input(node):
            identity_mask_propagation(node, graph)
        else:
            node.data['output_mask'] = None


class ConcatPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def generate_output_mask(cls, node: NNCFNode, graph: NNCFGraph,
                             tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> Optional[NNCFTensor]:
        """
        Generate output mask from input masks with all None replaced by identity masks.
        If all input masks is None return None.

        :param node: Node to determine it's sources.
        :param graph: NNCF graph to work with.
        :param tensor_processor: Interface with tensor processing methods.
        :return: Filled input masks.
        """
        input_edges = graph.get_input_edges(node)
        previous_nodes = [edge.from_node for edge in input_edges]
        input_masks = [input_node.data['output_mask'] for input_node in previous_nodes]

        not_empty_masks = [mask for mask in input_masks if mask is not None]  # type: List[NNCFTensor]
        if not not_empty_masks:
            return None

        first_non_empty_mask = not_empty_masks[0]
        device = first_non_empty_mask.device
        filled_input_masks = []
        for i, mask in enumerate(input_masks):
            if mask is None:
                concat_axis = node.layer_attributes.axis
                concat_dim = input_edges[i].tensor_shape[concat_axis]
                mask = tensor_processor.ones(concat_dim, device)
            filled_input_masks.append(mask)
        result_mask = tensor_processor.concatenate(filled_input_masks, 0)
        return result_mask

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        result_mask = cls.generate_output_mask(node, graph, tensor_processor)
        node.data['output_mask'] = result_mask


class ElementwisePruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        input_masks = get_input_masks(node, graph)
        output_mask = input_masks[0]
        if output_mask is not None:
            output_mask = tensor_processor.elementwise_mask_propagation(input_masks)

        node.data['input_masks'] = input_masks
        node.data['output_mask'] = output_mask


class ReshapePruningOp(BasePruningOp):
    @staticmethod
    def _is_flatten(node: NNCFNode) -> bool:
        return len(node.layer_attributes.output_shape) == 2

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        if node.layer_attributes is None:
            return False
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        if cls.accept_pruned_input(node):
            if cls._is_flatten(node):
                FlattenPruningOp.mask_propagation(node, graph, tensor_processor)
            else:
                identity_mask_propagation(node, graph)
        else:
            node.data['output_mask'] = None


class FlattenPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        if node.layer_attributes is not None:
            return True
        return False

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]):
        output_mask = None
        input_masks = get_input_masks(node, graph)
        assert len(input_masks) == 1
        input_mask = input_masks[0]
        if input_mask is not None and node.layer_attributes is not None:
            # We assume all layers known by the mask propagation algo except
            # StopMaskForwardOp have input/output batch dim == 0.
            # Besides, since input_mask is not None thus no StopMaskForwardOp operations
            # was in the path from mask producer node to this node. As all
            # known nodes have input/output batch dim == 0 previous has too.
            flatten_channels = node.layer_attributes.output_shape[1]
            mask_len = input_mask.shape[0]
            assert flatten_channels % mask_len == 0
            output_mask = tensor_processor.repeat(input_mask, repeats=flatten_channels // mask_len)

        node.data['output_mask'] = output_mask


class StopMaskForwardPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        return False

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        node.data['output_mask'] = None
