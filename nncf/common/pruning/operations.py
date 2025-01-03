# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Type

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFGraphEdge
from nncf.common.graph import NNCFNode
from nncf.common.graph.layer_attributes import GroupNormLayerAttributes
from nncf.common.pruning.symbolic_mask import SymbolicMask
from nncf.common.pruning.tensor_processor import NNCFPruningBaseTensorProcessor
from nncf.common.pruning.utils import get_input_masks
from nncf.common.pruning.utils import identity_mask_propagation
from nncf.common.pruning.utils import is_grouped_conv
from nncf.common.pruning.utils import is_prunable_depthwise_conv
from nncf.common.tensor import NNCFTensor


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
    def mask_propagation(
        cls, node: NNCFNode, graph: NNCFGraph, tensor_processor: Type[NNCFPruningBaseTensorProcessor]
    ) -> None:
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
    def mask_propagation(
        cls, node: NNCFNode, graph: NNCFGraph, tensor_processor: Type[NNCFPruningBaseTensorProcessor]
    ) -> None:
        node.attributes["output_mask"] = None


class OutputPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        return True

    @classmethod
    def mask_propagation(
        cls, node: NNCFNode, graph: NNCFGraph, tensor_processor: Type[NNCFPruningBaseTensorProcessor]
    ) -> None:
        node.attributes["output_mask"] = None


class IdentityMaskForwardPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        return True

    @classmethod
    def mask_propagation(
        cls, node: NNCFNode, graph: NNCFGraph, tensor_processor: Type[NNCFPruningBaseTensorProcessor]
    ) -> None:
        identity_mask_propagation(node, graph)


class ConvolutionPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        if is_grouped_conv(node) and not is_prunable_depthwise_conv(node):
            return False
        return True

    @classmethod
    def mask_propagation(
        cls, node: NNCFNode, graph: NNCFGraph, tensor_processor: Type[NNCFPruningBaseTensorProcessor]
    ) -> None:
        input_masks = get_input_masks(node, graph)
        output_mask = node.attributes.get("output_mask", None)

        if is_grouped_conv(node):
            output_mask = None
            if is_prunable_depthwise_conv(node):
                output_mask = input_masks[0]

        node.attributes["output_mask"] = output_mask


class TransposeConvolutionPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        if is_grouped_conv(node) and not is_prunable_depthwise_conv(node):
            return False
        return True

    @classmethod
    def mask_propagation(
        cls, node: NNCFNode, graph: NNCFGraph, tensor_processor: Type[NNCFPruningBaseTensorProcessor]
    ) -> None:
        input_masks = get_input_masks(node, graph)
        output_mask = node.attributes.get("output_mask", None)

        # In case of group convs we can't prune by output filters
        if is_grouped_conv(node):
            output_mask = None
            if is_prunable_depthwise_conv(node):
                output_mask = input_masks[0]

        node.attributes["output_mask"] = output_mask


class LinearPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        return True

    @classmethod
    def mask_propagation(
        cls, node: NNCFNode, graph: NNCFGraph, tensor_processor: Type[NNCFPruningBaseTensorProcessor]
    ) -> None:
        output_mask = node.attributes.get("output_mask", None)
        node.attributes["output_mask"] = output_mask


class BatchNormPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        return True

    @classmethod
    def mask_propagation(
        cls, node: NNCFNode, graph: NNCFGraph, tensor_processor: Type[NNCFPruningBaseTensorProcessor]
    ) -> None:
        identity_mask_propagation(node, graph)


class GroupNormPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        # For Instance Normalization
        return (
            isinstance(node.layer_attributes, GroupNormLayerAttributes)
            and node.layer_attributes.num_groups == node.layer_attributes.num_channels
        )

    @classmethod
    def mask_propagation(
        cls, node: NNCFNode, graph: NNCFGraph, tensor_processor: Type[NNCFPruningBaseTensorProcessor]
    ) -> None:
        if cls.accept_pruned_input(node):
            identity_mask_propagation(node, graph)
        else:
            node.attributes["output_mask"] = None


class LayerNormPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        return True

    @classmethod
    def mask_propagation(
        cls, node: NNCFNode, graph: NNCFGraph, tensor_processor: Type[NNCFPruningBaseTensorProcessor]
    ) -> None:
        identity_mask_propagation(node, graph)


class ConcatPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        return True

    @classmethod
    def generate_output_mask(
        cls, node: NNCFNode, graph: NNCFGraph, tensor_processor: Type[NNCFPruningBaseTensorProcessor]
    ) -> Optional[NNCFTensor]:
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
        input_masks = [input_node.attributes["output_mask"] for input_node in previous_nodes]
        input_masks = [
            input_mask[node.node_name] if isinstance(input_mask, dict) else input_mask for input_mask in input_masks
        ]

        not_empty_masks: List[NNCFTensor] = [mask for mask in input_masks if mask is not None]
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
    def mask_propagation(
        cls, node: NNCFNode, graph: NNCFGraph, tensor_processor: Type[NNCFPruningBaseTensorProcessor]
    ) -> None:
        result_mask = cls.generate_output_mask(node, graph, tensor_processor)
        node.attributes["output_mask"] = result_mask


class SplitPruningOp(BasePruningOp):
    @classmethod
    def match_multiple_output_masks(
        cls, output_masks: List[SymbolicMask], output_edges: List[NNCFGraphEdge], chunk_axis: int
    ) -> Dict["str", SymbolicMask]:
        """
        Match multiple input mask to each next nodes.

        :param output_masks: Given output masks.
        :param output_edges: Given output edges of the node.
        :param chunk_axis: Given the axis on which operation was performed.
        :return: Matched output mask for each next node.
        """
        result_masks = {}
        tmp_output_masks = output_masks.copy()
        tmp_output_masks_shape = [mask.shape[0] for mask in tmp_output_masks]
        for edge in output_edges:
            node_name = edge.to_node.node_name
            idx = tmp_output_masks_shape.index(edge.tensor_shape[chunk_axis])
            result_masks[node_name] = tmp_output_masks.pop(idx)
            tmp_output_masks_shape.pop(idx)

        return result_masks

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        if node.layer_attributes is not None:
            return True
        return False

    @classmethod
    def generate_output_masks(
        cls, node: NNCFNode, graph: NNCFGraph, tensor_processor: Type[NNCFPruningBaseTensorProcessor]
    ) -> Optional[NNCFTensor]:
        """
        Generate output mask from input masks for split/chunk operations.
        If input mask is None return None

        :param node: Node to determine its sources.
        :param graph: NNCF graph to work with.
        :param tensor_processor: Interface with tensor processing methods.
        :return: Filled input masks.
        """
        input_masks = get_input_masks(node, graph)
        if not input_masks:
            return None

        assert len(input_masks) == 1
        input_mask = input_masks[0]

        if not input_mask:
            return None

        chunk_axis = node.layer_attributes.axis

        output_edges = graph.get_output_edges(node)
        output_shapes = [edge.tensor_shape[chunk_axis] for edge in output_edges]

        # if identity split detected
        if len(output_shapes) == 1:
            # propagate as is
            return input_mask

        if input_mask.shape[0] != sum(output_shapes):
            return None

        split_masks = tensor_processor.split(input_mask, output_shapes)
        result_masks = cls.match_multiple_output_masks(split_masks, output_edges, chunk_axis)

        return result_masks

    @classmethod
    def mask_propagation(
        cls, node: NNCFNode, graph: NNCFGraph, tensor_processor: Type[NNCFPruningBaseTensorProcessor]
    ) -> None:
        result_masks = cls.generate_output_masks(node, graph, tensor_processor)
        node.attributes["output_mask"] = result_masks


class PadPruningOp(IdentityMaskForwardPruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        mode, value = node.layer_attributes.mode, node.layer_attributes.value
        if mode == "constant" and value != 0:
            return False
        return True


class ElementwisePruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        return True

    @classmethod
    def mask_propagation(
        cls, node: NNCFNode, graph: NNCFGraph, tensor_processor: Type[NNCFPruningBaseTensorProcessor]
    ) -> None:
        input_masks = get_input_masks(node, graph)
        output_mask = input_masks[0]
        if output_mask is not None:
            output_mask = tensor_processor.elementwise_mask_propagation(input_masks)

        node.attributes["output_mask"] = output_mask


class ReshapePruningOp(BasePruningOp):
    @staticmethod
    def _is_flatten(node: NNCFNode) -> bool:
        return len(node.layer_attributes.output_shape) == 2

    @staticmethod
    def _is_not_mixing_dim(node: NNCFNode) -> bool:
        input_shape = node.layer_attributes.input_shape
        output_shape = node.layer_attributes.output_shape

        # TODO(dlyakhov): Cover all corner cases that appear here (ticket 90976)
        if len(input_shape) == len(output_shape) and set(input_shape) == set(output_shape):
            return input_shape == output_shape
        return True

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        return node.layer_attributes is not None

    @classmethod
    def mask_propagation(
        cls, node: NNCFNode, graph: NNCFGraph, tensor_processor: Type[NNCFPruningBaseTensorProcessor]
    ) -> None:
        if cls.accept_pruned_input(node):
            if cls._is_flatten(node):
                FlattenPruningOp.mask_propagation(node, graph, tensor_processor)
            elif cls._is_not_mixing_dim(node):
                identity_mask_propagation(node, graph)
            else:
                node.attributes["output_mask"] = None
        else:
            node.attributes["output_mask"] = None


class FlattenPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        if node.layer_attributes is not None:
            return True
        return False

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph, tensor_processor: Type[NNCFPruningBaseTensorProcessor]):
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

        node.attributes["output_mask"] = output_mask


class StopMaskForwardPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        return False

    @classmethod
    def mask_propagation(
        cls, node: NNCFNode, graph: NNCFGraph, tensor_processor: Type[NNCFPruningBaseTensorProcessor]
    ) -> None:
        node.attributes["output_mask"] = None
