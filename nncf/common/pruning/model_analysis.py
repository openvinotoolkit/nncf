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

import numpy as np

from typing import Callable, List, Dict

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import NNCFBaseTensorProcessor
from nncf.common.pruning.clusterization import Cluster
from nncf.common.pruning.clusterization import Clusterization
from nncf.common.pruning.operations import BasePruningOp
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.common.pruning.utils import is_depthwise_conv
from nncf.common.pruning.utils import find_next_nodes_not_of_types
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry


class SymbolicMaskProcessor(NNCFBaseTensorProcessor):
    @classmethod
    def concatenate(cls, tensors: List[NNCFTensor], axis: int) -> NNCFTensor:
        ret_tensor = np.concatenate([t.tensor for t in tensors], axis=axis)
        producers = []
        for tensor in tensors:
            if tensor.mask_producers is not None:
                producers.extend(tensor.mask_producers)
        if not producers:
            producers = None

        return SymbolicMask(ret_tensor, producers)

    @classmethod
    def ones(cls, shape: List[int], device) -> NNCFTensor:
        ret_tensor = np.ones(shape)
        return SymbolicMask(ret_tensor)

    @classmethod
    def check_all_close(cls, tensors: List[NNCFTensor]) -> None:
        for input_mask in tensors[1:]:
            assert input_mask.tensor.shape == tensors[0].tensor.shape

    @classmethod
    def repeat(cls, tensor: NNCFTensor, repeats: int) -> NNCFTensor:
        ret_tensor = np.repeat(tensor.tensor, repeats)
        return SymbolicMask(ret_tensor, tensor.mask_producers)

    @classmethod
    def elementwise_output_mask_from_input_masks(cls, tensors: List[NNCFTensor]) -> NNCFTensor:
        cls.check_all_close(tensors)
        producers = list(set([p for t in tensors for p in t.mask_producers]))
        return SymbolicMask(tensors[0].tensor, producers)


class SymbolicMask(NNCFTensor):
    def __init__(self, tensor: np.array, mask_producers: List[int] = None):
        super().__init__(tensor, SymbolicMaskProcessor)
        self._mask_producers = mask_producers

    @property
    def mask_producers(self) -> List[int]:
        return self._mask_producers

    @property
    def device(self) -> None:
        return None


class SymbolicMaskPropagationAlgorithm(MaskPropagationAlgorithm):
    def symbolic_mask_propagation(self, prunable_layers: List[str], can_prune_after_analisys: Dict[int, bool]) -> Dict[int, bool]:
        """
        Mask propagation in graph:
        to propagate masks run method mask_propagation (of metaop of current node) on all nodes in topological order.
        """
        can_prune_convs = {node.node_id: None for node in self._graph.get_all_nodes()
                           if node.node_type in prunable_layers and not is_depthwise_conv(node)}
        for node in self._graph.topological_sort():
            if node.node_id in can_prune_convs and can_prune_after_analisys[node.node_id]:
                # Set output mask
                node.data['output_mask'] = SymbolicMask(np.empty(node.layer_attributes.out_channels), [node.node_id])
            # Propagate masks
            cls = self.get_meta_operation_by_type_name(node.node_type)
            cls.mask_propagation(node, self._graph)
            if node.node_id in can_prune_convs:
                # Check input mask producers
                if node.data['input_masks'][0] is not None:
                    input_masks = node.data['input_masks']
                    assert len(input_masks) == 1
                    input_mask = input_masks[0]
                    if input_mask.mask_producers is None:
                        continue

                    for producer in input_mask.mask_producers:
                        previously_dims_equal = True if can_prune_convs[producer] is None \
                            else can_prune_convs[producer]

                        is_dims_equal = node.layer_attributes.in_channels == input_mask.tensor.shape[0]
                        can_prune_convs[producer] = previously_dims_equal and is_dims_equal

        # Clean nodes masks
        for idx in can_prune_convs:
            node = self._graph.get_node_by_id(idx)
            node.data['input_masks'] = None
            node.data['output_mask'] = None

        return {k: bool(v) for k, v in can_prune_convs.items()}


def get_position(nodes_list: List[NNCFNode], idx: int):
    for i, node in enumerate(nodes_list):
        if node.node_id == idx:
            return i
    return None


def merge_clusters_for_nodes(nodes_to_merge: List[NNCFNode], clusterization: Clusterization):
    """
    Merges clusters to which nodes from nodes_to_merge belongs.

    :param nodes_to_merge: All nodes are clusters for which should be merged.
    :param clusterization: Clusterization of nodes to work with.
    """
    if len(nodes_to_merge) <= 1:
        return

    # Will merge cluster with highest importance with others pairwise
    max_importance_node_id = None
    max_importance = 0
    for node in nodes_to_merge:
        importance = clusterization.get_cluster_containing_element(node.node_id).importance
        if importance > max_importance:
            max_importance_node_id = node.node_id
            max_importance = importance

    max_importance_cluster_id = clusterization.get_cluster_containing_element(max_importance_node_id).id
    for node in nodes_to_merge:
        if node.node_id != max_importance_node_id:
            current_node_cluster_id = clusterization.get_cluster_containing_element(node.node_id).id
            if current_node_cluster_id != max_importance_cluster_id:
                clusterization.merge_clusters(max_importance_cluster_id, current_node_cluster_id)


def cluster_special_ops(graph: NNCFGraph, special_types: List[str],
                        identity_types: List[str]) -> Clusterization[NNCFNode]:
    """
    This model will cluster all operations with type from special_types. Connected nodes is nodes that:
        1. Have path between nodes with only identity type nodes on it
        2. Have common input (identity type nodes can be on path from this input)

    :param graph: Graph to work with.
    :param special_types: List of types that should be grouped to groups of dependent nodes.
    :return: Clusterization of `special_types` nodes to the dependent groups.
    """
    topologically_sorted_nodes = graph.topological_sort()
    all_special_nodes = [node for node in graph.get_all_nodes()
                         if node.node_type in special_types]

    # 0. Initially all nodes is a separate clusters
    clusterization = Clusterization[NNCFNode](lambda x: x.node_id)
    for i, node in enumerate(all_special_nodes):
        cluster = Cluster[NNCFNode](i, [node], [get_position(topologically_sorted_nodes, node.node_id)])
        clusterization.add_cluster(cluster)

    for node in topologically_sorted_nodes:
        if node.node_type in identity_types:
            continue

        all_outputs = find_next_nodes_not_of_types(graph, node, identity_types)
        all_output_special_nodes = [node for node in all_outputs
                                    if node.node_type in special_types]
        if node.node_type in special_types:
            all_output_special_nodes.append(node)
        merge_clusters_for_nodes(all_output_special_nodes, clusterization)

    return clusterization


class ModelAnalyzer:
    """
    Analyze the model before pruning to understand which parts could potentially be pruned without conflicts
    (all nodes that can't get pruned input will receive a non-pruned input).

    The algorithm consists of three steps:
        1. Set attribute `accept_pruned_input` to all nodes. This attribute shows whether this node can
        potentially get pruned input or node.
        2. Calculate `can_prune` attribute for all nodes by propagating `accept_pruned_input` up
        (from the result of the network to the inputs). Node can be pruned if all outputs of this node accept
        pruned input and all outputs can be pruned.
        3. Propagates `can_prune` down from input nodes to the outputs.

    As a result, all nodes are marked by the `can_prune` attribute as potentially prunable or not.
    """

    def __init__(self, graph: NNCFGraph,
                 prune_operations: List[str],
                 pruning_operator_metatypes: PruningOperationsMetatypeRegistry,
                 is_depthwise_conv_fn: Callable[[NNCFNode], bool]):
        self.graph = graph
        self._prune_operations = prune_operations

        self._pruning_operator_metatypes = pruning_operator_metatypes
        pruning_op_metatypes_dict = self._pruning_operator_metatypes.registry_dict
        self._stop_propagation_op_metatype = pruning_op_metatypes_dict['stop_propagation_ops']
        self._concat_op_metatype = pruning_op_metatypes_dict['concat']
        self._convolution_op_metatype = pruning_op_metatypes_dict['convolution']

        self._is_depthwise_conv_fn = is_depthwise_conv_fn

        self.can_prune = {idx: True for idx in self.graph.get_all_node_ids()}
        self.accept_pruned_input = {idx: True for idx in self.graph.get_all_node_ids()}

    def node_propagate_can_prune_attr(self, nncf_node: NNCFNode) -> bool:
        """
        Whether the node can propagate the `can_prune` attr. That means a node can propagate pruning mask
        (for example, activations propagate mask, but convolutions stop mask propagation).

        :param nncf_node: Node to work with.
        :return: Propagates this node can_prune throw or not.
        """
        node_type = nncf_node.node_type
        is_conv = node_type in self._convolution_op_metatype.get_all_op_aliases()
        return not is_conv or (is_conv and self._is_depthwise_conv_fn(nncf_node))

    def node_accept_different_inputs(self, nncf_node: NNCFNode) -> bool:
        """
        Returns whether node accepts pruned and not pruned inputs as inputs at the same time.

        :return: Whether node accepts pruned and not pruned inputs as inputs at the same time.
        """
        return nncf_node.node_type in self._concat_op_metatype.get_all_op_aliases()

    def get_meta_operation_by_type_name(self, type_name: str) -> BasePruningOp:
        """
        Returns class of metaop that corresponds to `type_name` type.

        :return: Class of metaop that corresponds to `type_name` type.
        """
        cls = self._pruning_operator_metatypes.get_operator_metatype_by_op_name(type_name)
        if cls is None:
            cls = self._stop_propagation_op_metatype
        return cls

    def propagate_can_prune_attr_up(self):
        """
        Propagating can_prune attribute in reversed topological order.
        This attribute depends on accept_pruned_input and can_prune attributes of output nodes.
        Node can_prune is True if all outputs accept_pruned_input is True and all outputs
        (except convs because conv can be pruned by input and output independently).
        """
        reversed_sorted_nodes = reversed(self.graph.topological_sort())
        for node in reversed_sorted_nodes:
            # Check all output nodes accept_pruned_input attribute
            out_nodes = self.graph.get_next_nodes(node)
            outputs_accept_pruned_input = all(self.accept_pruned_input[node.node_id] for node in out_nodes)

            # Check all output nodes can_prune attribute
            outputs_will_be_pruned = all(self.can_prune[node.node_id]
                                         for node in out_nodes if self.node_propagate_can_prune_attr(node))
            self.can_prune[node.node_id] = outputs_accept_pruned_input and outputs_will_be_pruned

    def propagate_can_prune_attr_down(self):
        """
        Propagating can_prune attribute down to fix all branching cases with one pruned and one not pruned
        branches.
        """
        sorted_nodes = self.graph.topological_sort()
        for node in sorted_nodes:
            # Propagate attribute only in not conv case
            if self.node_propagate_can_prune_attr(node):
                in_nodes = self.graph.get_previous_nodes(node)
                can_prune = all(self.can_prune[node.node_id] for node in in_nodes)
                can_prune_any = any(self.can_prune[node.node_id] for node in in_nodes)

                if (not self.node_accept_different_inputs(node) and not can_prune) or \
                        (self.node_accept_different_inputs(node) and not can_prune_any):
                    self.can_prune[node.node_id] = can_prune

    def set_accept_pruned_input_attr(self):
        for nncf_node in self.graph.get_all_nodes():
            cls = self.get_meta_operation_by_type_name(nncf_node.node_type)
            self.accept_pruned_input[nncf_node.node_id] = cls.accept_pruned_input(nncf_node)

    def check_pruned_dimensions(self):
        mask_prop_algo = SymbolicMaskPropagationAlgorithm(self.graph, self._pruning_operator_metatypes)
        can_prune_by_dim = mask_prop_algo.symbolic_mask_propagation(self._prune_operations, self.can_prune)
        diff = [idx for idx in can_prune_by_dim if not can_prune_by_dim[idx] and self.can_prune[idx]]
        can_prune_for_prunable_layers = \
            {node_id: self.can_prune[node_id] and can_prune_by_dim[node_id] for node_id in can_prune_by_dim}
        self.can_prune.update(can_prune_for_prunable_layers)
        ## Init output_masks and can prune for each prunable layer
        #for node in self.graph.get_all_nodes():
        #    if node.node_type in self._prune_operations and self.can_prune[node.node_id]:
        #        node.data['output_mask'] = SymbolicMask(node.layer_attributes.out_channels, node.node_id)
        ## Launch mask propagation
        #MaskPropagationAlgorithm(self.graph, self._pruning_operator_metatypes).mask_propagation()
        ## Check every pruning candidate has closing prunable layer
        ## and mask dimension is equal to closing prunable layer input dimensions
        #can_prune_by_dim = {}
        #for node in self.graph.get_all_nodes():
        #    if node.node_type in self._prune_operations and 'input_mask' in node.data:
        #        input_mask = node.data['input_mask'] # type: SymbolicMask
        #        if input_mask.mask_producers is None:
        #            continue

        #        for producer in input_mask.mask_producers:
        #            previous_can_prune = can_prune_by_dim[producer] if producer in can_prune_by_dim else True
        #            if input_mask.tensor.shape[0] == node.layer_attributes.in_channels:
        #                can_prune_by_dim[producer] = previous_can_prune
        #            else:
        #                can_prune_by_dim[producer] = False
        ## Update can_prune dict
        #self.can_prune = {k: False for k in self.can_prune}
        #self.can_prune.update(can_prune_by_dim)

    def analyse_model_before_pruning(self):
        self.set_accept_pruned_input_attr()
        self.propagate_can_prune_attr_up()
        self.propagate_can_prune_attr_down()
        self.check_pruned_dimensions()
        return self.can_prune
