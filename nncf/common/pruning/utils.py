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

import math
from enum import Enum
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import nncf
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.tensor import NNCFTensor
from nncf.common.utils.registry import Registry


def is_grouped_conv(node: NNCFNode) -> bool:
    """
    Returns `True` if a feeded node is a grouped convolution node.

    :param node: NNCFNode to check.
    :return: `True` if a feeded node a grouped convolution node.
    """
    return isinstance(node.layer_attributes, ConvolutionLayerAttributes) and node.layer_attributes.groups != 1


def is_batched_linear(node: NNCFNode, graph: NNCFGraph) -> bool:
    """
    Returns `True` if a feeded linear node output tensor has no more than two dimensions.
    A linear layer has more than two output dimensions means, that this
    linear layer multiplies several input matrices feeded by batch dimensions
    from the left/right or both inputs. Batch input dimensions are elements of [:-2] slice.

    :param node: NNCFNode to check.
    :param graph: NNCFGraph which feeded node is belonged to.
    :return: `True` if a feeded linear node output tensor has no more than two dimensions.
    """
    if not isinstance(node.layer_attributes, LinearLayerAttributes):
        return False

    edges = graph.get_output_edges(node)
    if not edges:
        return False

    return len(edges[0].tensor_shape) > 2


def get_sources_of_node(nncf_node: NNCFNode, graph: NNCFGraph, sources_types: List[str]) -> List[NNCFNode]:
    """
    Source is a node of source such that there is path from this node to `nncf_node` and on this path
    no node has one of `sources_types` type.

    :param sources_types: List of sources types.
    :param nncf_node: NNCFNode to get sources.
    :param graph: NNCF graph to work with.
    :return: List of all sources nodes.
    """
    visited = {node_id: False for node_id in graph.get_all_node_ids()}
    partial_traverse_function = partial(traverse_function, type_check_fn=lambda x: x in sources_types, visited=visited)
    nncf_nodes = [nncf_node]
    if nncf_node.node_type in sources_types:
        nncf_nodes = graph.get_previous_nodes(nncf_node)

    source_nodes = []
    for node in nncf_nodes:
        source_nodes.extend(graph.traverse_graph(node, partial_traverse_function, False))
    return source_nodes


def find_next_nodes_not_of_types(graph: NNCFGraph, nncf_node: NNCFNode, types: List[str]) -> List[NNCFNode]:
    """
    Traverse nodes in the graph from nncf node to find first nodes that aren't of type from types list.
    First nodes with some condition mean nodes:
        - for which this condition is true
        - reachable from `nncf_node` such that on the path from `nncf_node` to
          this nodes there are no other nodes with fulfilled condition

    :param graph: Graph to work with.
    :param nncf_node: NNCFNode to start search.
    :param types: List of types.
    :return: List of next nodes for nncf_node of type not from types list.
    """
    visited = {node_id: False for node_id in graph.get_all_node_ids()}
    partial_traverse_function = partial(traverse_function, type_check_fn=lambda x: x not in types, visited=visited)
    nncf_nodes = [nncf_node]
    if nncf_node.node_type not in types:
        nncf_nodes = graph.get_next_nodes(nncf_node)

    next_nodes = []
    for node in nncf_nodes:
        next_nodes.extend(graph.traverse_graph(node, partial_traverse_function))
    return next_nodes


def get_next_nodes_of_types(graph: NNCFGraph, nncf_node: NNCFNode, types: List[str]) -> List[NNCFNode]:
    """
    Looking for nodes with type from types list from `nncf_node` such that there is path
    from `nncf_node` to this node and on this path no node has one of types type.

    :param graph: Graph to work with.
    :param nncf_node: NNCFNode to start search.
    :param types: List of types to find.
    :return: List of next nodes of nncf_node with type from types list.
    """
    sources_types = types
    visited = {node_id: False for node_id in graph.get_all_node_ids()}
    partial_traverse_function = partial(traverse_function, type_check_fn=lambda x: x in sources_types, visited=visited)
    nncf_nodes = [nncf_node]
    if nncf_node.node_type in sources_types:
        nncf_nodes = graph.get_next_nodes(nncf_node)

    next_nodes = []
    for node in nncf_nodes:
        next_nodes.extend(graph.traverse_graph(node, partial_traverse_function))
    return next_nodes


def get_rounded_pruned_element_number(total: int, sparsity_rate: float, multiple_of: int = 8) -> int:
    """
    Calculates number of sparsified elements (approximately sparsity rate) from total such as
    number of remaining items will be multiple of some value.
    Always rounds number of remaining elements up.

    :param total: Total elements number.
    :param sparsity_rate: Proportion of zero elements in total.
    :param multiple_of: Number of remaining elements must be a multiple of `multiple_of`.
    :return: Number of elements to be zeroed.
    """
    remaining_elems = math.ceil((total - total * sparsity_rate) / multiple_of) * multiple_of
    return max(total - remaining_elems, 0)


def traverse_function(node: NNCFNode, output: List[NNCFNode], type_check_fn, visited) -> Tuple[bool, List[NNCFNode]]:
    if visited[node.node_id]:
        return True, output
    visited[node.node_id] = True

    if not type_check_fn(node.node_type):
        return False, output

    output.append(node)
    return True, output


def get_last_nodes_of_type(graph: NNCFGraph, op_types: List[str]) -> List[NNCFNode]:
    """
    Looking for last node in graph with type in `op_types`.
    Last == layer with type in `op_types`, that there is a path from this layer to the model output
    such that there are no other operations with type in `op_types` on it.

    :param op_types: Types of modules to track.
    :param graph: Graph to work with.
    :return: List of all last pruned nodes.
    """
    graph_outputs = graph.get_output_nodes()  # NNCFNodes here

    visited = {node_id: False for node_id in graph.get_all_node_ids()}
    partial_traverse_function = partial(traverse_function, type_check_fn=lambda x: x in op_types, visited=visited)
    last_nodes_of_type = []
    for output in graph_outputs:
        last_nodes_of_type.extend(graph.traverse_graph(output, partial_traverse_function, False))

    return last_nodes_of_type


def get_previous_convs(
    graph: NNCFGraph, nncf_node: NNCFNode, pruning_types: List[str], stop_propagation_ops: List[str]
) -> List[NNCFNode]:
    """
    Returns source convolutions of the node.

    :return: List of source convolutions of node.
    """
    sources = get_sources_of_node(nncf_node, graph, pruning_types + stop_propagation_ops)
    sources = [source for source in sources if source.node_type in pruning_types]
    return sources


def get_prunable_layers_in_out_channels(graph: NNCFGraph) -> Tuple[Dict[NNCFNodeName, int], Dict[NNCFNodeName, int]]:
    """
    Collects the number of input and output channels for each prunable layer in the graph.

    :param graph: NNCFGraph
    :return Dictionary with the number of input channels to convolution and linear layers:
            {node_name: input_channels_num}
            Dictionary with the number of output channels from convolution and linear layers:
            {node_name: output_channels_num}
    """
    in_channels, out_channels = {}, {}
    for node in graph.get_all_nodes():
        if isinstance(node.layer_attributes, (ConvolutionLayerAttributes, LinearLayerAttributes)):
            name = node.node_name
            if name in in_channels and name in out_channels:
                continue
            in_channels[name] = get_input_channels(node)
            out_channels[name] = get_output_channels(node)
    return in_channels, out_channels


class PruningOperationsMetatypeRegistry(Registry):
    def __init__(self, name):
        super().__init__(name)
        self._op_name_to_op_class = {}

    def register(self, name=None):
        name_ = name
        super_register = super()._register

        def wrap(obj):
            cls_name = name_
            if cls_name is None:
                cls_name = obj.__name__

            super_register(obj, cls_name)
            op_names = obj.get_all_op_aliases()
            for name in op_names:
                if name not in self._op_name_to_op_class:
                    self._op_name_to_op_class[name] = obj
                else:
                    assert (
                        self._op_name_to_op_class[name] == obj
                    ), "Inconsistent operator type registry - single patched op name maps to multiple metatypes!"
            return obj

        return wrap

    def get_operator_metatype_by_op_name(self, op_name: str):
        if op_name in self._op_name_to_op_class:
            return self._op_name_to_op_class[op_name]
        return None


class PruningAnalysisReason(Enum):
    """
    Enum of possible pruning analysis decisions reasons.
    """

    IGNORED_SCOPE = "node in ignored scope"
    FIRST_CONV = "this scope is one of the first convolutions"
    LAST_CONV = "this scope is convolution with output which directly affects model output dimensions"
    GROUP_CONV = "this scope is grouped convolution"
    DOWNSAMPLE_CONV = "this scope is convolution with downsample"
    MODEL_ANALYSIS = "of model analysis"
    DIMENSION_MISMATCH = "of dimension mismatch"
    CLOSING_CONV_MISSING = "closing convolution missing"
    IN_GROUP_OF_UNPRUNABLE = "is in the group with non prunable layers"
    BATCHED_LINEAR = "linear node has batched dimension(s)"
    INCOMPATIBLE_DIMS_IN_CLUSTER = "channels in cluster nodes have different values"

    @classmethod
    def message(cls, node_name: str, decision: Optional["PruningAnalysisDecision"]) -> str:
        """
        Returns the node pruning analysis decisions in a human-readable format.

        :param node_name: Name of given node.
        :param decision: Pruning analysis decision for given node.
        :return: Pruning analysis decision in a human-readable format.
        """
        prefix = f"ignored adding Weight Pruner in: {node_name}"
        reasons = decision.reasons
        if not reasons:
            return prefix
        # Filter messages
        if len(reasons) > 1 and cls.CLOSING_CONV_MISSING in reasons:
            reasons.remove(cls.CLOSING_CONV_MISSING)
        if len(reasons) == 1 and cls.IN_GROUP_OF_UNPRUNABLE in reasons:
            return ""
        return prefix + " because " + " and ".join([reason.value for reason in reasons])


class PruningAnalysisDecision:
    """
    Container for pruning analysis decisions. Contains decision which is boolean marker either
    node prunable or not (prunable if decision attribute is True) and
    pruning analysis reason in PruningAnalysisReason format. In case of positive
    decision (decision == True) possible reason will be ignored.
    """

    def __init__(
        self,
        decision: bool,
        possible_reasons: Optional[Union[List[PruningAnalysisReason], PruningAnalysisReason]] = None,
    ):
        self.decision = decision
        if not isinstance(possible_reasons, list):
            possible_reasons = [possible_reasons]
        self._reasons: Optional[List[PruningAnalysisReason]] = (
            possible_reasons if not decision and possible_reasons else None
        )

    def __repr__(self) -> str:
        representation = f"Prunable: {self.decision}"
        if not self.decision:
            representation += "; Reasons: " + str(self._reasons)
        return representation

    def __eq__(self, other: "PruningAnalysisDecision") -> bool:
        eq = self.decision == other.decision
        if self._reasons is None:
            return eq and other._reasons is None
        if other._reasons is None:
            return False
        return eq and set(self._reasons) == set(other._reasons)

    def __bool__(self) -> bool:
        return self.decision

    @property
    def reasons(self) -> Optional[List[PruningAnalysisReason]]:
        if self._reasons:
            return self._reasons.copy()
        return None

    def join(self, other: "PruningAnalysisDecision") -> "PruningAnalysisDecision":
        """
        Join two pruning analysis decisions about one NNCFNode.

        :param other: pruning analysis decision to join with.
        :return: Joint pruning analysis decision.
        """
        if self.decision and other.decision:
            return self

        reasons = []
        for decision in [self, other]:
            if decision.reasons:
                reasons.extend(decision.reasons)

        return PruningAnalysisDecision(False, reasons)


def is_prunable_depthwise_conv(node: NNCFNode) -> bool:
    # Only convolutions with in_channels == groups == out_channels are supported
    # by pruning algorithm. Depthwise convolutions support ticket: #68580
    return (
        isinstance(node.layer_attributes, ConvolutionLayerAttributes)
        and node.layer_attributes.groups == node.layer_attributes.in_channels
        and (node.layer_attributes.out_channels == node.layer_attributes.in_channels)
        and node.layer_attributes.in_channels > 1
    )


def is_conv_with_downsampling(node: NNCFNode) -> bool:
    layer_attrs = node.layer_attributes
    if isinstance(layer_attrs, ConvolutionLayerAttributes):
        return not np.all(np.array(layer_attrs.stride) == 1) and not layer_attrs.transpose
    return False


def get_input_masks(node: NNCFNode, graph: NNCFGraph) -> List[Optional[NNCFTensor]]:
    """
    Returns input masks for all inputs of given NNCFNode.

    :param node: Given NNCFNode.
    :param graph: Graph to work with.
    :return: Input masks.
    """
    retval = []
    input_masks = [input_edge.from_node.attributes["output_mask"] for input_edge in graph.get_input_edges(node)]
    for input_mask in input_masks:
        retval.append(input_mask[node.node_name] if isinstance(input_mask, dict) else input_mask)
    return retval


def get_input_channels(node: NNCFNode) -> int:
    """
    Returns count of input channels of an prunable node.

    :param node: Given prunable node.
    :return: Count of input channels of the given node.
    """
    layer_attrs: Union[ConvolutionLayerAttributes, LinearLayerAttributes] = node.layer_attributes
    if isinstance(layer_attrs, ConvolutionLayerAttributes):
        return layer_attrs.in_channels
    if isinstance(layer_attrs, LinearLayerAttributes):
        return layer_attrs.in_features
    raise nncf.InternalError(f"Can't get count of input channels from node {node}")


def get_output_channels(node: NNCFNode) -> int:
    """
    Returns count of output channels of an prunable node.

    :param node: Given prunable node.
    :return: Count of output channels of the given node.
    """
    layer_attrs: Union[ConvolutionLayerAttributes, LinearLayerAttributes] = node.layer_attributes
    if isinstance(layer_attrs, ConvolutionLayerAttributes):
        return layer_attrs.out_channels
    if isinstance(layer_attrs, LinearLayerAttributes):
        return layer_attrs.out_features
    raise nncf.InternalError(f"Can't get count of output channels from node {node}")


def identity_mask_propagation(node: NNCFNode, graph: NNCFGraph) -> None:
    """
    Propagates input mask through NNCFNode.

    :param node: Graph node to perform identity mask propagation on.
    :param graph: Graph to work with.
    """
    input_masks = get_input_masks(node, graph)
    if not input_masks:
        # In case for disconnected NNCFGraph
        input_masks = [None]
    assert len(input_masks) == 1

    node.attributes["output_mask"] = input_masks[0]
