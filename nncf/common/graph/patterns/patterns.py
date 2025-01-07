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
import copy
import itertools as it
import os
import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple, cast

import networkx as nx  # type: ignore
import networkx.algorithms.isomorphism as ism  # type: ignore

import nncf
from nncf.common.utils.dot_file_rw import write_dot_graph
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice


class Patterns:
    """
    Stores all layer patterns to be fused determined by hardware specific.
    This essence is used in the quantization algorithm.
    The operations in these patterns should be considered as a single
    during the quantization algorithm.
    """

    def __init__(self) -> None:
        self._patterns_dict: Dict[str, GraphPattern] = {}
        self._full_pattern_graph = GraphPattern()

    def register(self, pattern: "GraphPattern", name: str, match: bool = True) -> None:
        """
        Registers new pattern.

        :param pattern: pattern to be added
        :param name: name associated with the pattern
        :param match: whether should the pattern used as fussing pattern
        """
        if name in self._patterns_dict:
            raise KeyError("{} is already registered".format(name))
        self._patterns_dict[name] = pattern
        if match:
            self._full_pattern_graph.add_pattern_alternative(pattern)

    def get_full_pattern_graph(self) -> "GraphPattern":
        return self._full_pattern_graph

    def visualize_full_pattern_graph(self, path: str) -> None:
        self._full_pattern_graph.dump_graph(path)

    def visualize_all_patterns(self, dir_path: str) -> None:
        """
        Dump graphs of all registered patterns to dir_path
        """
        for patten_name, pattern in self._patterns_dict.items():
            pattern.dump_graph(os.path.join(dir_path, patten_name + ".dot"))

    def visualize_pattern(self, pattern_name: str, path: str) -> None:
        self._patterns_dict[pattern_name].dump_graph(os.path.join(path))


class GraphPattern:
    """
    Describes layer patterns in model's graph.
    This class is used in quantizer arrangement search algorithm, representing layer fusing patterns

    :param ANY_PATTERN_NODE_TYPE: Special node type, meaning any type inside the pattern.
    :param NON_PATTERN_NODE_TYPE: Special node type, meaning any type outside the pattern.
    """

    LABEL_ATTR = "label"
    METATYPE_ATTR = "type"
    NODE_TYPE_ATTR = "metatype"
    ANY_PATTERN_NODE_TYPE = "ANY_PATTERN_NODE"
    NON_PATTERN_NODE_TYPE = "NON_PATTERN_NODE"
    PATTERN_NODE_TO_EXCLUDE = "PATTERN_NODE_TO_EXCLUDE"

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._node_counter = 0

    def __add__(self, other: "GraphPattern") -> "GraphPattern":
        """
        Add DiGraph nodes of other to self and add edge between
        last node of self's graph and first node of other's graph.

        The first and the last nodes are found by nx.lexicographical_topological_sort().

        For more complex cases that are not covered by this function, use `join_patterns()`.

        :param other: GraphPattern that will be added.
        :return: resulted GraphPattern.
        """

        final_pattern = GraphPattern()
        for self_subgraph in self.get_weakly_connected_subgraphs():
            for other_subgraph in other.get_weakly_connected_subgraphs():
                # As this operation should output all graph combinations
                # It is essential to create copies of subgraphs and
                # add merge all possible connections
                # A: (a) (b)
                # B: (c) (d)
                #              (a_copy)  (a_copy)  (b_copy) (b_copy)
                # A + B ---->     |          |        |        |
                #              (c_copy)  (d_copy)  (c_copy) (d_copy)
                #
                subgraph_copy = final_pattern._unite_with_copy_of_graph(self_subgraph)
                other_subgraph_copy = final_pattern._unite_with_copy_of_graph(other_subgraph)
                final_pattern._add_edge_connected_subgraphs(subgraph_copy, other_subgraph_copy)

        return final_pattern

    def __or__(self, other: "GraphPattern") -> "GraphPattern":
        """
        Add other's DiGraph nodes to self's DiGraph as a new weakly connected components.
        It is a syntax sugar of 'add_pattern_alternative()'

        :param other: GraphPattern that will be added
        :return: resulted GraphPattern.
        """
        new_pattern = copy.deepcopy(self)
        new_pattern._unite_with_copy_of_graph(other.graph)
        return new_pattern

    def __eq__(self, other: object) -> bool:
        is_isomorphic: Callable[[Any, Any], bool] = ism.is_isomorphic
        return isinstance(other, GraphPattern) and is_isomorphic(self._graph, other.graph)

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    def _unite_with_copy_of_graph(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Creates a copy of 'graph', relabels node names according to self.node_counter
        and then unites relabeled graph with graph of 'self'.

        :param graph: graph, with which, self's graph will be united.
        :return: resulted graph.
        """
        mapping = {}
        for node in graph.nodes:
            new_node = self._node_counter
            mapping[node] = new_node
            self._node_counter += 1
        other_graph_copy = nx.relabel_nodes(graph, mapping, copy=True)
        self._graph = nx.union(self._graph, other_graph_copy)
        return other_graph_copy

    def _add_edge_connected_subgraphs(self, first_graph: nx.DiGraph, second_graph: nx.DiGraph) -> None:
        """
        Adds an edge between last node of 'first_graph' and first node of 'second_graph',
        which are found by nx.lexicographical_topological_sort().

        :param first_graph: the graph which will be traversed the first in the united graph.
        :param second_graph: the graph which will be traversed the second in the united graph.
        """
        self_graph = self._graph
        last_node_first_graph = list(nx.lexicographical_topological_sort(first_graph, key=int))[-1]
        assert first_graph.out_degree(last_node_first_graph) == 0
        first_node_second_graph = list(nx.lexicographical_topological_sort(second_graph, key=int))[0]
        assert second_graph.in_degree(first_node_second_graph) == 0

        # Special case when first node is ANY_PATTERN_NODE_TYPE or NON_PATTERN_NODE_TYPE
        if (
            GraphPattern.ANY_PATTERN_NODE_TYPE
            in second_graph.nodes[first_node_second_graph][GraphPattern.METATYPE_ATTR]
            or GraphPattern.NON_PATTERN_NODE_TYPE
            in second_graph.nodes[first_node_second_graph][GraphPattern.METATYPE_ATTR]
        ):
            successors = self_graph.successors(first_node_second_graph)
            new_edges = list(it.product([last_node_first_graph], successors))
            self_graph.add_edges_from(new_edges)
            self_graph.remove_node(first_node_second_graph)
        else:
            self_graph.add_edge(last_node_first_graph, first_node_second_graph)

    def add_pattern_alternative(self, other: "GraphPattern") -> None:
        """
        Adds 'other' pattern as a weakly connected component to 'self' pattern.

        :param other: GraphPattern that will be added
        """
        self._unite_with_copy_of_graph(other.graph)

    def join_patterns(self, other: "GraphPattern", edges: Optional[List[Tuple[Hashable, Hashable]]] = None) -> None:
        """
        Adds 'other' pattern to 'self' pattern and connect nodes from self to other specified by 'edges'.

        If edges is None, connect all weakly connected components of self and other by adding edges between
        the last nodes of every weakly component of self and the first nodes of every weakly component other.
        The first and the last nodes are found by nx.lexicographical_topological_sort().

        # A: (a) (b)
        # B: (c) (d)
        #              (a_copy)  (a_copy)  (b_copy) (b_copy)
        # A + B ---->     |          |        |        |
        #              (c_copy)  (d_copy)  (c_copy) (d_copy)
        #

        If other starts from a node with ANY_PATTERN_NODE_TYPE or NON_PATTERN_NODE_TYPE types,
        the input node of the other will be discarded from the final pattern.

        :param other: GraphPattern that will be added
        :param edges: List of edges between self and other graphs.
            Edges must begin at self and finish at other.
        """
        if edges is None:
            self._graph = (self + other).graph
        else:
            # Unite nodes
            other_graph_copy = copy.deepcopy(other.graph)
            node_mapping = {}
            for node_key in other_graph_copy.nodes:
                node_mapping[node_key] = self._node_counter
                self._node_counter += 1
            other_graph_copy = nx.relabel_nodes(other_graph_copy, node_mapping, copy=True)

            saved_graph = copy.deepcopy(self._graph)
            self._graph = nx.union(saved_graph, other_graph_copy)
            # Add edge/edges
            remapped_edges = []
            for edge in edges:
                new_edge = (edge[0], node_mapping[edge[1]])
                remapped_edges.append(new_edge)
            self._graph.add_edges_from(remapped_edges)

    def add_node(self, **attrs: Dict[str, Any]) -> int:
        if GraphPattern.METATYPE_ATTR in attrs and not isinstance(attrs[GraphPattern.METATYPE_ATTR], list):
            attrs[GraphPattern.METATYPE_ATTR] = cast(Any, [attrs[GraphPattern.METATYPE_ATTR]])
        self._graph.add_node(self._node_counter, **attrs)
        self._node_counter += 1
        return self._node_counter - 1

    def add_edge(self, u_name: str, v_name: str) -> None:
        self._graph.add_edge(u_name, v_name)

    def add_edges_from(self, ebunch_to_add: List[Any], **attr: Dict[str, Any]) -> None:
        self._graph.add_edges_from(ebunch_to_add, **attr)

    def get_weakly_connected_subgraphs(self) -> List[nx.DiGraph]:
        return [self._graph.subgraph(c) for c in nx.weakly_connected_components(self._graph)]

    def dump_graph(self, path: str) -> None:
        write_dot_graph(self._graph, pathlib.Path(path))


def merge_two_types_of_operations(first_op: Dict[str, Any], second_op: Dict[str, Any], label: str) -> Dict[str, Any]:
    if GraphPattern.METATYPE_ATTR in first_op and GraphPattern.METATYPE_ATTR in second_op:
        res = {GraphPattern.METATYPE_ATTR: first_op[GraphPattern.METATYPE_ATTR]}
        res[GraphPattern.METATYPE_ATTR].extend(second_op[GraphPattern.METATYPE_ATTR])
        res[GraphPattern.LABEL_ATTR] = label
        return res
    raise nncf.InternalError("Incorrect dicts of operations")


@dataclass
class PatternDesc:
    """
    Contains needed fields for the description of the pattern.

    :param name: Specific pattern name.
    :param devices: A field containing the list of devices
        for which this pattern should be taken into account when quantizing.
        None value means that this pattern is applicable to all devices.
    :param model_types: This field contains the list of the model types
        for which this pattern should be taken into account when quantizing.
        None value means that this pattern is applicable to all model types.
    """

    name: str
    devices: Optional[List[TargetDevice]] = None
    model_types: Optional[List[ModelType]] = None


class HWFusedPatternNames(Enum):
    """
    Describes the patterns that will be fused during integer execution
    and would not be quantized in compression pipeline.
    """

    # ATOMIC OPERATIONS
    L2_NORM = PatternDesc("l2_norm")
    MVN = PatternDesc("mvn")
    GELU = PatternDesc("gelu")

    # BLOCK PATTERNS
    ADD_SCALE_SHIFT_OUTPUT = PatternDesc("add_scale_shift_output")
    BATCH_INDEX = PatternDesc("batch_index")
    LINEAR_WITH_BIAS = PatternDesc("linear_with_bias")
    MVN_SCALE_SHIFT = PatternDesc("mvn_scale_shift")
    NORMALIZE_L2_MULTIPLY = PatternDesc("normalize_l2_multiply")
    SCALE_SHIFT = PatternDesc("scale_shift")
    SHIFT_SCALE = PatternDesc("shift_scale")
    SOFTMAX_DIV = PatternDesc("softmax_div")

    # ACTIVATIONS
    HSWISH_ACTIVATION = PatternDesc("hswish_activation")
    HSWISH_ACTIVATION_V2 = PatternDesc("hswish_activation_v2")
    HSWISH_ACTIVATION_WITHOUT_DENOMINATOR = PatternDesc("hswish_activation_without_denominator")
    SOFTMAX = PatternDesc("softmax")
    SWISH_WITH_HARD_SIGMOID = PatternDesc("swish_with_hard_sigmoid")
    SWISH_WITH_SIGMOID = PatternDesc("swish_with_sigmoid")

    # INPUT PROCESSING
    INPUT_CONVERT_TRANSPOSE_PROCESSING = PatternDesc("input_convert_transpose_processing")
    INPUT_CONVERT_TRANSPOSE_REVERSE_ADD = PatternDesc("input_convert_transpose_reverse_add")
    INPUT_CONVERT_TRANSPOSE_REVERSE_SCALE_SHIFT = PatternDesc("input_convert_transpose_reverse_scale_shift")
    INPUT_CONVERT_TRANSPOSE_SCALE_SHIFT = PatternDesc("input_convert_transpose_scale_shift")
    INPUT_PROCESSING = PatternDesc("input_processing")
    INPUT_REVERSE_ADD = PatternDesc("input_reverse_add")
    INPUT_REVERSE_SCALE_SHIFT = PatternDesc("input_reverse_scale_shift")
    INPUT_SCALE_SHIFT = PatternDesc("input_scale_shift")
    INPUT_SHIFT_SCALE = PatternDesc("input_shift_scale")
    INPUT_TRANSPOSE_PROCESSING = PatternDesc("input_transpose_processing")
    INPUT_TRANSPOSE_REVERSE_ADD = PatternDesc("input_transpose_reverse_add")
    INPUT_TRANSPOSE_SCALE_SHIFT = PatternDesc("input_transpose_scale_shift")

    # COMBINATIONS
    ACTIVATIONS_BATCH_NORM = PatternDesc("activations_batch_norm")
    ACTIVATIONS_SCALE_SHIFT = PatternDesc("activations_scale_shift")
    ARITHMETIC_ACTIVATIONS = PatternDesc("arithmetic_activations")
    ARITHMETIC_ACTIVATIONS_BATCH_NORM = PatternDesc("arithmetic_activations_batch_norm")
    # StyleGan2
    ARITHMETIC_ACTIVATIONS_ARITHMETIC = PatternDesc("arithmetic_activations_arithmetic")
    ARITHMETIC_ACTIVATIONS_SCALE_SHIFT = PatternDesc("arithmetic_activations_scale_shift")
    ARITHMETIC_BATCH_NORM = PatternDesc("arithmetic_batch_norm")
    ARITHMETIC_BATCH_NORM_ACTIVATIONS = PatternDesc("arithmetic_batch_norm_activations")
    ARITHMETIC_SCALE_SHIFT = PatternDesc("arithmetic_scale_shift")
    ARITHMETIC_SCALE_SHIFT_ACTIVATIONS = PatternDesc("arithmetic_scale_shift_activations")
    BATCH_NORM_ACTIVATIONS = PatternDesc("batch_norm_activations")
    BATCH_NORM_SCALE_SHIFT_ACTIVATIONS = PatternDesc("batch_norm_scale_shift_activations")
    GROUP_NORM_RELU = PatternDesc("group_norm_relu")
    LINEAR_ACTIVATIONS = PatternDesc("linear_activations")
    LINEAR_ACTIVATIONS_BATCH_NORM = PatternDesc("linear_activations_batch_norm")
    LINEAR_ACTIVATIONS_SCALE_SHIFT = PatternDesc("linear_activations_scale_shift")
    LINEAR_ARITHMETIC = PatternDesc("linear_arithmetic")
    LINEAR_SHIFT_SCALE = PatternDesc("linear_shift_scale")
    LINEAR_ARITHMETIC_ACTIVATIONS = PatternDesc("linear_arithmetic_activations")
    # Found in PicoDet models
    LINEAR_ARITHMETIC_ACTIVATIONS_ARITHMETIC = PatternDesc("linear_arithmetic_activations_arithmetic")
    LINEAR_BATCH_NORM = PatternDesc("linear_batch_norm")
    LINEAR_BATCH_NORM_ACTIVATIONS = PatternDesc("linear_batch_norm_activations")
    # MaskRCNN_Resnet_Atrous
    LINEAR_BATCH_TO_SPACE_SCALE_SHIFT_ACTIVATIONS = PatternDesc("linear_batch_to_space_scale_shift_activations")
    LINEAR_BATCH_TO_SPACE_ARITHMETIC_ACTIVATIONS = PatternDesc("linear_batch_to_space_arithmetic_activations")
    LINEAR_BATCH_NORM_SCALE_SHIFT_ACTIVATIONS = PatternDesc("linear_batch_norm_scale_shift_activations")
    LINEAR_SCALE_SHIFT_ACTIVATIONS = PatternDesc("linear_scale_shift_activations")
    LINEAR_CONST_MULTIPLY = PatternDesc("linear_const_multiply")
    LINEAR_SQUEEZE_ACTIVATIONS = PatternDesc("linear_squeeze_activations")
    LINEAR_SQUEEZE_ARITHMETIC_ACTIVATIONS = PatternDesc("linear_squeeze_arithmetic_activations")
    LINEAR_ACTIVATIONS_UNSQUEEZE_BN_SQUEEZE = PatternDesc("linear_activations_unsqueeze_bn_squeeze")
    SCALE_SHIFT_ACTIVATIONS = PatternDesc("scale_shift_activations")
    MVN_SCALE_SHIFT_ACTIVATIONS = PatternDesc("mvn_scale_shift_activations")

    # DEVICE PATTERNS
    HSWISH_ACTIVATION_CLAMP_MULTIPLY = PatternDesc(
        "hswish_activation_clamp_multiply",
        devices=[TargetDevice.ANY, TargetDevice.CPU, TargetDevice.GPU, TargetDevice.NPU],
    )
    LINEAR_SCALE_SHIFT = PatternDesc(
        "linear_scale_shift", devices=[TargetDevice.ANY, TargetDevice.CPU, TargetDevice.GPU]
    )
    LINEAR_BIASED_SCALE_SHIFT = PatternDesc(
        "linear_biased_scale_shift", devices=[TargetDevice.ANY, TargetDevice.CPU, TargetDevice.GPU]
    )
    LINEAR_ACTIVATION_SCALE_SHIFT = PatternDesc(
        "linear_activation_scale_shift", devices=[TargetDevice.ANY, TargetDevice.CPU, TargetDevice.GPU]
    )
    LINEAR_BIASED_ACTIVATION_SCALE_SHIFT = PatternDesc(
        "linear_biased_activation_scale_shift", devices=[TargetDevice.ANY, TargetDevice.CPU, TargetDevice.GPU]
    )
    LINEAR_ELEMENTWISE = PatternDesc(
        "linear_elementwise", devices=[TargetDevice.ANY, TargetDevice.CPU, TargetDevice.GPU]
    )
    LINEAR_BIASED_ELEMENTWISE = PatternDesc(
        "linear_biased_elementwise", devices=[TargetDevice.ANY, TargetDevice.CPU, TargetDevice.GPU]
    )
    LINEAR_ACTIVATION_ELEMENTWISE = PatternDesc(
        "linear_activation_elementwise", devices=[TargetDevice.ANY, TargetDevice.CPU, TargetDevice.GPU]
    )
    LINEAR_BIASED_ACTIVATION_ELEMENTWISE = PatternDesc(
        "linear_biased_activation_elementwise", devices=[TargetDevice.ANY, TargetDevice.CPU, TargetDevice.GPU]
    )


class IgnoredPatternNames(Enum):
    """
    Describes the patterns, which nodes should be ignored during FakeQuantize placement.
    """

    MULTIHEAD_ATTENTION_OUTPUT = PatternDesc(
        "multihead_attention_output",
        model_types=[ModelType.TRANSFORMER],
        devices=[TargetDevice.ANY, TargetDevice.CPU, TargetDevice.GPU, TargetDevice.NPU],
    )
    SE_BLOCK = PatternDesc("se_block")
    FC_BN_HSWISH_ACTIVATION = PatternDesc("fc_bn_hswish_activation")
    EQUAL_LOGICALNOT = PatternDesc("equal_logicalnot")
    ROPE = PatternDesc("rope", model_types=[ModelType.TRANSFORMER])
