"""
 Copyright (c) 2019-2023 Intel Corporation
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
import copy
import itertools as it
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Hashable, List, Optional, Tuple

import networkx as nx
import networkx.algorithms.isomorphism as ism

from nncf.common.utils.backend import BackendType
from nncf.common.utils.registry import Registry
from nncf.common.utils.dot_file_rw import write_dot_graph
from nncf.parameters import TargetDevice


class HWFusedPatterns:
    """
    Stores all layer patterns to be fused determined by hardware specific.
    This essence is used in the quantization algorithm.
    The operations in these patterns should be considered as a single
    during the quantization algorithm.
    """

    def __init__(self):
        self._patterns_dict = {}
        self._full_pattern_graph = GraphPattern()

    def register(self, pattern: 'GraphPattern', name: str, match: bool = True) -> None:
        """
        Registers new pattern.

        :param pattern: pattern to be added
        :param name: name associated with the pattern
        :param match: whether should the pattern used as fussing pattern
        """
        if name in self._patterns_dict:
            raise KeyError('{} is already registered'.format(name))
        self._patterns_dict[name] = pattern
        if match:
            self._full_pattern_graph.add_pattern_alternative(pattern)

    def get_full_pattern_graph(self) -> 'GraphPattern':
        return self._full_pattern_graph

    def visualize_full_pattern_graph(self, path: str) -> None:
        self._full_pattern_graph.dump_graph(path)

    def visualize_all_patterns(self, dir_path: str) -> None:
        """
        Dump graphs of all registered patterns to dir_path
        """
        for patten_name, pattern in self._patterns_dict.items():
            pattern.dump_graph(os.path.join(dir_path, patten_name + '.dot'))

    def visualize_pattern(self, pattern_name: str, path: str) -> None:
        self._patterns_dict[pattern_name].dump_graph(os.path.join(path))


class GraphPattern:
    """
    Describes layer patterns in model's graph.
    This class is used in quantizer arrangement search algorithm, representing layer fusing patterns

    :param ANY_PATTERN_NODE_TYPE: Special node type, meaning any type inside the pattern.
    :param NON_PATTERN_NODE_TYPE: Special node type, meaning any type outside the pattern.
    """
    LABEL_ATTR = 'label'
    METATYPE_ATTR = 'type'
    ANY_PATTERN_NODE_TYPE = 'ANY_PATTERN_NODE'
    NON_PATTERN_NODE_TYPE = 'NON_PATTERN_NODE'

    def __init__(self):
        self._graph = nx.DiGraph()
        self._node_counter = 0

    def __add__(self, other: 'GraphPattern') -> 'GraphPattern':
        """
        Add DiGraph nodes of other to self and add edge between
        last node of self's graph and first node of other's graph.

        The first and last nodes are found by nx.lexicographical_topological_sort().

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
                #              (a)  (a_copy)  (b)    (b_copy)
                # A + B ---->   |       |      |        |
                #              (c)     (d)  (c_copy) (d_copy)
                #
                subgraph_copy = final_pattern._unite_with_copy_of_graph(self_subgraph)
                other_subgraph_copy = final_pattern._unite_with_copy_of_graph(other_subgraph)
                final_pattern._add_edge_connected_subgraphs(subgraph_copy, other_subgraph_copy)

        return final_pattern

    def __or__(self, other: 'GraphPattern') -> 'GraphPattern':
        """
        Add other's DiGraph nodes to self's DiGraph as a new weakly connected components.
        It is a syntax sugar of 'add_pattern_alternative()'

        :param other: GraphPattern that will be added
        :return: resulted GraphPattern.
        """
        new_pattern = copy.deepcopy(self)
        new_pattern._unite_with_copy_of_graph(other.graph)
        return new_pattern

    def __eq__(self, other: 'GraphPattern') -> bool:
        return ism.is_isomorphic(self._graph, other.graph)

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

    def _add_edge_connected_subgraphs(self,
                                      first_graph: nx.DiGraph,
                                      second_graph: nx.DiGraph) -> None:
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
        if GraphPattern.ANY_PATTERN_NODE_TYPE in second_graph.nodes[first_node_second_graph][
            GraphPattern.METATYPE_ATTR] or \
                GraphPattern.NON_PATTERN_NODE_TYPE in second_graph.nodes[first_node_second_graph][
            GraphPattern.METATYPE_ATTR]:
            successors = self_graph.successors(first_node_second_graph)
            new_edges = list(it.product([last_node_first_graph], successors))
            self_graph.add_edges_from(new_edges)
            self_graph.remove_node(first_node_second_graph)
        else:
            self_graph.add_edge(last_node_first_graph, first_node_second_graph)

    def add_pattern_alternative(self, other: 'GraphPattern') -> None:
        """
        Adds 'other' pattern as a weakly connected component to 'self' pattern.

        :param other: GraphPattern that will be added
        """
        self._unite_with_copy_of_graph(other.graph)

    def join_patterns(self, other: 'GraphPattern',
                      edges: Optional[List[Tuple[Hashable, Hashable]]] = None) -> None:
        """
        Adds 'other' pattern to 'self' pattern and connect nodes from self to other specified by 'edges'.

        If edges is None, adds an edge between
        last node of self's graph and first node of other's graph,
        which are found by nx.lexicographical_topological_sort().

        If other starts from a node with ANY_PATTERN_NODE_TYPE or NON_PATTERN_NODE_TYPE types,
        the input node of the other will be discarded from the final pattern.

        :param other: GraphPattern that will be added
        :param edges: List of edges between self and other graphs.
            Edges must begin at self and finish at other.
        """
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
        if edges is None:
            self._add_edge_connected_subgraphs(saved_graph, other_graph_copy)
        else:
            remapped_edges = []
            for edge in edges:
                new_edge = (edge[0], node_mapping[edge[1]])
                remapped_edges.append(new_edge)
            self._graph.add_edges_from(remapped_edges)

    def add_node(self, **attrs) -> int:
        if GraphPattern.METATYPE_ATTR in attrs:
            if not isinstance(attrs[GraphPattern.METATYPE_ATTR], list):
                attrs[GraphPattern.METATYPE_ATTR] = [attrs[GraphPattern.METATYPE_ATTR]]
        self._graph.add_node(self._node_counter, **attrs)
        self._node_counter += 1
        return self._node_counter - 1

    def add_edge(self, u_name, v_name) -> None:
        self._graph.add_edge(u_name, v_name)

    def add_edges_from(self, ebunch_to_add, **attr) -> None:
        self._graph.add_edges_from(ebunch_to_add, **attr)

    def get_weakly_connected_subgraphs(self) -> List[nx.DiGraph]:
        return [self._graph.subgraph(c) for c in nx.weakly_connected_components(self._graph)]

    def dump_graph(self, path: str) -> None:
        write_dot_graph(self._graph, path)


def merge_two_types_of_operations(first_op: Dict, second_op: Dict, label: str) -> Dict:
    if GraphPattern.METATYPE_ATTR in first_op and GraphPattern.METATYPE_ATTR in second_op:
        res = {GraphPattern.METATYPE_ATTR: first_op[GraphPattern.METATYPE_ATTR]}
        res[GraphPattern.METATYPE_ATTR].extend(second_op[GraphPattern.METATYPE_ATTR])
        res[GraphPattern.LABEL_ATTR] = label
        return res
    raise RuntimeError('Incorrect dicts of operations')


@dataclass
class PatternDesc:
    name: str
    devices: List[TargetDevice] = None
    backends: List[BackendType] = None
    match: bool = True


class PatternsManager(Enum):

    # COMMON PATTERNS
    SWISH_ACTIVATION = PatternDesc('swish_activation',
                                   backends=[BackendType.OPENVINO])
    SE_BLOCK = PatternDesc('se_block',
                           backends=[BackendType.OPENVINO])
    OPERATION_WITH_BIAS = PatternDesc('operation_with_bias',
                                      backends=[BackendType.OPENVINO])
    SCALE_SHIFT_ADD = PatternDesc('scale_shift_add')
    ADD_SCALE_SHIFT = PatternDesc('add_scale_shift',
                                  backends=[BackendType.OPENVINO])
    MVN_SCALE_SHIFT = PatternDesc('mvn_scale_shift',
                                  backends=[BackendType.OPENVINO])
    NORMALIZE_L2 = PatternDesc('normalize_l2',
                               backends=[BackendType.OPENVINO])
    INPUT_SCALE_SHIFT = PatternDesc('input_scale_shift')
    INPUT_TRANSPOSE_SCALE_SHIFT = PatternDesc('input_transpose_scale_shift',
                                              backends=[BackendType.OPENVINO])
    INPUT_CONVERT_TRANSPOSE_SCALE_SHIFT = PatternDesc('input_convert_transpose_scale_shift',
                                                      backends=[BackendType.OPENVINO])
    INPUT_PROCESSING = PatternDesc('input_processing',
                                   backends=[BackendType.OPENVINO, BackendType.ONNX])
    INPUT_TRANSPOSE_PROCESSING = PatternDesc('input_transpose_processing',
                                             backends=[BackendType.OPENVINO])
    INPUT_CONVERT_TRANSPOSE_PROCESSING = PatternDesc('input_convert_transpose_processing',
                                                     backends=[BackendType.OPENVINO])
    INPUT_REVERSE_INPUT_CHANNELS_SCALE_SHIFT = PatternDesc('input_reverse_input_channels_scale_shift',
                                                           backends=[BackendType.OPENVINO])
    INPUT_CONVERT_TRANSPOSE_REVERSE_INPUT_CHANNELS_SCALE_SHIFT = PatternDesc(
        'input_convert_transpose_reverse_input_channels_scale_shift', backends=[BackendType.OPENVINO])
    INPUT_REVERSE_INPUT_CHANNELS_ADD = PatternDesc('input_reverse_input_channels_add',
                                                   backends=[BackendType.OPENVINO])
    INPUT_TRANSPOSE_REVERSE_INPUT_CHANNELS_ADD = PatternDesc('input_transpose_reverse_input_channels_add',
                                                             backends=[BackendType.OPENVINO])
    INPUT_CONVERT_TRANSPOSE_REVERSE_INPUT_CHANNELS_ADD = PatternDesc(
        'input_convert_transpose_reverse_input_channels_add', backends=[BackendType.OPENVINO])
    SOFTMAX = PatternDesc('softmax',
                          backends=[BackendType.OPENVINO])
    SOFTMAX_DIV = PatternDesc('softmax_div',
                              backends=[BackendType.OPENVINO])
    SOFTMAX_RESHAPE_MATMUL = PatternDesc('softmax_reshape_matmul',
                                         backends=[BackendType.OPENVINO])
    SOFTMAX_RESHAPE_TRANSPOSE_MATMUL = PatternDesc('softmax_reshape_transpose_matmul',
                                                   backends=[BackendType.OPENVINO])
    STABLE_DIFFUSION = PatternDesc('stable_diffusion',
                                   backends=[BackendType.OPENVINO])
    SOFTMAX_RESHAPE_TRANSPOSE_GATHER_MATMUL = PatternDesc('softmax_reshape_transpose_gather_matmul',
                                                          backends=[BackendType.OPENVINO])
    HSWISH_ACTIVATION_WITHOUT_DENOMINATOR = PatternDesc('hswish_activation_without_denominator',
                                                        backends=[BackendType.OPENVINO])
    HSWISH_ACTIVATION = PatternDesc('hswish_activation',
                                    backends=[BackendType.OPENVINO])
    HSWISH_ACTIVATION_V2 = PatternDesc('hswish_activation_v2',
                                       backends=[BackendType.OPENVINO])
    FC_BN_HSWISH_ACTIVATION = PatternDesc('fc_bn_hswish_activation',
                                          backends=[BackendType.OPENVINO])
    BATCH_INDEX = PatternDesc('batch_index',
                              backends=[BackendType.OPENVINO])
    EXPERIMENTAL_DETECTRON_DETECTION_OUTPUT_ADD = PatternDesc('experimental_detectron_detection_output_add',
                                                              backends=[BackendType.OPENVINO])
    EXPERIMENTAL_DETECTRON_ROI_FEATURE_EXTRACTOR_ADD = PatternDesc('experimental_detectron_roi_feature_extractor_add',
                                                                   backends=[BackendType.OPENVINO])
    EQUAL_LOGICALNOT = PatternDesc('equal_logicalnot',
                                   backends=[BackendType.OPENVINO])
    BATCH_NORM_ACTIVATIONS = PatternDesc('batch_norm_activations',
                                         backends=[BackendType.ONNX])
    ACTIVATIONS_BATCH_NORM = PatternDesc('activations_batch_norm',
                                         backends=[BackendType.ONNX])
    ACTIVATIONS_SCALE_SHIFT = PatternDesc('activations_scale_shift',
                                          backends=[BackendType.ONNX])
    INPUT_ADD_MULTIPLY = PatternDesc('input_add_multiply',
                                     backends=[BackendType.ONNX])
    SWISH_WITH_SIGMOID = PatternDesc('swish_with_sigmoid',
                                     backends=[BackendType.ONNX])
    SWISH_WITH_HARD_SIGMOID = PatternDesc('swish_with_hard_sigmoid',
                                          backends=[BackendType.ONNX])
    INPUT_MULTIPLY_SUBTRACT = PatternDesc('input_multiply_subtract',
                                          backends=[BackendType.ONNX])
    SWISH_SIGMOID_BATCH_NORM = PatternDesc('swish_sigmoid_batch_norm',
                                           backends=[BackendType.ONNX])
    SWISH_SIGMOID_SCALE_SHIFT = PatternDesc('swish_sigmoid_scale_shift',
                                            backends=[BackendType.ONNX])
    SWISH_HARD_SIGMOID_BATCH_NORM = PatternDesc('swish_hard_sigmoid_batch_norm',
                                                backends=[BackendType.ONNX])
    SWISH_HARD_SIGMOID_SCALE_SHIFT = PatternDesc('swish_hard_sigmoid_scale_shift',
                                                 backends=[BackendType.ONNX])
    ARITHMETIC_BATCH_NORM_ACTIVATIONS = PatternDesc('arithmetic_batch_norm_activations',
                                                    backends=[BackendType.ONNX])
    ARITHMETIC_BATCH_NORM_SWISH_SIGMOID = PatternDesc('arithmetic_batch_norm_swish_sigmoid',
                                                      backends=[BackendType.ONNX])
    ARITHMETIC_BATCH_NORM_SWISH_HARD_SIGMOID = PatternDesc('arithmetic_batch_norm_swish_hard_sigmoid',
                                                           backends=[BackendType.ONNX])
    ARITHMETIC_SCALE_SHIFT_ACTIVATIONS = PatternDesc('arithmetic_scale_shift_activations',
                                                     backends=[BackendType.ONNX])
    ARITHMETIC_SCALE_SHIFT_SWISH_SIGMOID = PatternDesc('arithmetic_scale_shift_swish_sigmoid',
                                                       backends=[BackendType.ONNX])
    ARITHMETIC_SCALE_SHIFT_SWISH_HARD_SIGMOID = PatternDesc('arithmetic_scale_shift_swish_hard_sigmoid',
                                                            backends=[BackendType.ONNX])
    ARITHMETIC_ACTIVATIONS_BATCH_NORM = PatternDesc('arithmetic_activations_batch_norm',
                                                    backends=[BackendType.ONNX])
    ARITHMETIC_ACTIVATIONS_SCALE_SHIFT = PatternDesc('arithmetic_activations_scale_shift',
                                                     backends=[BackendType.ONNX])
    ARITHMETIC_BATCH_NORM = PatternDesc('arithmetic_batch_norm',
                                        backends=[BackendType.ONNX])
    ARITHMETIC_ACTIVATIONS = PatternDesc('arithmetic_activations',
                                         backends=[BackendType.ONNX])
    ARITHMETIC_SWISH_SIGMOID = PatternDesc('arithmetic_swish_sigmoid',
                                           backends=[BackendType.ONNX])
    ARITHMETIC_SWISH_HARD_SIGMOID = PatternDesc('arithmetic_swish_hard_sigmoid',
                                                backends=[BackendType.ONNX])
    ARITHMETIC_SCALE_SHIFT = PatternDesc('arithmetic_scale_shift',
                                         backends=[BackendType.ONNX])
    ARITHMETIC_SWISH_SIGMOID_BATCH_NORM = PatternDesc('arithmetic_swish_sigmoid_batch_norm',
                                                      backends=[BackendType.ONNX])
    ARITHMETIC_HARD_SIGMOID_SCALE_SHIFT = PatternDesc('arithmetic_hard_sigmoid_scale_shift',
                                                      backends=[BackendType.ONNX])
    ARITHMETIC_SWISH_HARD_SIGMOID_BATCH_NORM = PatternDesc('arithmetic_swish_hard_sigmoid_batch_norm',
                                                           backends=[BackendType.ONNX])
    ARITHMETIC_HARD_HARD_SIGMOID_SCALE_SHIFT = PatternDesc('arithmetic_hard_hard_sigmoid_scale_shift',
                                                           backends=[BackendType.ONNX])
    BATCH_NORM_SWISH_SIGMOID = PatternDesc('batch_norm_swish_sigmoid',
                                           backends=[BackendType.ONNX])
    BATCH_NORM_SWISH_HARD_SIGMOID = PatternDesc('batch_norm_swish_hard_sigmoid',
                                                backends=[BackendType.ONNX])
    SCALE_SHIFT_ACTIVATIONS = PatternDesc('scale_shift_activations',
                                          backends=[BackendType.ONNX])
    SCALE_SHIFT_SWISH_SIGMOID = PatternDesc('scale_shift_swish_sigmoid',
                                            backends=[BackendType.ONNX])
    SCALE_SHIFT_SWISH_HARD_SIGMOID = PatternDesc('scale_shift_swish_hard_sigmoid',
                                                 backends=[BackendType.ONNX])
    LINEAR_ARITHMETIC = PatternDesc('linear_arithmetic',
                                    backends=[BackendType.ONNX])
    LINEAR_BATCH_NORM = PatternDesc('linear_batch_norm',
                                    backends=[BackendType.ONNX])
    LINEAR_ACTIVATIONS = PatternDesc('linear_activations',
                                     backends=[BackendType.ONNX])
    LINEAR_SWISH_SIGMOID = PatternDesc('linear_swish_sigmoid',
                                       backends=[BackendType.ONNX])
    LINEAR_SWISH_HARD_SIGMOID = PatternDesc('linear_swish_hard_sigmoid',
                                            backends=[BackendType.ONNX])
    LINEAR_BATCH_NORM_ACTIVATIONS = PatternDesc('linear_batch_norm_activations',
                                                backends=[BackendType.ONNX])
    LINEAR_BATCH_NORM_SWISH_SIGMOID = PatternDesc('linear_batch_norm_swish_sigmoid',
                                                  backends=[BackendType.ONNX])
    LINEAR_BATCH_NORM_SWISH_HARD_SIGMOID = PatternDesc('linear_batch_norm_swish_hard_sigmoid',
                                                       backends=[BackendType.ONNX])
    LINEAR_SCALE_SHIFT_ACTIVATIONS = PatternDesc('linear_scale_shift_activations',
                                                 backends=[BackendType.ONNX])
    LINEAR_SCALE_SHIFT_SWISH_SIGMOID = PatternDesc('linear_scale_shift_swish_sigmoid',
                                                   backends=[BackendType.ONNX])
    LINEAR_SCALE_SHIFT_SWISH_HARD_SIGMOID = PatternDesc('linear_scale_shift_swish_hard_sigmoid',
                                                        backends=[BackendType.ONNX])
    LINEAR_ACTIVATIONS_BATCH_NORM = PatternDesc('linear_activations_batch_norm',
                                                backends=[BackendType.ONNX])
    LINEAR_SWISH_SIGMOID_BATCH_NORM = PatternDesc('linear_swish_sigmoid_batch_norm',
                                                  backends=[BackendType.ONNX])
    LINEAR_SWISH_HARD_SIGMOID_BATCH_NORM = PatternDesc('linear_swish_hard_sigmoid_batch_norm',
                                                       backends=[BackendType.ONNX])
    LINEAR_ACTIVATIONS_SCALE_SHIFT = PatternDesc('linear_activations_scale_shift',
                                                 backends=[BackendType.ONNX])
    LINEAR_SWISH_SIGMOID_SCALE_SHIFT = PatternDesc('linear_swish_sigmoid_scale_shift',
                                                   backends=[BackendType.ONNX])
    LINEAR_SWISH_HARD_SIGMOID_SCALE_SHIFT = PatternDesc('linear_swish_hard_sigmoid_scale_shift',
                                                        backends=[BackendType.ONNX])

    # DEVICE PATTERNS
    HSWISH_ACTIVATION_CLAMP_MULTIPLY = PatternDesc('hswish_activation_clamp_multiply',
                                               backends=[BackendType.OPENVINO],
                                               devices=[TargetDevice.CPU, TargetDevice.GPU, TargetDevice.VPU])
    LINEAR_SCALE_SHIFT = PatternDesc('',
                                 backends=[BackendType.OPENVINO, BackendType.ONNX],
                                 devices=[TargetDevice.CPU, TargetDevice.GPU])
    LINEAR_BIASED_SCALE_SHIFT = PatternDesc('linear_biased_scale_shift',
                                        backends=[BackendType.OPENVINO],
                                        devices=[TargetDevice.CPU, TargetDevice.GPU])
    LINEAR_ACTIVATION_SCALE_SHIFT = PatternDesc('linear_activation_scale_shift',
                                            backends=[BackendType.OPENVINO],
                                            devices=[TargetDevice.CPU, TargetDevice.GPU])
    LINEAR_BIASED_ACTIVATION_SCALE_SHIFT = PatternDesc('linear_biased_activation_scale_shift',
                                                   backends=[BackendType.OPENVINO],
                                                   devices=[TargetDevice.CPU, TargetDevice.GPU])
    LINEAR_ELEMENTWISE = PatternDesc('linear_elementwise',
                                 backends=[BackendType.OPENVINO],
                                 devices=[TargetDevice.CPU, TargetDevice.GPU])
    LINEAR_BIASED_ELEMENTWISE = PatternDesc('linear_biased_elementwise',
                                        backends=[BackendType.OPENVINO],
                                        devices=[TargetDevice.CPU, TargetDevice.GPU])
    LINEAR_ACTIVATION_ELEMENTWISE = PatternDesc('linear_activation_elementwise',
                                            backends=[BackendType.OPENVINO],
                                            devices=[TargetDevice.CPU, TargetDevice.GPU])
    LINEAR_BIASED_ACTIVATION_ELEMENTWISE = PatternDesc('linear_biased_activation_elementwise',
                                                   backends=[BackendType.OPENVINO],
                                                   devices=[TargetDevice.CPU, TargetDevice.GPU])

    # OPERATIONS
    LINEAR_OPERATIONS = PatternDesc('linear_operations', match=False)
    BATCH_NORMALIZATION_OPERATIONS = PatternDesc('batch_normalization_operations', match=False)
    ATOMIC_ACTIVATIONS_OPERATIONS = PatternDesc('atomic_activations_operations', match=False)
    ARITHMETIC_OPERATIONS = PatternDesc('arithmetic_operations', match=False)

    @staticmethod
    def _get_pattern_desc_by_backend(backend: BackendType) -> List[PatternDesc]:
        output = []
        for pattern in PatternsManager:
            pattern_backends = pattern.value.backends
            if pattern_backends is None or backend in pattern_backends:
                output.append(pattern)
        return output

    @staticmethod
    def _check(registered_patterns: List[PatternDesc], all_patterns: List[PatternDesc]):
        diff = set(all_patterns) - set(registered_patterns)
        if len(diff) != 0:
            raise RuntimeError('Not all patterns was registred in the backend!')

    @staticmethod
    def _filter_pattern_names_by_device(pattern_names: List[PatternDesc], device: TargetDevice) -> List[PatternDesc]:
        output = []
        for backend_pattern in pattern_names:
            pattern_devices = backend_pattern.value.devices
            if pattern_devices is None or device in pattern_devices:
                output.append(backend_pattern)
        return output

    @staticmethod
    def _get_fused_patterns_by_backend(backend: BackendType) -> Tuple[Registry, List[PatternDesc]]:
        if backend == BackendType.OPENVINO:
            from nncf.experimental.openvino_native.hardware.fused_patterns import OPENVINO_HW_FUSED_PATTERNS as ov_reg

            pattern_names = PatternsManager._get_pattern_desc_by_backend(backend)
            PatternsManager._check(ov_reg.registry_dict.keys(), pattern_names)
            return ov_reg, pattern_names

        if backend == BackendType.ONNX:
            from nncf.onnx.hardware.fused_patterns import ONNX_HW_FUSED_PATTERNS as onnx_reg

            pattern_names = PatternsManager._get_pattern_desc_by_backend(backend)
            PatternsManager._check(onnx_reg.registry_dict.keys(), pattern_names)
            return onnx_reg, pattern_names

        raise RuntimeError(f'Backend {backend} has no patterns!')

    @staticmethod
    def get_patterns(backend: BackendType, device: TargetDevice) -> HWFusedPatterns:
        hw_fused_patterns = HWFusedPatterns()
        registry, pattern_names = PatternsManager._get_fused_patterns_by_backend(backend)

        for pattern_info in PatternsManager._filter_pattern_names_by_device(pattern_names, device):
            pattern = registry.get(pattern_info)()
            hw_fused_patterns.register(pattern, pattern_info.value.name, pattern_info.value.match)
        return hw_fused_patterns
