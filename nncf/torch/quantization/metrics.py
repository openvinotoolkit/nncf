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

from collections import deque
from copy import deepcopy
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

from nncf.common.collector import StatisticsCollector
from nncf.common.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.graph_matching import find_subgraphs_matching_pattern
from nncf.common.graph.patterns.manager import PatternsManager
from nncf.common.graph.patterns.manager import TargetDevice
from nncf.common.quantization.collectors import QuantizationStatisticsCollector
from nncf.common.quantization.collectors import QuantizerDescription
from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.common.quantization.structs import NonWeightQuantizerId
from nncf.common.quantization.structs import WeightQuantizerId
from nncf.common.utils.backend import BackendType
from nncf.common.utils.debug import is_debug
from nncf.torch.nncf_module_replacement import is_nncf_module
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.nncf_network import PTNNCFGraph
from nncf.torch.quantization.default_quantization import DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.quantization.statistics import MemoryConsumptionStatistics
from nncf.torch.quantization.statistics import QuantizationConfigurationStatistics
from nncf.torch.quantization.structs import NonWeightQuantizerInfo
from nncf.torch.quantization.structs import WeightQuantizerInfo


class QuantizationShareBuildTimeInfo:
    def __init__(self, aq_potential_num: int, wq_potential_num: int):
        self.aq_potential_num = aq_potential_num
        self.wq_potential_num = wq_potential_num

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {"aq_potential_num": self.aq_potential_num, "wq_potential_num": self.wq_potential_num}

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "QuantizationShareBuildTimeInfo":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        return cls(**state)


class PTQuantizationStatisticsCollector(QuantizationStatisticsCollector):
    """
    Implementation of the quantization statistics collector for the PyTorch backend.
    """

    def __init__(
        self,
        weight_quantizers: Dict[WeightQuantizerId, WeightQuantizerInfo],
        non_weight_quantizers: Dict[NonWeightQuantizerId, NonWeightQuantizerInfo],
        build_time_info: QuantizationShareBuildTimeInfo,
    ):
        """
        Initializes a collector of the quantization statistics.
        """
        self._weight_quantizers = {k: v.quantizer_module_ref for k, v in weight_quantizers.items()}
        self._non_weight_quantizers = {k: v.quantizer_module_ref for k, v in non_weight_quantizers.items()}
        self._info = build_time_info

    def _collect_quantizers_descriptions(self) -> List[QuantizerDescription]:
        """
        Collects descriptions of the quantizers.

        :return: Descriptions of the quantizers.
        """
        # `True` for weight quantizer, `False` otherwise.
        quantizers = chain(
            map(lambda x: (True, x), self._weight_quantizers.values()),
            map(lambda x: (False, x), self._non_weight_quantizers.values()),
        )

        quantizers_descriptions = []
        for is_weight_quantizer, q in quantizers:
            is_symmetric = isinstance(q, SymmetricQuantizer)

            quantizers_descriptions.append(
                QuantizerDescription(
                    q.num_bits, q.per_channel, q.signed, is_symmetric, is_weight_quantizer, q.is_enabled_quantization()
                )
            )

        return quantizers_descriptions

    def _get_potential_quantizers_num(self) -> Tuple[int, int]:
        """
        Returns a potential number of quantizers for weights and activations.

        :return: A tuple (wq_potential_num, aq_potential_num) where
            - `wq_potential_num` is a potential number of quantizers for weights.
            - `aq_potential_num` is a potential number of quantizers for activations.
        """
        aq_potential_num = self._info.aq_potential_num if is_debug() else None
        return self._info.wq_potential_num, aq_potential_num


class MemoryConsumptionStatisticsCollector(StatisticsCollector):
    """
    This metric considers:
        - how many times memory consumption for network weights will decrease.
        - how many times memory consumption* for activations tensor will decrease.

    * Reflects host memory consumption, assuming only the final low-precision output activation tensors are stored
      in host memory (i.e. assuming intermediate accumulation results are only stored in device memory)
    """

    def __init__(
        self,
        compressed_model: NNCFNetwork,
        weight_quantizers: Dict[WeightQuantizerId, WeightQuantizerInfo],
        non_weight_quantizers: Dict[NonWeightQuantizerId, NonWeightQuantizerInfo],
    ):
        """
        Initializes collector of the memory consumption statistics.
        """
        self._compressed_model = compressed_model
        self._weight_quantizers = weight_quantizers
        self._non_weight_quantizers = non_weight_quantizers

    def collect(self) -> MemoryConsumptionStatistics:
        stats = MemoryConsumptionStatistics()

        fp_num_bits = 32
        nncf_modules = self._compressed_model.nncf.get_nncf_modules()
        for nncf_module in nncf_modules:
            count_el = np.prod(nncf_module.weight.shape)
            stats.fp32_weight_size += count_el * fp_num_bits
            quantizer = self._get_weight_quantizer_for_module(nncf_module)
            if quantizer is not None:
                num_bits = quantizer.num_bits
                stats.quantized_weight_size += count_el * num_bits
            else:
                stats.quantized_weight_size += count_el * fp_num_bits

        try:
            stats.weight_memory_consumption_decrease = stats.fp32_weight_size / stats.quantized_weight_size
        except ZeroDivisionError:
            stats.weight_memory_consumption_decrease = 0

        stats.quantized_weight_size /= 2**23
        stats.fp32_weight_size /= 2**23

        original_graph = deepcopy(self._compressed_model.nncf.get_original_graph())

        memory_consumption_fp_model = {}
        memory_consumption_compressed_model = {}

        original_nx_graph = original_graph._nx_graph
        nx.set_edge_attributes(original_nx_graph, 32, "precision")

        for u, v in original_nx_graph.edges:
            shape = original_nx_graph.edges[u, v][NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR]
            num_bits = self._get_precision_for_activation_tensor(u, v, original_nx_graph)
            original_nx_graph.edges[u, v]["precision"] = num_bits
            u_node_name = original_nx_graph.nodes[u][NNCFNode.NODE_NAME_ATTR]
            memory_consumption_fp_model[u_node_name] = np.prod(shape) * fp_num_bits
            memory_consumption_compressed_model[u_node_name] = np.prod(shape) * num_bits
        try:
            stats.max_fp32_activation_size = max(memory_consumption_fp_model.values()) / 2**23
            stats.max_compressed_activation_size = max(memory_consumption_compressed_model.values()) / 2**23
        except ValueError:
            stats.max_fp32_activation_size = 0
            stats.max_compressed_activation_size = 0
        return stats

    def _get_precision_for_activation_tensor(self, u_node: str, v_node: str, original_nx_graph: nx.DiGraph) -> int:
        pred_u_nodes = original_nx_graph._pred[u_node]
        precision_enter_activation_tensor = max(
            [0] + [original_nx_graph.edges[pred_u_node, u_node]["precision"] for pred_u_node in pred_u_nodes]
        )
        u_node_name = original_nx_graph.nodes[u_node][NNCFNode.NODE_NAME_ATTR]
        module = self._compressed_model.nncf.get_containing_module(u_node_name)
        if is_nncf_module(module):
            quantizer = self._get_weight_quantizer_for_module(module)
            if quantizer is not None:
                precision = max(quantizer.num_bits, precision_enter_activation_tensor)
            else:
                precision = 32
            return precision

        for aq_id, aq in self._non_weight_quantizers.items():
            if u_node_name == aq_id.target_node_name:
                precision = aq.quantizer_module_ref.num_bits
                break
        else:
            precision = precision_enter_activation_tensor
        return precision

    def _get_weight_quantizer_for_module(self, module: torch.nn.Module) -> Optional[BaseQuantizer]:
        for wq_info in self._weight_quantizers.values():
            if wq_info.quantized_module is module:
                return wq_info.quantizer_module_ref
        return None


class ShareEdgesQuantizedDataPathStatisticsCollector(StatisticsCollector):
    """
    This metric calculates the percentage of quantized edges relative to the total number of edges
    in the original network graph. "Quantized edge" is an edge representing a quantized activation tensor.
    """

    QUANTIZED_EDGES_ATTR = "quantized"
    PASSED_EDGES_ATTR = "passed"
    NODES_GRAPH_ATTR = "nodes"
    IS_MERGED_GRAPH_ATTR = "is_merged"

    def __init__(
        self, compressed_model: NNCFNetwork, qctrl: "QuantizationController", target_device: TargetDevice  # noqa: F821
    ):  # noqa: E501, F821
        self._compressed_model = compressed_model
        self._qctrl = qctrl
        self.stats = QuantizationConfigurationStatistics(0, 0)
        self._target_device = target_device

    def collect(self) -> QuantizationConfigurationStatistics:
        merged_original_graph = self.get_merged_original_graph_with_patterns(
            self._compressed_model.nncf.get_original_graph()
        )
        self.stats.quantized_edges_in_cfg = 0
        nx.set_edge_attributes(merged_original_graph, False, self.QUANTIZED_EDGES_ATTR)
        nx.set_edge_attributes(merged_original_graph, False, self.PASSED_EDGES_ATTR)

        input_nodes = [node for node in merged_original_graph.nodes if len(merged_original_graph._pred[node]) == 0]
        queue = deque()
        for input_node in input_nodes:
            next_nodes = merged_original_graph._succ[input_node]
            for next_node_key in next_nodes:
                edge = merged_original_graph.edges[input_node, next_node_key]
                edge[self.PASSED_EDGES_ATTR] = True
                edge[self.QUANTIZED_EDGES_ATTR] = True
                self.stats.quantized_edges_in_cfg += 1
                queue.appendleft(next_node_key)
        visited_nodes = {}

        while len(queue) != 0:
            node_key = queue.pop()
            if node_key in visited_nodes:
                continue
            if self._all_enter_edges_in_node_of_type(merged_original_graph, node_key, self.PASSED_EDGES_ATTR):
                visited_nodes[node_key] = True
                node = merged_original_graph.nodes[node_key]
                if node[self.IS_MERGED_GRAPH_ATTR]:
                    last_node = node[self.NODES_GRAPH_ATTR][-1]
                    node_name = str(last_node[NNCFNode.NODE_NAME_ATTR])
                    matched = False
                    for aq_info in self._qctrl.non_weight_quantizers.values():
                        for target_point in aq_info.affected_insertions:
                            if node_name == target_point.target_node_name:
                                matched = True
                                break
                    if matched:
                        self._marking_edges(merged_original_graph, node_key, queue)
                    else:
                        self._marking_edges(merged_original_graph, node_key, queue, False)
                else:
                    node_name = str(node[NNCFNode.NODE_NAME_ATTR])

                    matched = False
                    for aq_key in self._compressed_model.nncf.external_quantizers:
                        if node_name in aq_key:
                            matched = True
                            break
                    if matched:
                        self._marking_edges(merged_original_graph, node_key, queue)
                    else:
                        is_op_non_change_precision_activation_tensor = True
                        node_metatype = node[NNCFNode.METATYPE_ATTR]
                        is_op_non_change_precision_activation_tensor = (
                            node_metatype not in DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT[QuantizationTrait.INPUTS_QUANTIZABLE]
                        )

                        status = is_op_non_change_precision_activation_tensor and self._all_enter_edges_in_node_of_type(
                            merged_original_graph, node_key, self.QUANTIZED_EDGES_ATTR
                        )
                        self._marking_edges(merged_original_graph, node_key, queue, status)
            else:
                queue.appendleft(node_key)
        self.num_merged_original_graph_edges = len(merged_original_graph.edges)
        self.stats.total_edges_in_cfg = self.num_merged_original_graph_edges
        return self.stats

    def _all_enter_edges_in_node_of_type(self, graph, node_key, type_edge):
        prev_nodes = graph._pred[node_key]
        retval = True
        for prev_node_key in prev_nodes:
            edge = graph.edges[prev_node_key, node_key]
            if not edge[type_edge]:
                retval = False
                break
        return retval

    def _marking_edges(self, graph, node_key, queue, mark=True):
        next_nodes = graph._succ[node_key]
        for next_node_key in next_nodes:
            edge = graph.edges[node_key, next_node_key]
            edge[self.QUANTIZED_EDGES_ATTR] = mark
            edge[self.PASSED_EDGES_ATTR] = True
            queue.appendleft(next_node_key)
            if mark:
                self.stats.quantized_edges_in_cfg += 1

    def get_merged_original_graph_with_patterns(self, original_graph: PTNNCFGraph):
        pattern = PatternsManager.get_full_hw_pattern_graph(backend=BackendType.TORCH, device=self._target_device)

        matches = find_subgraphs_matching_pattern(original_graph._nx_graph, pattern)
        merged_graph = deepcopy(original_graph._nx_graph)
        nx.set_node_attributes(merged_graph, False, self.IS_MERGED_GRAPH_ATTR)
        for match in matches:
            if len(match) == 1:
                continue

            input_node_key = match[0]
            output_node_key = match[-1]
            in_edges = list(merged_graph.in_edges(input_node_key))
            out_edges = list(merged_graph.out_edges(output_node_key))

            in_edge_copies_dict = {}
            for in_edge_key in in_edges:
                in_edge_copies_dict[in_edge_key] = deepcopy(merged_graph.edges[in_edge_key])
            out_edge_copies_dict = {}
            for out_edge_key in out_edges:
                out_edge_copies_dict[out_edge_key] = deepcopy(merged_graph.edges[out_edge_key])

            merged_node_key = ""
            merged_nodes = []
            for node_key in match:
                merged_node_key += node_key + "\n"

                merged_nodes.append(original_graph._nx_graph.nodes[node_key])
                merged_graph.remove_node(node_key)
            merged_node_attrs = {
                NNCFNode.KEY_NODE_ATTR: merged_node_key,
                self.NODES_GRAPH_ATTR: merged_nodes,
                self.IS_MERGED_GRAPH_ATTR: True,
            }
            merged_graph.add_node(merged_node_key, **merged_node_attrs)
            for in_edge_key, in_edge_attrs in in_edge_copies_dict.items():
                merged_graph.add_edge(in_edge_key[0], merged_node_key, **in_edge_attrs)
            for out_edge_key, out_edge_attrs in out_edge_copies_dict.items():
                merged_graph.add_edge(merged_node_key, out_edge_key[1], **out_edge_attrs)

        return merged_graph

    @staticmethod
    def visualize_marked_graph(merged_original_graph):
        out_graph = nx.DiGraph()
        for node_key, _ in merged_original_graph.nodes.items():
            out_graph.add_node(node_key)
        for u, v in merged_original_graph.edges:
            edge = merged_original_graph.edges[u, v]
            if edge[ShareEdgesQuantizedDataPathStatisticsCollector.QUANTIZED_EDGES_ATTR]:
                attrs = {"color": "blue"}
            out_graph.add_edge(u, v, **attrs)
        return out_graph
