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
from collections import Counter
from collections import deque
from copy import deepcopy

import numpy as np
import networkx as nx

from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.nncf_network import NNCFNetwork, PTNNCFGraph
from nncf.torch.dynamic_graph.transform_graph import is_nncf_module
from nncf.torch.quantization.quantizer_propagation import DEFAULT_QUANT_TRAIT_TO_OP_DICT, QuantizationTrait
from nncf.torch.quantization.quantizer_id import WeightQuantizerId
from nncf.torch.quantization.quantizer_id import NonWeightQuantizerId
from nncf.torch.quantization.structs import WeightQuantizerInfo
from nncf.torch.quantization.structs import NonWeightQuantizerInfo
from nncf.common.collector import StatisticsCollector
from nncf.common.quantization.statistics import QuantizationShareStatistics
from nncf.common.quantization.statistics import QuantizersCounter
from nncf.common.quantization.statistics import BitwidthDistributionStatistics
from nncf.common.quantization.statistics import MemoryConsumptionStatistics
from nncf.common.quantization.statistics import QuantizationConfigurationStatistics


class QuantizationShareBuildTimeInfo:
    def __init__(self, aq_potential_num: int, wq_potential_num: int):
        self.aq_potential_num = aq_potential_num
        self.wq_potential_num = wq_potential_num


class QuantizationShareStatisticsCollector(StatisticsCollector):
    """
    This is a metric representing the share of the model that has been quantized.
    It includes the calculation of the following numbers:
    - Percentage of symmetric/asymmetric/per-channel/per-tensor weight quantizers relative
      to the number of placed weight quantizers
    - Percentage of symmetric/asymmetric/per-channel/per-tensor non weight quantizers relative
      to the number of placed non weight quantizers
    - Percentage of weight quantizers and non weight quantizers for each precision relative
      to the number potential* quantizers / placed quantizers
    Bitwidth distribution data is also collected.

    * The maximum possible number of potential quantizers depends on the presence of ignored
    scopes and the mode of quantizer setup that is used at the time of collecting the metric.
    """

    NAME_STR = 'quantization_share_statistics'

    def __init__(self,
                 weight_quantizers: Dict[WeightQuantizerId, WeightQuantizerInfo],
                 non_weight_quantizers: Dict[NonWeightQuantizerId, NonWeightQuantizerInfo],
                 build_time_info: QuantizationShareBuildTimeInfo):
        self._weight_quantizers = {k: v.quantizer_module_ref for k, v in weight_quantizers.items()}
        self._non_weight_quantizers = {k: v.quantizer_module_ref for k, v in non_weight_quantizers.items()}
        self._info = build_time_info

    def collect(self) -> QuantizationShareStatistics:
        """
        Collects quantization share statistics.
        """
        all_quantizers = {**self._weight_quantizers, **self._non_weight_quantizers}

        wq_counter = QuantizersCounter()
        aq_counter = QuantizersCounter()
        for qid, quantizer in all_quantizers.items():  # type: Tuple[QuantizerId, BaseQuantizer]
            counter = wq_counter if qid in self._weight_quantizers else aq_counter

            if quantizer.per_channel:
                counter.num_per_channel += 1
            else:
                counter.num_per_tensor += 1

            if quantizer.signed:
                counter.num_signed += 1
            else:
                counter.num_unsigned += 1

            if isinstance(quantizer, SymmetricQuantizer):
                counter.num_symmetric += 1
            else:
                counter.num_asymmetric += 1

        wq_total_num = len(self._weight_quantizers)
        aq_total_num = len(self._non_weight_quantizers)

        return QuantizationShareStatistics(wq_total_num, aq_total_num, self._info.wq_potential_num,
                                           self._info.aq_potential_num, wq_counter, aq_counter)


class BitwidthDistributionStatisticsCollector(StatisticsCollector):
    """
    Collects bit width distribution statistics.
    """

    NAME_STR = 'bitwidth_distribution_statistics'

    def __init__(self,
                 weight_quantizers: Dict[WeightQuantizerId, WeightQuantizerInfo],
                 non_weight_quantizers: Dict[NonWeightQuantizerId, NonWeightQuantizerInfo]):
        """
        Initializes collector of the bit width distribution statistics.
        """
        self._weight_quantizers = {k: v.quantizer_module_ref for k, v in weight_quantizers.items()}
        self._non_weight_quantizers = {k: v.quantizer_module_ref for k, v in non_weight_quantizers.items()}

    def collect(self) -> BitwidthDistributionStatistics:
        """
        Collects bit width distribution statistics.
        """
        all_quantizers = {**self._weight_quantizers, **self._non_weight_quantizers}
        wq_bitwidths = []
        aq_bitwidths = []
        for qid, quantizer in all_quantizers.items():
            if qid in self._weight_quantizers:
                wq_bitwidths.append(quantizer.num_bits)
            else:
                aq_bitwidths.append(quantizer.num_bits)

        return BitwidthDistributionStatistics(dict(Counter(wq_bitwidths)),
                                              dict(Counter(aq_bitwidths)))


class MemoryConsumptionStatisticsCollector(StatisticsCollector):
    """
    This metric considers:
        - how many times memory consumption for network weights will decrease.
        - how many times memory consumption* for activations tensor will decrease.

    * Reflects host memory consumption, assuming only the final low-precision output activation tensors are stored
      in host memory (i.e. assuming intermediate accumulation results are only stored in device memory)
    """

    NAME_STR = 'memory_consumption_statistics'

    def __init__(self,
                 compressed_model: NNCFNetwork,
                 weight_quantizers: Dict[WeightQuantizerId, WeightQuantizerInfo],
                 non_weight_quantizers: Dict[NonWeightQuantizerId, NonWeightQuantizerInfo]):
        """
        Initializes collector of the memory consumption statistics.
        """
        self._compressed_model = compressed_model
        self._weight_quantizers = {k: v.quantizer_module_ref for k, v in weight_quantizers.items()}
        self._non_weight_quantizers = {k: v.quantizer_module_ref for k, v in non_weight_quantizers.items()}

    def collect(self) -> MemoryConsumptionStatistics:
        stats = MemoryConsumptionStatistics()

        fp_num_bits = 32
        nncf_modules = self._compressed_model.get_nncf_modules()

        for scope_module, nncf_module in nncf_modules.items():
            count_el = np.prod(nncf_module.weight.shape)
            stats.fp32_weight_size += count_el * fp_num_bits
            status, quantizer = self._get_quantizer_for_scope(scope_module, self._weight_quantizers)
            if status:
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

        original_graph = deepcopy(self._compressed_model.get_original_graph())

        memory_consumption_fp_model = {}
        memory_consumption_compressed_model = {}
        # pylint: disable=protected-access
        original_nx_graph = original_graph._nx_graph
        nx.set_edge_attributes(original_nx_graph, 32, "precision")
        input_nodes = original_graph.get_input_nodes()
        input_node_keys = []
        for input_node in input_nodes:
            input_node_key = original_graph.get_node_key_by_id(input_node.node_id)
            input_node_keys.append(input_node_key)
            next_nodes = original_graph.get_next_nodes(input_node)
            for next_node in next_nodes:
                scope = next_node.ia_op_exec_context.scope_in_model
                status, quantizer = self._get_quantizer_for_scope(scope, self._non_weight_quantizers)
                if status:
                    next_node_key = original_graph.get_node_key_by_id(next_node.node_id)
                    num_bits = quantizer.num_bits
                    original_nx_graph.edges[input_node_key, next_node_key]['precision'] = num_bits

        for u, v in original_nx_graph.edges:
            if u in input_node_keys:
                continue

            shape = original_nx_graph.edges[u, v][PTNNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR]
            u_node_scope_str = str(original_nx_graph.nodes[u][PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR])
            num_bits = self._get_precision_for_activation_tensor(u, v, original_nx_graph)
            original_nx_graph.edges[u, v]['precision'] = num_bits
            memory_consumption_fp_model[u_node_scope_str] = np.prod(shape) * fp_num_bits
            memory_consumption_compressed_model[u_node_scope_str] = np.prod(shape) * num_bits

        try:
            stats.max_fp32_activation_size = max(memory_consumption_fp_model.values()) / 2**23
            stats.max_compressed_activation_size = max(memory_consumption_compressed_model.values()) / 2**23
        except ValueError:
            stats.max_fp32_activation_size = 0
            stats.max_compressed_activation_size = 0
        return stats

    def _get_precision_for_activation_tensor(self, u_node, v_node, original_nx_graph):
        scope_u_node = original_nx_graph.nodes[u_node][PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR].scope_in_model
        # pylint: disable=protected-access
        pred_u_nodes = original_nx_graph._pred[u_node]
        precision_enter_activation_tensor =\
             max([0] + [original_nx_graph.edges[pred_u_node, u_node]['precision'] for pred_u_node in pred_u_nodes])
        module = self._compressed_model.get_module_by_scope(scope_u_node)
        if is_nncf_module(module):
            status, quantizer = self._get_quantizer_for_scope(scope_u_node, self._weight_quantizers)
            if status:
                precision = max(quantizer.num_bits, precision_enter_activation_tensor)
            else:
                precision = 32
            return precision

        u_node_scope_str = str(original_nx_graph.nodes[u_node][PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR])
        for aq_id, aq in self._non_weight_quantizers.items():
            if u_node_scope_str in str(aq_id.ia_op_exec_context):
                precision = aq.num_bits
                break
        else:
            precision = precision_enter_activation_tensor
        return precision

    def _get_quantizer_for_scope(self, scope, quatizers):
        for quantizer_id, quantizer in quatizers.items():
            if quantizer_id.get_scope() == scope:
                return True, quantizer
        return False, None


class ShareEdgesQuantizedDataPathStatisticsCollector(StatisticsCollector):
    """
    This metric calculates the percentage of quantized edges relative to the total number of edges
    in the original network graph. "Quantized edge" is an edge representing a quantized activation tensor.
    """

    NAME_STR = 'quantization_configuration_statistics'
    QUANTIZED_EDGES_ATTR = 'quantized'
    PASSED_EDGES_ATTR = 'passed'
    NODES_GRAPH_ATTR = 'nodes'
    IS_MERGED_GRAPH_ATTR = 'is_merged'

    def __init__(self, compressed_model: NNCFNetwork, qctrl: 'QuantizationController'):
        self._compressed_model = compressed_model
        self._qctrl = qctrl
        self.stats = QuantizationConfigurationStatistics(0, 0)

    def collect(self) -> QuantizationConfigurationStatistics:
        # pylint: disable=too-many-branches
        merged_original_graph =\
            self.get_merged_original_graph_with_patterns(self._compressed_model.get_original_graph())
        self.stats.quantized_edges_in_cfg = 0
        nx.set_edge_attributes(merged_original_graph, False, self.QUANTIZED_EDGES_ATTR)
        nx.set_edge_attributes(merged_original_graph, False, self.PASSED_EDGES_ATTR)
        # pylint: disable=protected-access
        input_nodes = [node for node in merged_original_graph.nodes if len(merged_original_graph._pred[node]) == 0]
        queue = deque()
        for input_node in input_nodes:
            # pylint: disable=protected-access
            next_nodes = merged_original_graph._succ[input_node]
            for next_node_key in next_nodes:
                edge = merged_original_graph.edges[input_node, next_node_key]
                edge[self.PASSED_EDGES_ATTR] = True
                edge[self.QUANTIZED_EDGES_ATTR] = True
                self.stats.quantized_edges_in_cfg += 1
                queue.appendleft(next_node_key)
        visited_nodes = {}
        #pylint: disable=too-many-nested-blocks
        while len(queue) != 0:
            node_key = queue.pop()
            if node_key in visited_nodes:
                continue
            if self._all_enter_edges_in_node_of_type(merged_original_graph, node_key, self.PASSED_EDGES_ATTR):
                visited_nodes[node_key] = True
                node = merged_original_graph.nodes[node_key]
                if node[self.IS_MERGED_GRAPH_ATTR]:
                    last_node = node[self.NODES_GRAPH_ATTR][-1]
                    scope_str = str(last_node[PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR])
                    matched = False
                    for aq_info in self._qctrl.non_weight_quantizers.values():
                        for target_point in aq_info.affected_insertions:
                            if scope_str in str(target_point.ia_op_exec_context):
                                matched = True
                                break
                    if matched:
                        self._marking_edges(merged_original_graph, node_key, queue)
                    else:
                        self._marking_edges(merged_original_graph, node_key, queue, False)
                else:
                    scope_str = str(node[PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR])

                    matched = False
                    for aq_key in self._compressed_model.external_quantizers.keys():
                        if scope_str in aq_key:
                            matched = True
                            break
                    if matched:
                        self._marking_edges(merged_original_graph, node_key, queue)
                    else:
                        is_op_non_change_precision_activation_tensor = True
                        node_op_name = node[PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR].operator_name
                        for op in DEFAULT_QUANT_TRAIT_TO_OP_DICT[QuantizationTrait.INPUTS_QUANTIZABLE]:
                            op_names = [op.name]
                            if op.torch_tensor_patch_spec is not None:
                                op_names = op.torch_tensor_patch_spec.underlying_function_names
                            if node_op_name in op_names:
                                is_op_non_change_precision_activation_tensor = False
                                break
                        status = is_op_non_change_precision_activation_tensor and\
                            self._all_enter_edges_in_node_of_type(merged_original_graph,\
                                node_key, self.QUANTIZED_EDGES_ATTR)
                        self._marking_edges(merged_original_graph, node_key, queue, status)
            else:
                queue.appendleft(node_key)
        self.num_merged_original_graph_edges = len(merged_original_graph.edges)
        self.stats.total_edges_in_cfg = self.num_merged_original_graph_edges
        return self.stats

    def _all_enter_edges_in_node_of_type(self, graph, node_key, type_edge):
        # pylint: disable=protected-access
        prev_nodes = graph._pred[node_key]
        retval = True
        for prev_node_key in prev_nodes:
            edge = graph.edges[prev_node_key, node_key]
            if not edge[type_edge]:
                retval = False
                break
        return retval

    def _marking_edges(self, graph, node_key, queue, mark=True):
        # pylint: disable=protected-access
        next_nodes = graph._succ[node_key]
        for next_node_key in next_nodes:
            edge = graph.edges[node_key, next_node_key]
            edge[self.QUANTIZED_EDGES_ATTR] = mark
            edge[self.PASSED_EDGES_ATTR] = True
            queue.appendleft(next_node_key)
            if mark:
                self.stats.quantized_edges_in_cfg += 1

    def get_merged_original_graph_with_patterns(self, original_graph: PTNNCFGraph):
        import nncf.torch.graph.patterns as p
        from nncf.torch.graph.graph_matching import search_all

        pattern = p.LINEAR_OPS + p.ANY_BN_ACT_COMBO | p.LINEAR_OPS + p.ELTWISE_UNIFORM_OPS
        # pylint: disable=protected-access
        matches, _ = search_all(original_graph._nx_graph, pattern)
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
                merged_node_key += node_key + '\n'
                # pylint: disable=protected-access
                merged_nodes.append(original_graph._nx_graph.nodes[node_key])
                merged_graph.remove_node(node_key)
            merged_node_attrs = {
                PTNNCFGraph.KEY_NODE_ATTR: merged_node_key,
                self.NODES_GRAPH_ATTR: merged_nodes,
                self.IS_MERGED_GRAPH_ATTR: True
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
