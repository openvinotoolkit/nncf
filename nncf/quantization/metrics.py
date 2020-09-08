import numpy as np
import networkx as nx
from copy import deepcopy
from texttable import Texttable
from collections import deque

from nncf.quantization.layers import SymmetricQuantizer
from nncf.nncf_network import NNCFNetwork, NNCFGraph
from nncf.dynamic_graph.transform_graph import is_nncf_module
from nncf.quantization.quantizer_propagation import DEFAULT_QUANT_TRAIT_TO_OP_DICT, QuantizationTrait

class BaseMetric:
    def __init__(self):
        pass

    def collect(self):
        pass

    def get_metric_table(self):
        pass


class NetworkQuantizationShareMetric(BaseMetric):
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
    NAME_STR = 'NetworkQuantizationShare'

    WEIGHTS_RATIO_STR = ' WQs / All placed WQs' # WQ - weight quantizer
    ACTIVATIONS_RATIO_STR = ' AQs / All placed AQs' # AQ - activation quantizer
    TOTAL_RATIO_STR = ' Qs (out of total placed)'

    PARAMS_STR = 'Quantizer parameter'
    SYMMETRIC_STR = 'Symmetric'
    ASYMMETRIC_STR = 'Asymmetric'
    PER_CHANNEL_STR = 'Per-channel'
    SIGNED_STR = 'Signed'
    PER_TENSOR_STR = 'Per-tensor'
    UNSIGNED_STR = 'Unsigned'
    SHARE_WEIGHT_QUANTIZERS_STR = 'Placed WQs / Potential WQs'
    SHARE_ACTIVATION_QUANTIZERS_STR = 'Placed AQs / Potential AQs'

    def __init__(self, compressed_model, weights_quantizers, non_weights_quantizers, quantizer_setup_type):
        super().__init__()
        self._compressed_model = compressed_model
        self._quantizer_setup_type = quantizer_setup_type # type: QuantizerSetupType
        self.non_weights_quantizers = {k: v.quantizer_module_ref for k, v in non_weights_quantizers.items()}
        self.weights_quantizers = weights_quantizers
        self._all_quantizations = {**self.weights_quantizers, **self.non_weights_quantizers}
        self.header = [self.PARAMS_STR, self.WEIGHTS_RATIO_STR, self.ACTIVATIONS_RATIO_STR, self.TOTAL_RATIO_STR]
        self.params = {self.PER_CHANNEL_STR, self.PER_TENSOR_STR, self.UNSIGNED_STR, self.SIGNED_STR,
                       self.SYMMETRIC_STR, self.ASYMMETRIC_STR}
        self.params_bits_stat = set()
        self.num_potential_quantized_weights = len(compressed_model.get_nncf_modules())
        self.num_potential_quantized_activations = self._get_num_potential_quantized_activations()
        self.num_placed_weight_quantizers = len(self.weights_quantizers)
        self.num_placed_activation_quantizers = len(self.non_weights_quantizers)
        self.num_all_potential_quantizer = self.num_potential_quantized_weights +\
             self.num_potential_quantized_activations
        self.stat = {}
        self._ratio = {
            self.WEIGHTS_RATIO_STR: len(self.weights_quantizers),
            self.ACTIVATIONS_RATIO_STR: len(self.non_weights_quantizers),
            self.TOTAL_RATIO_STR: len(self._all_quantizations)}

    def _get_num_potential_quantized_activations(self):
        from nncf.quantization.algo import QuantizerSetupType
        retval = 0
        if self._quantizer_setup_type == QuantizerSetupType.PATTERN_BASED:
            from nncf.quantization.algo import QuantizationBuilder
            # pylint: disable=protected-access
            default_pattern = QuantizationBuilder._make_default_quantizable_subgraph_pattern()
            retval = len(self._compressed_model.get_post_pattern_insertion_points(default_pattern))
        else:
            from nncf.quantization.algo import QuantizerPropagationSolver
            insertion_point_graph = self._compressed_model.get_insertion_point_graph()
            prop_graph_solver = QuantizerPropagationSolver()
            insertion_data = prop_graph_solver.run_on_ip_graph(insertion_point_graph)
            retval = len(insertion_data)
        return retval

    def collect(self):
        for quantizer in self._all_quantizations.values():
            self.params_bits_stat.add(quantizer.num_bits)

        for h in self.header:
            self.stat[h] = {}
            for p in self.params:
                self.stat[h][p] = 0
            for p in self.params_bits_stat:
                self.stat[h][p] = 0

        for quantizer in self._all_quantizations.values():  # type: BaseQuantizer
            num_bits = quantizer.num_bits
            self.stat[self.TOTAL_RATIO_STR][num_bits] += 1
            type_ = self.WEIGHTS_RATIO_STR if quantizer.is_weights else self.ACTIVATIONS_RATIO_STR
            self.stat[type_][num_bits] += 1
            if quantizer.per_channel:
                self.stat[type_][self.PER_CHANNEL_STR] += 1
            else:
                self.stat[type_][self.PER_TENSOR_STR] += 1
            if quantizer.signed:
                self.stat[type_][self.SIGNED_STR] += 1
            else:
                self.stat[type_][self.UNSIGNED_STR] += 1
            if isinstance(quantizer, SymmetricQuantizer):
                self.stat[type_][self.SYMMETRIC_STR] += 1
            else:
                self.stat[type_][self.ASYMMETRIC_STR] += 1

    def _get_copy_statistics(self):
        statistics = deepcopy(self.stat)
        for h in self.header[1:]:
            for key, _ in statistics[h].items():
                try:
                    statistics[h][key] /= self._ratio[h]
                    statistics[h][key] *= 100
                except ZeroDivisionError:
                    statistics[h][key] = 0
        return statistics

    def get_metric_table(self):
        table_with_bits_stats = Texttable()
        table_with_other_stats = Texttable()
        data = [['Metric type', 'Value']]
        for h in (self.WEIGHTS_RATIO_STR, self.ACTIVATIONS_RATIO_STR):
            for p in self.params:
                try:
                    row = ['{} '.format(p) + str(h), '{:.2f} % ({} / {}) '.format(\
                        self.stat[h][p] / self._ratio[h] * 100, self.stat[h][p], self._ratio[h])]
                except ZeroDivisionError:
                    row = ['{} '.format(p) + h, 0]
                data.append(row)
        try:
            row = [self.SHARE_WEIGHT_QUANTIZERS_STR, '{:.2f} % ({} / {}) '.format(\
                   self.num_placed_weight_quantizers / self.num_potential_quantized_weights * 100,
                   self.num_placed_weight_quantizers, self.num_potential_quantized_weights)]
        except ZeroDivisionError:
            row = [self.SHARE_WEIGHT_QUANTIZERS_STR, '{} % '.format(0)]

        data.append(row)
        try:
            row = [self.SHARE_ACTIVATION_QUANTIZERS_STR, '{:.2f} % ({} / {}) '.format(\
                   self.num_placed_activation_quantizers / self.num_potential_quantized_activations * 100,
                   self.num_placed_activation_quantizers, self.num_potential_quantized_activations)]
        except ZeroDivisionError:
            row = [self.SHARE_ACTIVATION_QUANTIZERS_STR, '{} % '.format(0)]
        data.append(row)

        table_with_other_stats.add_rows(data)

        data = [['Num bits (N)', 'N-bits WQs / Placed WQs', 'N-bits AQs / Placed AQs', 'N-bits Qs / Placed Qs']]
        for p in self.params_bits_stat:
            row = [p]
            for h in (self.WEIGHTS_RATIO_STR, self.ACTIVATIONS_RATIO_STR, self.TOTAL_RATIO_STR):
                try:
                    row.append('{:.2f} % ({} / {}) '.format(\
                        self.stat[h][p] / self._ratio[h] * 100, self.stat[h][p], self._ratio[h]))
                except ZeroDivisionError:
                    row.append(0)
            data.append(row)
        table_with_bits_stats.add_rows(data)

        retval = {
            "Share quantization statistics:" : table_with_other_stats,
            "Bitwidth distribution:" : table_with_bits_stats
        }
        return retval

    def get_bits_stat(self):
        table = Texttable()
        data = [['Num bits (N)', 'N-bits WQs / Placed Qs', 'N-bits AQs / Placed Qs', 'N-bits Qs / Placed Qs']]
        for p in self.params_bits_stat:
            row = [p]
            for h in (self.WEIGHTS_RATIO_STR, self.ACTIVATIONS_RATIO_STR, self.TOTAL_RATIO_STR):
                try:
                    row.append(self.stat[h][p] / self._ratio[self.TOTAL_RATIO_STR] * 100)
                except ZeroDivisionError:
                    row.append(0)
            data.append(row)
        table.add_rows(data)
        return table

class MemoryСostMetric(BaseMetric):
    """

    This metric considers:
        - how many times memory consumption for network weights will decrease.
        - how many times memory consumption* for activations tensor will decrease.

    * Reflects host memory consumption, assuming only the final low-precision output activation tensors are stored
      in host memory (i.e. assuming intermediate accumulation results are only stored in device memory)

    """
    PARAMS_STR = 'params'
    NAME_STR = 'MemoryCost'

    EXPECTED_MEMORY_CONSUMPTION_DECREASE_STR = 'Memory consumption decrease for weights'
    SIZE_MEMORY_FP_WEIGHTS_STR = 'Memory consumption for full-precision weights'
    SIZE_MEMORY_COMPRESSED_WEIGHTS_STR = 'Memory consumption for quantized weights'
    MAX_MEMORY_CONSUMPTION_ACTIVATION_TENSOR_IN_FP32_MODEL_STR =\
         'Max memory consumption for an activation tensor in FP32 model'
    MAX_MEMORY_CONSUMPTION_ACTIVATION_TENSOR_IN_COMPRESSED_MODEL_STR =\
         'Max memory consumption for an activation tensor in compressed model'

    def __init__(self, compressed_model: NNCFNetwork, weights_quantizers, non_weight_quantizers):
        super().__init__()
        self._compressed_model = compressed_model
        self._weights_quantizers = weights_quantizers
        self._non_weight_quantizers = {k: v.quantizer_module_ref for k, v in non_weight_quantizers.items()}
        self.header = [self.EXPECTED_MEMORY_CONSUMPTION_DECREASE_STR, self.SIZE_MEMORY_FP_WEIGHTS_STR,\
             self.SIZE_MEMORY_COMPRESSED_WEIGHTS_STR,\
             self.MAX_MEMORY_CONSUMPTION_ACTIVATION_TENSOR_IN_FP32_MODEL_STR,\
             self.MAX_MEMORY_CONSUMPTION_ACTIVATION_TENSOR_IN_COMPRESSED_MODEL_STR]
        self.stat = {}

    def collect(self):
        self.stat[self.SIZE_MEMORY_FP_WEIGHTS_STR] = 0
        self.stat[self.SIZE_MEMORY_COMPRESSED_WEIGHTS_STR] = 0
        fp_num_bits = 32
        nncf_modules = self._compressed_model.get_nncf_modules()

        for scope_module, nncf_module in nncf_modules.items():
            count_el = np.prod(nncf_module.weight.shape)
            self.stat[self.SIZE_MEMORY_FP_WEIGHTS_STR] += count_el * fp_num_bits
            status, quantizer = self._get_quantizer_for_scope(scope_module, self._weights_quantizers)
            if status > 0:
                num_bits = quantizer.num_bits
                self.stat[self.SIZE_MEMORY_COMPRESSED_WEIGHTS_STR] += count_el * num_bits
            else:
                self.stat[self.SIZE_MEMORY_COMPRESSED_WEIGHTS_STR] += count_el * fp_num_bits
        try:
            self.stat[self.EXPECTED_MEMORY_CONSUMPTION_DECREASE_STR] = self.stat[self.SIZE_MEMORY_FP_WEIGHTS_STR] /\
             self.stat[self.SIZE_MEMORY_COMPRESSED_WEIGHTS_STR]
        except ZeroDivisionError:
            self.stat[self.EXPECTED_MEMORY_CONSUMPTION_DECREASE_STR] = 0
        self.stat[self.SIZE_MEMORY_COMPRESSED_WEIGHTS_STR] /= 2**23
        self.stat[self.SIZE_MEMORY_FP_WEIGHTS_STR] /= 2**23

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
                scope = next_node.op_exec_context.scope_in_model
                status, quantizer = self._get_quantizer_for_scope(scope, self._non_weight_quantizers)
                if status:
                    next_node_key = original_graph.get_node_key_by_id(next_node.node_id)
                    num_bits = quantizer.num_bits
                    original_nx_graph.edges[input_node_key, next_node_key]['precision'] = num_bits

        for u, v in original_nx_graph.edges:
            if u in input_node_keys:
                continue

            shape = original_nx_graph.edges[u, v][NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR]
            u_node_scope_str = str(original_nx_graph.nodes[u]['op_exec_context'].input_agnostic)
            num_bits = self.get_precision_for_activation_tensor(u, v, original_nx_graph)
            original_nx_graph.edges[u, v]['precision'] = num_bits
            memory_consumption_fp_model[u_node_scope_str] = np.prod(shape) * fp_num_bits
            memory_consumption_compressed_model[u_node_scope_str] = np.prod(shape) * num_bits
        try:
            self.stat[self.MAX_MEMORY_CONSUMPTION_ACTIVATION_TENSOR_IN_FP32_MODEL_STR] =\
                max(memory_consumption_fp_model.values()) / 2**23
            self.stat[self.MAX_MEMORY_CONSUMPTION_ACTIVATION_TENSOR_IN_COMPRESSED_MODEL_STR] =\
                max(memory_consumption_compressed_model.values()) / 2**23
        except ValueError:
            self.stat[self.MAX_MEMORY_CONSUMPTION_ACTIVATION_TENSOR_IN_FP32_MODEL_STR] = 0
            self.stat[self.MAX_MEMORY_CONSUMPTION_ACTIVATION_TENSOR_IN_COMPRESSED_MODEL_STR] = 0

    def get_precision_for_activation_tensor(self, u_node, v_node, original_nx_graph):
        scope_u_node = original_nx_graph.nodes[u_node][NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].scope_in_model
        # pylint: disable=protected-access
        pred_u_nodes = original_nx_graph._pred[u_node]
        precision_enter_activation_tensor =\
             max([0] + [original_nx_graph.edges[pred_u_node, u_node]['precision'] for pred_u_node in pred_u_nodes])
        module = self._compressed_model.get_module_by_scope(scope_u_node)
        if is_nncf_module(module):
            status, quantizer = self._get_quantizer_for_scope(scope_u_node, self._weights_quantizers)
            if status:
                precision = max(quantizer.num_bits, precision_enter_activation_tensor)
            else:
                precision = 32
            return precision

        u_node_scope_str = str(original_nx_graph.nodes[u_node][NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].input_agnostic)
        if u_node_scope_str in self._compressed_model.activation_quantizers:
            precision = self._compressed_model.activation_quantizers[u_node_scope_str].num_bits
        else:
            precision = precision_enter_activation_tensor
        return precision

    def _get_quantizer_for_scope(self, scope, quatizers):
        for quantizer_id, quantizer in quatizers.items():
            if quantizer_id.get_scope() == scope:
                return True, quantizer
        return False, None

    def get_metric_table(self):
        table = Texttable()
        data = [['Metric type', 'Value']]
        data.append([self.header[0], self.stat[self.header[0]]])
        for h in self.header[1:]:
            data.append([h + ' (Mbyte)', self.stat[h]])
        table.add_rows(data)

        retval = {"Memory consumption statistics:": table}
        return retval


class ShareEdgesQuantizedDataPath(BaseMetric):
    """

    This metric calculates the percentage of quantized edges relative to the total number of edges
    in the original network graph. "Quantized edge" is an edge representing a quantized activation tensor.

    """
    NAME_STR = 'ShareEdgesQuantizedDataPath'
    COUNT_QUANTIZED_EDGES_STR = 'Share edges of the quantized data path'
    QUANTIZED_EDGES_ATTR = 'quantized'
    PASSED_EDGES_ATTR = 'passed'
    NODES_GRAPH_ATTR = 'nodes'
    IS_MERGED_GRAPH_ATTR = 'is_merged'


    def __init__(self, compressed_model: NNCFNetwork):
        super().__init__()
        self._compressed_model = compressed_model
        self.stat = {}

    def collect(self):
        # pylint: disable=too-many-branches
        merged_original_graph =\
            self.get_merged_original_graph_with_patterns(self._compressed_model.get_original_graph())
        self.stat[self.COUNT_QUANTIZED_EDGES_STR] = 0
        self.header = [self.COUNT_QUANTIZED_EDGES_STR]
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
                self.stat[self.COUNT_QUANTIZED_EDGES_STR] += 1
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
                    scope_str = str(last_node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].input_agnostic)
                    if scope_str in self._compressed_model.activation_quantizers:
                        self._marking_edges(merged_original_graph, node_key, queue)
                    else:
                        self._marking_edges(merged_original_graph, node_key, queue, False)
                else:
                    scope_str = str(node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].input_agnostic)
                    if scope_str in self._compressed_model.activation_quantizers:
                        self._marking_edges(merged_original_graph, node_key, queue)
                    else:
                        is_op_non_change_precision_activation_tensor = True
                        node_op_name = node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].operator_name
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

    def _get_copy_statistics(self):
        statistics = deepcopy(self.stat)
        try:
            statistics[self.COUNT_QUANTIZED_EDGES_STR] /= self.num_merged_original_graph_edges
            statistics[self.COUNT_QUANTIZED_EDGES_STR] *= 100
        except ZeroDivisionError:
            statistics[self.COUNT_QUANTIZED_EDGES_STR] = 0

        return statistics

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
                self.stat[self.COUNT_QUANTIZED_EDGES_STR] += 1

    def get_metric_table(self):
        table = Texttable()
        data = [['Metric type', 'Value']]
        try:
            data.append([self.header[0], '{:.2f} % ({} / {})'.format(
                self.stat[self.COUNT_QUANTIZED_EDGES_STR] / self.num_merged_original_graph_edges * 100,
                self.stat[self.COUNT_QUANTIZED_EDGES_STR], self.num_merged_original_graph_edges)])
        except ZeroDivisionError:
            data.append([self.header[0], '{} % '.format(0)])
        table.add_rows(data)

        retval = {"Quantization configuration statistics:" : table}
        return retval

    def get_merged_original_graph_with_patterns(self, original_graph: NNCFGraph):
        import nncf.dynamic_graph.patterns as p
        from nncf.dynamic_graph.graph_matching import search_all

        pattern = p.LINEAR_OPS + p.ANY_BN_ACT_COMBO | p.LINEAR_OPS + p.ELTWISE_UNIFORM_OPS
        # pylint: disable=protected-access
        matches = search_all(original_graph._nx_graph, pattern)
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
                NNCFGraph.KEY_NODE_ATTR: merged_node_key,
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
            attrs = {"color": "black"}
            if edge[ShareEdgesQuantizedDataPath.QUANTIZED_EDGES_ATTR]:
                attrs = {"color": "blue"}
            out_graph.add_edge(u, v, **attrs)
        return out_graph
