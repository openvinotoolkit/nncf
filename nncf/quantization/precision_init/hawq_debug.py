"""
 Copyright (c) 2020-2021 Intel Corporation
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
from collections import OrderedDict
from pathlib import Path
from typing import Dict
from typing import List

import networkx as nx
import os
import torch
from torch import Tensor

from nncf.common.utils.logger import logger as nncf_logger
from nncf.graph.graph import PTNNCFGraph
from nncf.layers import NNCFConv2d
from nncf.nncf_network import ExtraCompressionModuleType
from nncf.nncf_network import NNCFNetwork
from nncf.quantization.adjust_padding import add_adjust_padding_nodes
from nncf.quantization.layers import QUANTIZATION_MODULES
from nncf.quantization.precision_init.adjacent_quantizers import GroupsOfAdjacentQuantizers
from nncf.quantization.precision_init.perturbations import PerturbationObserver
from nncf.quantization.precision_init.perturbations import Perturbations
from nncf.quantization.precision_init.traces_order import TracesPerLayer
from nncf.quantization.quantizer_id import NonWeightQuantizerId
from nncf.utils import get_all_modules_by_type


class HAWQDebugger:
    def __init__(self,
                 weight_qconfig_sequences_in_trace_order: List['QConfigSequenceForHAWQToEvaluate'],
                 perturbations: Perturbations,
                 weight_observers_for_each_covering_configuration: List[List[PerturbationObserver]],
                 traces_per_layer: TracesPerLayer,
                 bitwidths: List[int]):
        self._weight_qconfig_sequences_in_trace_order = weight_qconfig_sequences_in_trace_order
        self._num_weights = len(traces_per_layer.traces_order)
        self._perturbations = perturbations

        from nncf.debug import DEBUG_LOG_DIR
        self._dump_dir = Path(DEBUG_LOG_DIR) / Path("hawq_dumps")
        self._dump_dir.mkdir(parents=True, exist_ok=True)

        self._traces_order = traces_per_layer.traces_order
        self._traces_per_layer = traces_per_layer.get_all()

        num_of_weights = []
        norm_of_weights = []
        for i in range(self._num_weights):
            trace_index = self._traces_order.get_execution_index_by_traces_index(i)
            num_of_weights.append(weight_observers_for_each_covering_configuration[0][trace_index].get_numels())
            norm_of_weights.append(weight_observers_for_each_covering_configuration[0][trace_index].get_input_norm())
        self._num_weights_per_layer = torch.Tensor(num_of_weights)
        self._norm_weights_per_layer = torch.Tensor(norm_of_weights)

        bits_in_megabyte = 2 ** 23
        self._model_sizes = []
        for qconfig_sequence in self._weight_qconfig_sequences_in_trace_order:
            size = torch.sum(torch.Tensor([qconfig.num_bits for qconfig in qconfig_sequence]) *
                             self._num_weights_per_layer).item() / bits_in_megabyte
            self._model_sizes.append(size)
        self._bitwidths = bitwidths

    @staticmethod
    def get_all_quantizers_per_full_scope(model):
        all_quantizations = OrderedDict()
        for class_type in QUANTIZATION_MODULES.registry_dict.values():
            quantization_type = class_type.__name__
            all_quantizations.update(
                get_all_modules_by_type(
                    model.get_compression_modules_by_type(ExtraCompressionModuleType.EXTERNAL_QUANTIZER),
                    quantization_type))
            all_quantizations.update(get_all_modules_by_type(model.get_nncf_wrapped_model(), quantization_type))
        all_quantizations = OrderedDict(sorted(all_quantizations.items(), key=lambda x: str(x[0])))
        return all_quantizations

    @staticmethod
    def _paint_activation_quantizer_node(nncf_graph: PTNNCFGraph,
                                         quantizer_id: NonWeightQuantizerId,
                                         quantizer_info: 'NonWeightQuantizerInfo',
                                         bitwidth_color_map: Dict[int, str],
                                         groups_of_adjacent_quantizers: GroupsOfAdjacentQuantizers):
        # pylint:disable=too-many-branches
        affected_insertion_points_list = quantizer_info.affected_insertions  # type: List[PTInsertionPoint]

        for insertion_point in affected_insertion_points_list:
            input_agnostic_op_exec_context = insertion_point.ia_op_exec_context
            affected_nncf_node_key = nncf_graph.get_node_key_by_iap_context(input_agnostic_op_exec_context)
            affected_nx_node = nncf_graph.get_nx_node_by_key(affected_nncf_node_key)
            node_id = affected_nx_node[PTNNCFGraph.ID_NODE_ATTR]

            affected_nncf_node = nncf_graph.get_node_by_id(node_id)

            in_port_id = insertion_point.input_port_id

            if in_port_id is None:
                # Post-hooking used for activation quantization
                # Currently only a single post-hook can immediately follow an operation
                succs = list(nncf_graph.get_successors(affected_nncf_node_key))
                assert len(succs) == 1
                target_nncf_node_key = succs[0]
            else:
                # Pre-hooking used for activation quantization
                previous_nodes = nncf_graph.get_previous_nodes(affected_nncf_node)
                target_node = None
                for prev_node in previous_nodes:
                    prev_edge = nncf_graph.get_nx_edge(prev_node, affected_nncf_node)
                    if prev_edge[PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR] == in_port_id:
                        target_node = prev_node
                        break

                assert target_node is not None, "Could not find a pre-hook quantizer node for a specific " \
                                                "input port!"
                target_nncf_node_id = target_node.node_id
                target_nncf_node_key = nncf_graph.get_node_key_by_id(target_nncf_node_id)

            activation_fq_node = nncf_graph.get_nx_node_by_key(target_nncf_node_key)
            bitwidth = quantizer_info.quantizer_module_ref.num_bits
            activation_fq_node['color'] = bitwidth_color_map[bitwidth]
            activation_fq_node['style'] = 'filled'
            node_id = activation_fq_node[PTNNCFGraph.ID_NODE_ATTR]

            activation_fq_node['label'] = 'AFQ_[{}]_#{}'.format(
                quantizer_info.quantizer_module_ref.get_quantizer_config(),
                str(node_id))
            grouped_mode = bool(groups_of_adjacent_quantizers)
            if grouped_mode:
                group_id_str = 'UNDEFINED'
                group_id = groups_of_adjacent_quantizers.get_group_id_for_quantizer(quantizer_id)
                if node_id is None:
                    nncf_logger.error('No group for activation quantizer: {}'.format(target_nncf_node_key))
                else:
                    group_id_str = str(group_id)
                activation_fq_node['label'] += "_G" + group_id_str

    @staticmethod
    def get_bitwidth_graph(algo_ctrl, model,
                           groups_of_adjacent_quantizers: GroupsOfAdjacentQuantizers,
                           add_flops=False) -> PTNNCFGraph:
        # Overwrites nodes that were obtained during graph tracing and correspond to quantizer
        # nodes with the nodes whose 'label' attribute is set to a more display-friendly representation
        # of the quantizer's bitwidth.
        # pylint:disable=too-many-branches
        if add_flops:
            flops_per_module = model.get_flops_per_module()
        grouped_mode = bool(groups_of_adjacent_quantizers)
        nncf_graph = model.get_graph()
        for node_key in nncf_graph.get_all_node_keys():
            node = nncf_graph.get_nx_node_by_key(node_key)
            color = ''
            if node[PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR]:
                operator_name = node[PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR].operator_name
                quantized_module_scope = node[PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR].scope_in_model
                module = model.get_module_by_scope(quantized_module_scope)
                if isinstance(module, NNCFConv2d):
                    color = 'lightblue'
                    if module.groups == module.in_channels and module.in_channels > 1:
                        operator_name = 'DW_Conv2d'
                        color = 'purple'
                    kernel_size = 'x'.join(map(str, module.kernel_size))
                    operator_name += f'_k{kernel_size}'
                    padding_values = set(module.padding)
                    padding_enabled = len(padding_values) >= 1 and padding_values.pop()
                    if padding_enabled:
                        operator_name += '_PAD'
                    if add_flops:
                        operator_name += f'_FLOPS:{str(flops_per_module[quantized_module_scope])}'
                operator_name += '_#{}'.format(str(node[PTNNCFGraph.ID_NODE_ATTR]))
                node['label'] = operator_name
                node['style'] = 'filled'
                if color:
                    node['color'] = color

        non_weight_quantizers = algo_ctrl.non_weight_quantizers
        bitwidth_color_map = {2: 'purple', 4: 'red', 8: 'green', 6: 'orange'}
        for quantizer_id, quantizer_info in non_weight_quantizers.items():
            HAWQDebugger._paint_activation_quantizer_node(nncf_graph, quantizer_id,
                                                          quantizer_info, bitwidth_color_map,
                                                          groups_of_adjacent_quantizers)
        for wq_id, wq_info in algo_ctrl.weight_quantizers.items():
            quantized_module_scope = wq_id.get_scope()
            quantizer = wq_info.quantizer_module_ref

            nodes = nncf_graph.get_op_nodes_in_scope(quantized_module_scope)
            if not nodes:
                raise AttributeError('Failed to get any nodes by scope={}'.format(str(quantized_module_scope)))
            wq_nodes = []
            for pot_wq_node in nodes:
                if 'UpdateWeight' in str(pot_wq_node[PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR]):
                    wq_nodes.append(pot_wq_node)
            assert len(wq_nodes) == 1

            node = wq_nodes[0]
            bitwidths = quantizer.num_bits
            node_id = node[PTNNCFGraph.ID_NODE_ATTR]
            node['label'] = 'WFQ_[{}]_#{}'.format(quantizer.get_quantizer_config(), str(node_id))
            if grouped_mode:
                group_id_str = 'UNDEFINED'
                group_id = groups_of_adjacent_quantizers.get_group_id_for_quantizer(wq_id)
                if group_id is None:
                    nncf_logger.error('No group for weight quantizer: {}'.format(quantized_module_scope))
                else:
                    group_id_str = str(group_id)
                node['label'] += '_G' + group_id_str
            node['color'] = bitwidth_color_map[bitwidths]
            node['style'] = 'filled'
        return nncf_graph

    def dump_avg_traces(self):
        import matplotlib.pyplot as plt
        dump_file = os.path.join(self._dump_dir, 'avg_traces_per_layer')
        torch.save(self._traces_per_layer, dump_file)
        fig = plt.figure()
        fig.suptitle('Average Hessian Trace')
        ax = fig.add_subplot(2, 1, 1)
        ax.set_yscale('log')
        ax.set_xlabel('weight quantizers')
        ax.set_ylabel('average hessian trace')
        ax.plot(self._traces_per_layer.cpu().numpy())
        plt.savefig(dump_file)

    def dump_metric_MB(self, metric_per_qconfig_sequence: List[Tensor]):
        import matplotlib.pyplot as plt
        list_to_plot = [cm.item() for cm in metric_per_qconfig_sequence]
        fig = plt.figure()
        fig.suptitle('Pareto Frontier')
        ax = fig.add_subplot(2, 1, 1)
        ax.set_yscale('log')
        ax.set_xlabel('Model Size (MB)')
        ax.set_ylabel('Metric value (total perturbation)')
        ax.scatter(self._model_sizes, list_to_plot, s=20, facecolors='none', edgecolors='r')
        cm = torch.Tensor(metric_per_qconfig_sequence)
        cm_m = cm.median().item()
        qconfig_index = metric_per_qconfig_sequence.index(cm_m)
        ms_m = self._model_sizes[qconfig_index]
        ax.scatter(ms_m, cm_m, s=30, facecolors='none', edgecolors='b', label='median from all metrics')
        ax.legend()
        plt.savefig(os.path.join(self._dump_dir, 'Pareto_Frontier'))
        nncf_logger.info(
            'Distribution of HAWQ metrics: min_value={:.3f}, max_value={:.3f}, median_value={:.3f}, '
            'median_index={}, total_number={}'.format(cm.min().item(), cm.max().item(), cm_m,
                                                      qconfig_index,
                                                      len(metric_per_qconfig_sequence)))

    def dump_metric_flops(self, metric_per_qconfig_sequence: List[Tensor], flops_per_config: List[float],
                          choosen_qconfig_index: int):
        import matplotlib.pyplot as plt
        list_to_plot = [cm.item() for cm in metric_per_qconfig_sequence]
        fig = plt.figure()
        fig.suptitle('Pareto Frontier')
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Compression ratio: total INT8 Bits Complexity / total MIXED INT Bits Complexity')
        ax.set_ylabel('Metric value (total perturbation)')
        ax.scatter(flops_per_config, list_to_plot, s=10, alpha=0.3)  # s=20, facecolors='none', edgecolors='r')
        flops_per_config = [torch.Tensor([v]) for v in flops_per_config]
        cm = torch.Tensor(flops_per_config)
        cm_m = cm.median().item()
        configuration_index = flops_per_config.index(cm_m)
        ms_m = metric_per_qconfig_sequence[configuration_index].item()
        ax.scatter(cm_m, ms_m, s=30, facecolors='none', edgecolors='b', label='median from all metrics')
        cm_c = metric_per_qconfig_sequence[choosen_qconfig_index].item()
        fpc_c = flops_per_config[choosen_qconfig_index].item()
        ax.scatter(fpc_c, cm_c, s=30, facecolors='none', edgecolors='r', label='chosen config')

        ax.legend()
        plt.savefig(os.path.join(self._dump_dir, 'Pareto_Frontier_compress_ratio'))

    def dump_density_of_quantization_noise(self):
        noise_per_config = []  # type: List[Tensor]
        for qconfig_sequence in self._weight_qconfig_sequences_in_trace_order:
            qnoise = 0
            for i in range(self._num_weights):
                execution_index = self._traces_order.get_execution_index_by_traces_index(i)
                qnoise += self._perturbations.get(layer_id=execution_index, qconfig=qconfig_sequence[i])
            noise_per_config.append(qnoise)

        list_to_plot = [cm.item() for cm in noise_per_config]
        import matplotlib.pyplot as plt
        fig = plt.figure()
        fig.suptitle('Density of quantization noise')
        ax = fig.add_subplot(2, 1, 1)
        ax.set_yscale('log')
        ax.set_xlabel('Blocks')
        ax.set_ylabel('Noise value')
        ax.scatter(self._model_sizes, list_to_plot, s=20, alpha=0.3)
        ax.legend()
        plt.savefig(os.path.join(self._dump_dir, 'Density_of_quantization_noise'))

    def dump_perturbations_ratio(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        fig.suptitle('Quantization noise vs Average Trace')
        ax = fig.add_subplot(2, 1, 1)
        ax.set_xlabel('Blocks')
        ax.set_yscale('log')
        perturbations_per_layer_id = list(self._perturbations.get_all().values())
        perturb = []
        max_bitwidths = []
        for perturbations_for_all_observed_qconfig_sequence_in_current_layer in perturbations_per_layer_id:
            qconfig_sequence = perturbations_for_all_observed_qconfig_sequence_in_current_layer.keys()
            max_bitwidth_qconfig = max(qconfig_sequence, key=lambda x: x.num_bits)
            perturb.append(perturbations_for_all_observed_qconfig_sequence_in_current_layer[max_bitwidth_qconfig])
            max_bitwidths.append(max_bitwidth_qconfig.num_bits)
        ax.plot(
            [p / m / n for p, m, n in zip(perturb, self._num_weights_per_layer, self._norm_weights_per_layer)],
            label='normalized n-bit noise')
        ax.plot(perturb, label='n-bit noise')
        ax.plot(max_bitwidths, label='n')
        ax.plot(self._traces_per_layer.cpu().numpy(), label='trace')
        ax.plot([n * p for n, p in zip(self._traces_per_layer, perturb)], label='trace * noise')
        ax.legend()
        plt.savefig(os.path.join(self._dump_dir, 'Quantization_noise_vs_Average_Trace'))

    def dump_bitwidth_graph(self, algo_ctrl: 'QuantizationController', model: NNCFNetwork,
                            groups_of_adjacent_quantizers: GroupsOfAdjacentQuantizers):
        nncf_graph = self.get_bitwidth_graph(algo_ctrl, model, groups_of_adjacent_quantizers)
        nx_graph = add_adjust_padding_nodes(nncf_graph, model)
        nx.drawing.nx_pydot.write_dot(nx_graph, self._dump_dir / Path('bitwidth_graph.dot'))
