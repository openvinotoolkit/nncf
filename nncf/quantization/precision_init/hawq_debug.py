"""
 Copyright (c) 2020 Intel Corporation
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
from typing import List

import os
import torch
from torch import Tensor

from nncf.nncf_logger import logger as nncf_logger
from nncf.nncf_network import CompressionModuleType
from nncf.quantization.layers import QUANTIZATION_MODULES
from .hawq_init import GroupsOfAdjacentQuantizers
from .perturbations import Perturbations, PerturbationObserver
from .traces_order import TracesPerLayer
from ...dynamic_graph.graph import NNCFGraph
from ...layers import NNCFConv2d
from ...utils import get_all_modules_by_type


class HAWQDebugger:
    def __init__(self, bits_configurations: List[List[int]],
                 perturbations: Perturbations,
                 weight_observers: List[PerturbationObserver],
                 traces_per_layer: TracesPerLayer, bits: List[int]):
        self._bits_configurations = bits_configurations
        self._num_weights = len(weight_observers)
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
            num_of_weights.append(weight_observers[trace_index].get_numels())
            norm_of_weights.append(weight_observers[trace_index].get_input_norm())
        self._num_weights_per_layer = torch.Tensor(num_of_weights)
        self._norm_weights_per_layer = torch.Tensor(norm_of_weights)

        bits_in_megabyte = 2 ** 23
        self._model_sizes = []
        for bits_config in self._bits_configurations:
            size = torch.sum(torch.Tensor(bits_config) * self._num_weights_per_layer).item() / bits_in_megabyte
            self._model_sizes.append(size)
        self._bits = bits

    @staticmethod
    def get_all_quantizers_per_full_scope(model):
        all_quantizations = OrderedDict()
        for class_type in QUANTIZATION_MODULES.registry_dict.values():
            quantization_type = class_type.__name__
            all_quantizations.update(
                get_all_modules_by_type(
                    model.get_compression_modules_by_type(CompressionModuleType.ACTIVATION_QUANTIZER),
                    quantization_type))
            all_quantizations.update(
                get_all_modules_by_type(
                    model.get_compression_modules_by_type(CompressionModuleType.FUNCTION_QUANTIZER),
                    quantization_type))
            all_quantizations.update(get_all_modules_by_type(model.get_nncf_wrapped_model(), quantization_type))
        all_quantizations = OrderedDict(sorted(all_quantizations.items(), key=lambda x: str(x[0])))
        return all_quantizations

    # pylint: disable=too-many-branches
    @staticmethod
    def get_bitwidth_graph(algo_ctrl, model, all_quantizers_per_full_scope,
                           groups_of_adjacent_quantizers: GroupsOfAdjacentQuantizers) -> NNCFGraph:
        grouped_mode = bool(groups_of_adjacent_quantizers)
        nncf_graph = model.get_graph()
        for node_key in nncf_graph.get_all_node_keys():
            node = nncf_graph.get_nx_node_by_key(node_key)
            color = ''
            if node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR]:
                operator_name = node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].operator_name
                scope = node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].input_agnostic.scope_in_model
                module = model.get_module_by_scope(scope)
                if isinstance(module, NNCFConv2d):
                    color = 'lightblue'
                    if module.groups == module.in_channels:
                        operator_name = 'DW_Conv2d'
                        color = 'purple'
                if not grouped_mode:
                    operator_name += '_#{}'.format(str(node[NNCFGraph.ID_NODE_ATTR]))
                node['label'] = operator_name
                node['style'] = 'filled'
                if color:
                    node['color'] = color

        non_weight_quantizers = algo_ctrl.non_weight_quantizers
        bits_color_map = {4: 'red', 8: 'green', 6: 'orange'}
        for quantizer_id, quantizer_info in non_weight_quantizers.items():
            affected_iap_ctx_list = quantizer_info.affected_ia_op_exec_contexts
            for activation_iap_ctx in affected_iap_ctx_list:
                post_hooked_nx_node_key = nncf_graph.get_node_id_by_iap_context(activation_iap_ctx)
                post_hooked_module_node = nncf_graph.get_nx_node_by_key(post_hooked_nx_node_key)
                operator_name = post_hooked_module_node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].operator_name
                if not grouped_mode:
                    operator_name += '_#{}'.format(str(post_hooked_module_node[NNCFGraph.ID_NODE_ATTR]))
                post_hooked_module_node['label'] = operator_name

                for next_nx_node_key in nncf_graph.get_successors(post_hooked_nx_node_key):
                    activation_fq_node = nncf_graph.get_nx_node_by_key(next_nx_node_key)
                    activation_quantizer = non_weight_quantizers[quantizer_id].quantizer_module_ref
                    bits = activation_quantizer.num_bits

                    activation_fq_node['color'] = bits_color_map[bits]
                    activation_fq_node['style'] = 'filled'
                    node_id = activation_fq_node[NNCFGraph.ID_NODE_ATTR]
                    if grouped_mode:
                        node_id = groups_of_adjacent_quantizers.get_group_id_for_quantizer(activation_quantizer)
                        if node_id is None:
                            nncf_logger.error('No group for activation quantizer: {}'.format(next_nx_node_key))
                            node_id = 'UNDEFINED'
                    activation_fq_node['label'] = '{}_bit__AFQ_#{}'.format(bits, str(node_id))

        for scope, quantizer in all_quantizers_per_full_scope.items():
            if quantizer.is_weights:
                node = nncf_graph.find_node_in_nx_graph_by_scope(scope)
                if node is None:
                    raise AttributeError('Failed to get node by scope={}'.format(str(scope)))
                if node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR]:
                    bits = quantizer.num_bits
                    node_id = node[NNCFGraph.ID_NODE_ATTR]
                    if grouped_mode:
                        node_id = groups_of_adjacent_quantizers.get_group_id_for_quantizer(quantizer)
                        if node_id is None:
                            nncf_logger.error('No group for weight quantizer: {}'.format(scope))
                            node_id = 'UNDEFINED'
                    node['label'] = '{}_bit__WFQ_#{}'.format(bits, str(node_id))
                    node['color'] = bits_color_map[bits]
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

    def dump_metric_MB(self, configuration_metric: List[Tensor]):
        import matplotlib.pyplot as plt
        list_to_plot = [cm.item() for cm in configuration_metric]
        fig = plt.figure()
        fig.suptitle('Pareto Frontier')
        ax = fig.add_subplot(2, 1, 1)
        ax.set_yscale('log')
        ax.set_xlabel('Model Size (MB)')
        ax.set_ylabel('Metric value (total perturbation)')
        ax.scatter(self._model_sizes, list_to_plot, s=20, facecolors='none', edgecolors='r')
        cm = torch.Tensor(configuration_metric)
        cm_m = cm.median().item()
        configuration_index = configuration_metric.index(cm_m)
        ms_m = self._model_sizes[configuration_index]
        ax.scatter(ms_m, cm_m, s=30, facecolors='none', edgecolors='b', label='median from all metrics')
        ax.legend()
        plt.savefig(os.path.join(self._dump_dir, 'Pareto_Frontier'))
        nncf_logger.info(
            'Distribution of HAWQ metrics: min_value={:.3f}, max_value={:.3f}, median_value={:.3f}, '
            'median_index={}, total_number={}'.format(cm.min().item(), cm.max().item(), cm_m,
                                                      configuration_index,
                                                      len(configuration_metric)))

    def dump_metric_flops(self, configuration_metric: List[Tensor], flops_per_config: List[float],
                          choosen_config_index: int):
        import matplotlib.pyplot as plt
        list_to_plot = [cm.item() for cm in configuration_metric]
        fig = plt.figure()
        fig.suptitle('Pareto Frontier')
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Compression ratio: total INT8 FLOPS_BITS / total MIXED INT FLOPS_BITS')
        ax.set_ylabel('Metric value (total perturbation)')
        ax.scatter(flops_per_config, list_to_plot, s=10, alpha=0.3)  # s=20, facecolors='none', edgecolors='r')
        flops_per_config = [torch.Tensor([v]) for v in flops_per_config]
        cm = torch.Tensor(flops_per_config)
        cm_m = cm.median().item()
        configuration_index = flops_per_config.index(cm_m)
        ms_m = configuration_metric[configuration_index].item()
        ax.scatter(cm_m, ms_m, s=30, facecolors='none', edgecolors='b', label='median from all metrics')
        cm_c = configuration_metric[choosen_config_index].item()
        fpc_c = flops_per_config[choosen_config_index].item()
        ax.scatter(fpc_c, cm_c, s=30, facecolors='none', edgecolors='r', label='chosen config')

        ax.legend()
        plt.savefig(os.path.join(self._dump_dir, 'Pareto_Frontier_compress_ratio'))

    def dump_density_of_quantization_noise(self):
        noise_per_config = []  # type: List[Tensor]
        for bits_config in self._bits_configurations:
            qnoise = 0
            for i in range(self._num_weights):
                layer_bits = bits_config[i]
                execution_index = self._traces_order.get_execution_index_by_traces_index(i)
                qnoise += self._perturbations.get(layer_id=execution_index, bitwidth=layer_bits)
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
        b = max(self._bits)
        perturb = [p[b] for p in self._perturbations.get_all().values()]
        ax.plot(
            [p / m / n for p, m, n in zip(perturb, self._num_weights_per_layer, self._norm_weights_per_layer)],
            label='normalized {}-bit noise'.format(b))
        ax.plot(perturb, label='{}-bit noise'.format(b))
        ax.plot(self._traces_per_layer.cpu().numpy(), label='trace')
        ax.plot([n * p for n, p in zip(self._traces_per_layer, perturb)], label='trace * noise')
        ax.legend()
        plt.savefig(os.path.join(self._dump_dir, 'Quantization_noise_vs_Average_Trace'))

    def dump_bitwidth_graph(self, algo_ctrl: 'QuantizationController', model: 'NNCFNetwork',
                            groups_of_adjacent_quantizers: GroupsOfAdjacentQuantizers):
        all_quantizers_per_full_scope = self.get_all_quantizers_per_full_scope(model)
        graph = self.get_bitwidth_graph(algo_ctrl, model, all_quantizers_per_full_scope, groups_of_adjacent_quantizers)
        graph.dump_graph(self._dump_dir / Path('bitwidth_graph.dot'))
