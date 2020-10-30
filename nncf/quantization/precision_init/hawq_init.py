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
import itertools
from collections import OrderedDict
from enum import Enum
from typing import List, Dict, Union, Tuple, NamedTuple, Callable, Any

import torch
import warnings
from bisect import bisect_left
from operator import itemgetter
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss

from nncf.debug import is_debug
from nncf.dynamic_graph.context import Scope
from nncf.nncf_logger import logger as nncf_logger
from nncf.quantization.layers import QuantizersSwitcher
from .adjacent_quantizers import GroupsOfAdjacentQuantizers, AdjacentQuantizers
from .compression_ratio import CompressionRatioCalculator
from .hawq_debug import HAWQDebugger
from .manual_init import ManualPrecisionInitializer
from .perturbations import Perturbations, PerturbationObserver
from .traces_order import TracesPerLayer, TracesOrder
from ..hessian_trace import HessianTraceEstimator
from ..hw_precision_constraints import HWPrecisionConstraints
from ..quantizer_id import QuantizerId
from ...layer_utils import ProxyModule
from ...module_operations import UpdateParameter
from ...structures import QuantizationPrecisionInitArgs


class BitwidthAssignmentMode(Enum):
    STRICT = 'strict'
    LIBERAL = 'liberal'

    @staticmethod
    def from_str(config_value: str) -> 'BitwidthAssignmentMode':
        if config_value == BitwidthAssignmentMode.STRICT.value:
            return BitwidthAssignmentMode.STRICT
        if config_value == BitwidthAssignmentMode.LIBERAL.value:
            return BitwidthAssignmentMode.LIBERAL
        raise RuntimeError("Unknown bitwidth assignment mode")


class HAWQPrecisionInitializer(ManualPrecisionInitializer):
    def __init__(self, algo: 'QuantizationController', config: 'NNCFConfig',
                 init_args: QuantizationPrecisionInitArgs):
        super().__init__(algo, config, init_args)
        self._criterion_fn = init_args.criterion_fn
        self._criterion = init_args.criterion
        self._data_loader = init_args.data_loader
        self._traces_per_layer_path = config.get('traces_per_layer_path', None)
        self._num_data_points = config.get('num_data_points', 1000)
        self._iter_number = config.get('iter_number', 500)
        self._tolerance = config.get('tolerance', 1e-5)
        self._compression_ratio = config.get('compression_ratio', 1.5)
        self._bits = self._hw_precision_constraints.get_all_unique_bits() \
            if self._hw_precision_constraints else config.get('bits', [4, 8])
        self._init_device = init_args.device
        self.flops_counter = CompressionRatioCalculator(self._model, self._quantizers_handler)
        self._groups_of_adjacent_quantizers = GroupsOfAdjacentQuantizers(algo)
        self._dump_hawq_data = config.get('dump_hawq_data', False)
        bitwidth_assignment_mode_str = config.get('bitwidth_assignment_mode', BitwidthAssignmentMode.LIBERAL.value)
        self._bitwidth_assignment_mode = BitwidthAssignmentMode.from_str(bitwidth_assignment_mode_str)

    def apply_init(self):
        if not self._quantizers_handler.get_weight_quantizers_in_execution_order_per_id():
            return None
        original_device = next(self._model.parameters()).device
        self._model.to(self._init_device)

        traces_per_layer = self._calc_traces(self._criterion_fn, self._criterion, self._iter_number, self._tolerance)
        if not traces_per_layer:
            raise RuntimeError('Failed to calculate hessian traces!')

        traces_order = traces_per_layer.traces_order
        num_weights = len(self._weight_quantizations_by_execution_order)
        bits_configurations = self.get_configs_constrained_by_traces_order(self._bits, num_weights)

        weight_quantizer_ids_in_execution_order = list(self._weight_quantizations_by_execution_order.keys())

        if self._bitwidth_assignment_mode == BitwidthAssignmentMode.STRICT:
            self._merge_constraints_for_adjacent_quantizers(self._groups_of_adjacent_quantizers,
                                                            self._hw_precision_constraints)

        bits_configurations = self._filter_configs_by_precision_constraints(bits_configurations,
                                                                            self._hw_precision_constraints,
                                                                            weight_quantizer_ids_in_execution_order,
                                                                            traces_order)
        if not bits_configurations:
            warnings.warn('All bits configurations are incompatible with HW Config!', RuntimeWarning)
            return None

        if self._bitwidth_assignment_mode == BitwidthAssignmentMode.STRICT:
            bits_configurations = \
                self._filter_configs_by_grouped_weight_quantizers(bits_configurations,
                                                                  weight_quantizer_ids_in_execution_order,
                                                                  self._groups_of_adjacent_quantizers,
                                                                  traces_order)
        if not bits_configurations:
            warnings.warn('No bits configurations are left after removing inconsistent groups of weight quantizers'
                          ' with adjacent activation quantizers!', RuntimeWarning)
            return None

        flops_bits_per_config = self.get_flops_bits_per_config(bits_configurations, traces_order)
        min_ratio = min(flops_bits_per_config)
        max_ratio = max(flops_bits_per_config)
        if not min_ratio <= self._compression_ratio <= max_ratio:
            raise AttributeError('Invalid compression ratio={}. Should be within range [{:.3f}, {:.3f}]'.format(
                self._compression_ratio, min_ratio, max_ratio))

        perturbations, weight_observers = self.calc_quantization_noise()

        configuration_metric = self.calc_hawq_metric_per_configuration(bits_configurations, perturbations,
                                                                       traces_per_layer, self._init_device)

        config_index = self.choose_configuration(configuration_metric, flops_bits_per_config)
        chosen_config_in_traces_order = bits_configurations[config_index]
        chosen_config_in_execution_order = traces_order.get_execution_order_config(chosen_config_in_traces_order)
        nncf_logger.info('Chosen HAWQ configuration with ratio={:.2f}, bitwidth per weightable layer={}'.format(
            flops_bits_per_config[config_index], chosen_config_in_execution_order))
        nncf_logger.debug('Order of the weightable layers in the HAWQ configuration (in descending order of average '
                          'Hessian traces) ={}'.format(traces_order))

        self.set_chosen_config(chosen_config_in_execution_order)
        self._model.rebuild_graph()
        if is_debug() or self._dump_hawq_data:
            hawq_debugger = HAWQDebugger(bits_configurations, perturbations,
                                         weight_observers, traces_per_layer, self._bits)
            hawq_debugger.dump_metric_MB(configuration_metric)
            hawq_debugger.dump_metric_flops(configuration_metric, flops_bits_per_config, config_index)
            hawq_debugger.dump_avg_traces()
            hawq_debugger.dump_density_of_quantization_noise()
            hawq_debugger.dump_perturbations_ratio()
            hawq_debugger.dump_bitwidth_graph(self._algo, self._model, self._groups_of_adjacent_quantizers)
        str_bw = [str(element) for element in self.get_bitwidth_per_scope()]
        nncf_logger.info('\n'.join(['\n\"bitwidth_per_scope\": [', ',\n'.join(str_bw), ']']))

        self._model.to(original_device)

        ordered_metric_per_layer = self.get_metric_per_layer(chosen_config_in_execution_order, perturbations,
                                                             traces_per_layer)
        return ordered_metric_per_layer

    @staticmethod
    def _merge_constraints_for_adjacent_quantizers(groups_of_adjacent_quantizers, hw_precision_constraints):
        if not hw_precision_constraints:
            return
        for group in groups_of_adjacent_quantizers:
            all_bits_sets = []
            quantizer_ids = []
            all_quantizers = group.weight_quantizers + group.activation_quantizers
            for quantizer_id, _ in all_quantizers:
                all_bits_sets.append(hw_precision_constraints.get(quantizer_id))
                quantizer_ids.append(quantizer_id)
            minimal_set_bits = set.intersection(*all_bits_sets)
            for quantizer_id in quantizer_ids:
                if not minimal_set_bits:
                    raise RuntimeError(
                        'No bits configurations are left after removing inconsistent groups of weight quantizers'
                        ' with adjacent activation quantizers!')
                hw_precision_constraints.replace(quantizer_id, minimal_set_bits)

    def get_flops_bits_per_config(self, bits_configurations: List[List[int]], traces_order: TracesOrder) -> List[float]:
        skipped = self._quantizers_handler.get_skipped_weight_quantizers_per_id()
        flops_bits_per_config = []
        for bits_config in bits_configurations:
            execution_order_config = traces_order.get_execution_order_config(bits_config)
            flops_bits_per_config.append(
                self.flops_counter.ratio_for_bits_configuration(execution_order_config, skipped))
        return flops_bits_per_config

    def get_bitwidth_per_scope(self) -> List[List[Union[int, str]]]:
        sorted_quantizers = OrderedDict(sorted(self._all_quantizers_per_scope.items(), key=lambda x: str(x[0])))
        full_bitwidth_per_scope = []
        for scope, quantizer in sorted_quantizers.items():
            full_bitwidth_per_scope.append([quantizer.num_bits, str(scope)])
        return full_bitwidth_per_scope

    class ParamsToRestore(NamedTuple):
        originally_disabled_gradients: List[str]
        skipped_gradients_to_enable: List[Tuple[nn.Module, str]]

    @staticmethod
    def disable_all_gradients_except_weights_of_quantized_modules(
            quantizers_switcher: QuantizersSwitcher,
            quantized_weight_modules_registry: Dict[str, torch.nn.Module],
            model: nn.Module,
            scopes_of_skipped_weight_quantizers: List[
                'Scope'] = None) -> ParamsToRestore:  # pylint: disable=undefined-variable
        """
        Disables gradients of all parameters, except for layers that have quantizers for weights, which wasn't skipped
        because of single precision constraints.
        :param quantizers_switcher: object that is responsible for enabling and disabling quantizers
        :param quantized_weight_modules_registry: modules with quantized weights per scope
        :param model: model to access all parameters
        :param scopes_of_skipped_weight_quantizers: list of string scopes of layers that have a single precision
        constraint and which weights should be skipped from bitwidth initialization
        :return: list of names of the parameters that were originally disabled
        """
        originally_disabled_gradients = []
        skipped_gradients_to_enable = []

        # Some quantizers can be disabled in a staged scenario on creation of staged scheduler
        # Need to save originally disabled quantizers for restoring their state after initialization
        quantizers_switcher.disable_quantizers()

        # remember gradients of quantized modules that were enabled
        gradients_to_enable = []
        for scope, quantized_module in quantized_weight_modules_registry.items():
            is_skipped = False
            for skipped_weight_quantizer_scope in scopes_of_skipped_weight_quantizers:
                if skipped_weight_quantizer_scope in Scope.from_str(scope):
                    is_skipped = True
                    break
            for param_name, param in quantized_module.named_parameters():
                if param.requires_grad:
                    # disable gradients for skipped module for optimization of Hessian Trace search
                    if is_skipped:
                        skipped_gradients_to_enable.append((quantized_module, param_name))
                        param.requires_grad = False
                    else:
                        gradients_to_enable.append((quantized_module, param_name))

        # disable all gradients, except already disabled
        for param_name, param in model.named_parameters():
            if not param.requires_grad:
                originally_disabled_gradients.append(param_name)
            else:
                param.requires_grad = False

        # enable gradients of quantized modules that were disabled
        for quantized_module in quantized_weight_modules_registry.values():
            for param_name, param in quantized_module.named_parameters():
                if (quantized_module, param_name) in gradients_to_enable and not 'bias' in param_name:
                    param.requires_grad = True
        return HAWQPrecisionInitializer.ParamsToRestore(originally_disabled_gradients, skipped_gradients_to_enable)

    def _calc_traces(self, criterion_fn: Callable[[Any, Any, _Loss], torch.Tensor], criterion: _Loss,
                     iter_number: int, tolerance: float) -> TracesPerLayer:
        if self._traces_per_layer_path:
            return TracesPerLayer(torch.load(self._traces_per_layer_path).to(self._init_device))

        quantizers_switcher = QuantizersSwitcher(list(self._all_quantizers_per_scope.values()))
        params_to_restore = self.disable_all_gradients_except_weights_of_quantized_modules(
            quantizers_switcher,
            self._algo.quantized_weight_modules_registry,
            self._model,
            self._quantizers_handler.get_scope_of_skipped_weight_quantizers())

        trace_estimator = HessianTraceEstimator(self._model, criterion_fn, criterion, self._init_device,
                                                self._data_loader, self._num_data_points)
        avg_traces = trace_estimator.get_average_traces(max_iter=iter_number, tolerance=tolerance)

        self.restore_disabled_gradients(quantizers_switcher, self._model, self._algo.quantized_weight_modules_registry,
                                        params_to_restore)

        return TracesPerLayer(avg_traces)

    @staticmethod
    def restore_disabled_gradients(quantizers_switcher: QuantizersSwitcher,
                                   model: nn.Module,
                                   quantized_weight_modules_registry: Dict[str, torch.nn.Module],
                                   params_to_restore: ParamsToRestore):
        """
        Restore requires_grad property of all parameters back, except for ones that were originally disabled
        :param quantizers_switcher: object that is responsible for enabling and disabling quantizers
        :param model: model to access all parameters
        :param quantized_weight_modules_registry: modules with quantized weights per scope
        :param params_to_restore: storage names of the parameters that should restore reguires_grad property
        """
        for quantized_module in quantized_weight_modules_registry.values():
            for param_name, param in quantized_module.named_parameters():
                if (quantized_module, param_name) in params_to_restore.skipped_gradients_to_enable:
                    param.requires_grad = True

        for param_name, param in model.named_parameters():
            if param_name not in params_to_restore.originally_disabled_gradients:
                param.requires_grad = True
        quantizers_switcher.enable_quantizers()

    @staticmethod
    def get_configs_constrained_by_traces_order(bits_: List[int], num_layers: int) -> List[List[int]]:
        bit_configs = []
        if num_layers == 0:
            return bit_configs
        bits = sorted(bits_)
        m = len(bits)
        L = num_layers
        for j in range(1, m + 1):
            for combo_bits in itertools.combinations(bits, j):
                for combo_partitions in itertools.combinations(list(range(1, L)), j - 1):
                    bit_config = []
                    prev_p = 0
                    for (p, b) in zip(combo_partitions + (L,), combo_bits):
                        bit_config += [b] * (p - prev_p)
                        prev_p = p
                    bit_configs.append(bit_config)
        return bit_configs

    @staticmethod
    def _filter_configs_by_precision_constraints(bits_configurations: List[List[int]],
                                                 hw_precision_constraints: HWPrecisionConstraints,
                                                 ordered_weight_ids: List[QuantizerId],
                                                 traces_order: TracesOrder) -> List[List[int]]:
        if not hw_precision_constraints:
            return bits_configurations

        filtered_bits_configurations = []
        for bits_configuration in bits_configurations:
            is_all_bitwidth_compatible = True
            ordered_config = traces_order.get_execution_order_config(bits_configuration)
            for i, bitwidth in enumerate(ordered_config):
                weight_id = ordered_weight_ids[i]
                bits_constraints = hw_precision_constraints.get(weight_id)
                if bitwidth not in bits_constraints:
                    is_all_bitwidth_compatible = False
                    break
            if is_all_bitwidth_compatible:
                filtered_bits_configurations.append(bits_configuration)
        return filtered_bits_configurations

    def calc_quantization_noise(self) -> [Perturbations, List[PerturbationObserver]]:
        hook_handles = []
        observers = []
        for module in self._weight_quantizations_by_execution_order.values():
            observer = PerturbationObserver(self._init_device)
            hook_handles.append(module.register_forward_hook(observer.calc_perturbation))
            observers.append(observer)

        perturbations = Perturbations()
        for b in self._bits:
            for wi in self._weight_quantizations_by_execution_order.values():
                wi.num_bits = b

            # TODO: replace with do_dummy_forward call on compressing in eval mode only
            # Call each UpdateWeight op, instead of calling dummy_forward. It's needed because dummy_forward must be
            # run with force_eval=False, which overrides BatchNorm statistics. This requirement comes from the models
            # with quantizers on the branches, which are enabled in train mode (AuxLogits for Inception3)
            for quantized_module in self._algo.quantized_weight_modules_registry.values():
                ops = [op for op in quantized_module.pre_ops.values() if isinstance(op, UpdateParameter)]
                ops += [op for op in quantized_module.post_ops.values() if isinstance(op, UpdateParameter)]
                for op in ops:
                    op(ProxyModule(quantized_module), None)

            for i, observer in enumerate(observers):
                perturbations.add(layer_id=i, bitwidth=b, perturbation=observer.get_observation().to(self._init_device))

        for handle in hook_handles:
            handle.remove()
        return perturbations, observers

    @staticmethod
    def calc_hawq_metric_per_configuration(bits_configurations: List[List[int]], perturbations: Perturbations,
                                           traces_per_layer: TracesPerLayer, device) -> List[Tensor]:
        configuration_metric = []
        for bits_config in bits_configurations:
            hawq_metric = torch.Tensor([0]).to(device)
            for trace_index, layer_bits in enumerate(bits_config):
                execution_index = traces_per_layer.traces_order.get_execution_index_by_traces_index(trace_index)
                hawq_metric += traces_per_layer.get_by_trace_index(trace_index) * perturbations.get(
                    layer_id=execution_index, bitwidth=layer_bits)
            configuration_metric.append(hawq_metric)
        return configuration_metric

    def choose_configuration(self, configuration_metric: List[Tensor], flops_bits_per_config: List[float]) -> int:
        num_configs = len(configuration_metric)

        sorted_flops_order = [x[0] for x in sorted(enumerate(flops_bits_per_config), reverse=False, key=lambda x: x[1])]
        sorted_flops_bits_per_config = sorted(flops_bits_per_config)

        boundary_index = bisect_left(sorted_flops_bits_per_config, self._compression_ratio)
        indexes_to_check = [sorted_flops_order[i] for i in range(boundary_index, num_configs)]
        best_metric = min(list(itemgetter(*indexes_to_check)(configuration_metric)))
        best_config_index = configuration_metric.index(best_metric)
        return best_config_index

    def set_chosen_config(self, weight_bitwidth_in_execution_order: List[int]):
        for wq, bits in zip(self._weight_quantizations_by_execution_order.values(), weight_bitwidth_in_execution_order):
            wq.num_bits = bits
        if self._groups_of_adjacent_quantizers:
            for group in self._groups_of_adjacent_quantizers:
                weight_bitwidth_set = {wq.num_bits for _, wq in group.weight_quantizers}
                if self._bitwidth_assignment_mode == BitwidthAssignmentMode.STRICT:
                    self._set_activations_bitwidth_strictly(group, weight_bitwidth_set)
                else:
                    self._set_activation_bitwidth_liberally(group, weight_bitwidth_set)
        else:
            # TODO: delete not-consistent pairs of activation and weights for pattern-based approach
            pairs = self._algo.get_weights_activation_quantizers_pairs()
            for pair in pairs:
                wqs, aq = pair
                aq.num_bits = max([wq.num_bits for wq in wqs])

    def _set_activation_bitwidth_liberally(self, group: AdjacentQuantizers, weight_bitwidth_set):
        for quantizer_id, aq in group.activation_quantizers:
            activation_bitwidth_set = self._hw_precision_constraints.get(quantizer_id)
            intersection = activation_bitwidth_set.intersection(weight_bitwidth_set)
            if activation_bitwidth_set.__len__() == 1:
                aq.num_bits = activation_bitwidth_set.pop()
            elif intersection:
                aq.num_bits = min(intersection)
            elif activation_bitwidth_set:
                aq.num_bits = min(activation_bitwidth_set)
            elif weight_bitwidth_set:
                aq.num_bits = min(weight_bitwidth_set)

    def _set_activations_bitwidth_strictly(self, group: AdjacentQuantizers, weight_bitwidth_set):
        if len(weight_bitwidth_set) > 1:
            raise RuntimeError('Invalid grouping of weight quantizers')
        all_constraints = set()
        for quantizer_id, aq in group.activation_quantizers:
            all_constraints.update(self._hw_precision_constraints.get(quantizer_id))
        common_constraints = set(all_constraints)
        for quantizer_id, aq in group.activation_quantizers:
            constraint = self._hw_precision_constraints.get(quantizer_id)
            common_constraints = common_constraints.intersection(constraint)
        if weight_bitwidth_set:
            common_constraints = common_constraints.intersection(weight_bitwidth_set)
        if not common_constraints:
            raise RuntimeError('No hardware compatible bitwidth for activation quantizers')
        for quantizer_id, aq in group.activation_quantizers:
            aq.num_bits = sorted(list(common_constraints))[0]

    def get_metric_per_layer(self, chosen_config_in_execution_order: List[int], perturbations: Perturbations,
                             traces_per_layer: TracesPerLayer):
        metric_per_layer = []
        for i, layer_bits in enumerate(chosen_config_in_execution_order):
            metric_per_layer.append(traces_per_layer.get_by_execution_index(i) * perturbations.get(i, layer_bits))
        ordered_metric_per_layer = [i[0] for i in
                                    sorted(enumerate(metric_per_layer), reverse=True, key=lambda x: x[1])]
        return ordered_metric_per_layer

    @staticmethod
    def _filter_configs_by_grouped_weight_quantizers(bits_configurations: List[List[int]],
                                                     weight_quantization_ids_by_execution_order: List[QuantizerId],
                                                     groups_of_adjacent_quantizers: GroupsOfAdjacentQuantizers,
                                                     traces_order: TracesOrder) -> List[List[int]]:
        """ removes configs where adjacent weight quantizers have different bitwidth. Adjacency is defined by common
        activation quantizers"""
        filtered_bits_configurations = []
        all_grouped_indexes = []
        for group_of_adjacent_quantizers in groups_of_adjacent_quantizers:
            wqs = group_of_adjacent_quantizers.weight_quantizers
            if len(wqs) > 1:
                indexes_of_grouped_wq = []
                for quantizer_id, _ in wqs:
                    index_by_execution_order = weight_quantization_ids_by_execution_order.index(quantizer_id)
                    indexes_of_grouped_wq.append(index_by_execution_order)
                all_grouped_indexes.append(indexes_of_grouped_wq)

        if not all_grouped_indexes:
            return bits_configurations

        for bits_configuration in bits_configurations:
            bitwidth_by_execution_order = traces_order.get_execution_order_config(bits_configuration)
            keep_config = True
            for indexes_of_grouped_wq in all_grouped_indexes:
                grouped_bits = [bitwidth_by_execution_order[index] for index in indexes_of_grouped_wq]
                if grouped_bits[1:] != grouped_bits[:-1]:
                    keep_config = False
                    break
            if keep_config:
                filtered_bits_configurations.append(bits_configuration)

        return filtered_bits_configurations
