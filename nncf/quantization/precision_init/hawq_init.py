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
from copy import deepcopy
from enum import Enum
from typing import List, Dict, Union, Tuple, NamedTuple, Callable, Any, Set

import torch
import warnings
from bisect import bisect_left
from operator import itemgetter

from torch import Tensor, nn
from torch.nn.modules.loss import _Loss

from nncf.debug import is_debug
from nncf.dynamic_graph.context import Scope
from nncf.nncf_logger import logger as nncf_logger
from nncf.quantization.layers import QuantizersSwitcher, QuantizerConfig
from nncf.quantization.precision_init.adjacent_quantizers import GroupsOfAdjacentQuantizers
from nncf.quantization.precision_init.compression_ratio import CompressionRatioCalculator
from nncf.quantization.precision_init.hawq_debug import HAWQDebugger
from nncf.quantization.precision_init.base_init import BasePrecisionInitParams, BasePrecisionInitializer
from nncf.quantization.precision_init.perturbations import Perturbations, PerturbationObserver
from nncf.quantization.precision_init.traces_order import TracesPerLayer, TracesOrder
from nncf.quantization.hessian_trace import HessianTraceEstimator
from nncf.quantization.precision_constraints import HardwareQuantizationConstraints
from nncf.quantization.quantizer_id import QuantizerId, WeightQuantizerId
from nncf.quantization.structs import WeightQuantizerInfo
from nncf.quantization.quantizer_setup import QuantizationPointId, SingleConfigQuantizerSetup
from nncf.structures import QuantizationPrecisionInitArgs


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


class HAWQPrecisionInitParams(BasePrecisionInitParams):
    def __init__(self,
                 user_init_args: QuantizationPrecisionInitArgs,
                 bits: List[int] = None,
                 bitwidth_per_scope: List[List] = None,
                 traces_per_layer_path: str = None,
                 num_data_points: int = None,
                 iter_number: int = None,
                 tolerance: float = None,
                 compression_ratio: float = None,
                 dump_hawq_data: bool = None,
                 bitwidth_assignment_mode: BitwidthAssignmentMode = None):
        super().__init__(user_init_args)
        self.bits = bits
        self.bitwidth_per_scope = bitwidth_per_scope
        self.traces_per_layer_path = traces_per_layer_path
        self.num_data_points = num_data_points
        self.iter_number = iter_number
        self.tolerance = tolerance
        self.compression_ratio = compression_ratio
        self.dump_hawq_data = dump_hawq_data
        self.bitwidth_assignment_mode = bitwidth_assignment_mode

    @classmethod
    def from_config(cls, hawq_init_config_dict: Dict,
                    user_init_args: QuantizationPrecisionInitArgs) -> 'HAWQPrecisionInitParams':
        return cls(
            user_init_args=user_init_args,
            bits=hawq_init_config_dict.get('bits', [2, 4, 8]),
            traces_per_layer_path=hawq_init_config_dict.get('traces_per_layer_path', None),
            num_data_points=hawq_init_config_dict.get('num_data_points', 100),
            iter_number=hawq_init_config_dict.get('iter_number', 200),
            tolerance=hawq_init_config_dict.get('tolerance', 1e-4),
            compression_ratio=hawq_init_config_dict.get('compression_ratio', 1.5),
            dump_hawq_data=hawq_init_config_dict.get('dump_init_precision_data', False),
            bitwidth_assignment_mode=BitwidthAssignmentMode.from_str(
                hawq_init_config_dict.get('bitwidth_assignment_mode', BitwidthAssignmentMode.LIBERAL.value)
            )
        )


ConfigurationForHAWQToEvaluate = List[QuantizerConfig]
CoveringConfigurationForQuantNoiseCalculation = List[QuantizerConfig]


class TraceOrderBitwidthMatcher:
    def __init__(self, available_bits: List[int], traces_order: TracesOrder):
        self._available_bits = available_bits
        self._traces_order = traces_order
        self._bit_sequences = self.get_all_non_decreasing_bit_sequences()

    def get_all_non_decreasing_bit_sequences(self) -> List[List[int]]:
        sequences = []
        bits_ = deepcopy(self._available_bits)
        seq_len = len(self._traces_order)
        if seq_len == 0:
            return sequences
        bits = sorted(bits_)
        m = len(bits)
        L = seq_len
        for j in range(1, m + 1):
            for combo_bits in itertools.combinations(bits, j):
                for combo_partitions in itertools.combinations(list(range(1, L)), j - 1):
                    bit_config = []
                    prev_p = 0
                    for (p, b) in zip(combo_partitions + (L,), combo_bits):
                        bit_config += [b] * (p - prev_p)
                        prev_p = p
                    sequences.append(bit_config)
        return sequences

    @staticmethod
    def _select_first_closest_bitwidth_qconf(qconf_list: List[QuantizerConfig],
                                             target_bitwidth: int) -> QuantizerConfig:
        bw_diffs = [abs(qc.bits - target_bitwidth) for qc in qconf_list]
        _, min_idx = min((val, idx) for (idx, val) in enumerate(bw_diffs))
        return qconf_list[min_idx]

    def _deduplicate(self, qconf_sequences_to_search: List[ConfigurationForHAWQToEvaluate]) -> \
            List[ConfigurationForHAWQToEvaluate]:
        tupled_sequence = [tuple(seq) for seq in qconf_sequences_to_search]
        odict = OrderedDict.fromkeys(tupled_sequence)
        deduped_tupled_sequence = list(odict.keys())
        return [list(tup) for tup in deduped_tupled_sequence]

    def _generate_covering_configurations(self, observed_qconfs: List[Dict[QuantizerConfig, QuantizerConfig]]):
        covering_confs = []  # type: List[CoveringConfigurationForQuantNoiseCalculation]
        # For each index, put the largest qconf subset that only varies in bitwidth on top
        # so that the associated covering configurations would not require model regeneration
        optimized_observed_qconfs = []  # type: List[List[QuantizerConfig]]
        for qconf_oset in observed_qconfs:
            variants = []  # type: List[List[QuantizerConfig]]
            for qconf in qconf_oset.keys():
                variants.append(list(filter(qconf.is_a_bitwidth_variant, qconf_oset.keys())))
            max_bw_varying_variant = max(variants, key=len)
            other_qconfs = list(filter(lambda x: x not in max_bw_varying_variant, qconf_oset.keys()))
            optimized_observed_qconfs.append(max_bw_varying_variant + other_qconfs)

        max_depth = max([len(qconfs_for_trace_idx) for qconfs_for_trace_idx in optimized_observed_qconfs])
        for i in range(max_depth):
            covering_conf = []  # type: CoveringConfigurationForQuantNoiseCalculation
            for qconfs_for_trace_idx in optimized_observed_qconfs:
                if i < len(qconfs_for_trace_idx):
                    covering_conf.append(qconfs_for_trace_idx[i])
                else:
                    covering_conf.append(qconfs_for_trace_idx[-1])
            covering_confs.append(covering_conf)
        return covering_confs

    def get_qconfig_sequences_constrained_by_trace_order(self,
                                                         configuration_space_in_trace_order: List[
                                                             List[QuantizerConfig]],
                                                         indices_for_bitwidth_adjustment_only: Set[int]) -> \
            Tuple[List[ConfigurationForHAWQToEvaluate], List[CoveringConfigurationForQuantNoiseCalculation]]:
        """The 'constraint' is so that the each qconfig sequence should have non-decreasing bitwidths. It
        might be impossible to apply this constraint for a given configuration space (consider [[2], [6, 8], [4]]).
        In such a case, for trace order index positions where it was impossible to select a bitwidth so that the entire
        sequence is non-decreasing, the bitwidth closest to this target will be chosen instead."""
        if len(configuration_space_in_trace_order) != len(self._traces_order):
            raise ValueError("The size of the configuration space and the traces do not match!")
        retval = []  # type: List[ConfigurationForHAWQToEvaluate]
        observed_qconfs_in_retval = [OrderedDict()
                                     for _ in range(len(self._traces_order))]
        for bit_seq in self._bit_sequences:
            current_config_sequence_in_trace_order = []  # type: ConfigurationForHAWQToEvaluate
            for trace_idx, bitwidth in enumerate(bit_seq):

                if trace_idx in indices_for_bitwidth_adjustment_only:
                    bit_adjusted_default_qconfig = deepcopy(configuration_space_in_trace_order[trace_idx][0])
                    bit_adjusted_default_qconfig.bits = bitwidth
                    qconfig = bit_adjusted_default_qconfig
                else:
                    # TODO: do a selection based on strategy ("exhaustive" = add all available configurations,
                    # "preset" = do a selection based on a certain preset, "first" = select first match (as below),
                    # "custom" = use a custom selection function to be passed as arg to the HAWQ initializer
                    # OR: do non-bitwidth disambiguation higher up the stack, make sure that the configuration
                    # space at this spot only has 1 qconfig option for each bitwidth.
                    possible_qconfigs_for_current_trace_idx = configuration_space_in_trace_order[trace_idx]
                    first_closest_qconf = self._select_first_closest_bitwidth_qconf(
                        possible_qconfigs_for_current_trace_idx, bitwidth)
                    qconfig = deepcopy(first_closest_qconf)

                current_config_sequence_in_trace_order.append(qconfig)
                observed_qconfs_in_retval[trace_idx][qconfig] = qconfig
            retval.append(current_config_sequence_in_trace_order)
        return self._deduplicate(retval), self._generate_covering_configurations(observed_qconfs_in_retval)


class HAWQPrecisionInitializer(BasePrecisionInitializer):
    def __init__(self, algo: 'ExperimentalQuantizationController',
                 params: HAWQPrecisionInitParams,
                 hw_precision_constraints: HardwareQuantizationConstraints):
        self._groups_of_adjacent_quantizers = algo.groups_of_adjacent_quantizers
        self._bitwidth_assignment_mode = params.bitwidth_assignment_mode
        if self._bitwidth_assignment_mode == BitwidthAssignmentMode.STRICT:
            hw_precision_constraints = self._merge_constraints_for_adjacent_quantizers(
                self._groups_of_adjacent_quantizers,
                hw_precision_constraints)
        super().__init__(algo, params, hw_precision_constraints)
        init_args = params.user_init_args
        self._criterion_fn = init_args.criterion_fn
        self._criterion = init_args.criterion
        self._data_loader = init_args.data_loader
        self._traces_per_layer_path = params.traces_per_layer_path
        self._num_data_points = params.num_data_points
        self._iter_number = params.iter_number
        self._tolerance = params.tolerance
        self._compression_ratio = params.compression_ratio
        self._bits = self._hw_precision_constraints.get_all_unique_bits() \
            if self._hw_precision_constraints else params.bits
        self._init_device = init_args.device
        if self._init_device is None:
            self._init_device = next(self._model.parameters()).device
        self.flops_counter = CompressionRatioCalculator(self._model, self._quantizers_handler)
        self._dump_hawq_data = params.dump_hawq_data
        self._original_qp_id_vs_quantizer_module_id_dict = deepcopy(algo.setup_to_module_id_translation_dict)

    def apply_init(self) -> SingleConfigQuantizerSetup:
        if not self._quantizers_handler.get_weight_quantizers_in_execution_order_per_id():
            return self._algo.get_quantizer_setup_for_current_state()

        original_device = next(self._model.parameters()).device
        self._model.to(self._init_device)

        traces_per_layer = self._calc_traces(self._criterion_fn, self._criterion, self._iter_number, self._tolerance)
        if not traces_per_layer:
            raise RuntimeError('Failed to calculate hessian traces!')

        traces_order = traces_per_layer.traces_order
        weight_qconfigs_in_trace_order, covering_configurations = self.get_configs_constrained_by_traces_order(
            traces_order)

        weight_quantizer_ids_in_execution_order = list(self._weight_quantizations_by_execution_order.keys())


        if not weight_qconfigs_in_trace_order:
            warnings.warn('All bits configurations are incompatible with HW Config!', RuntimeWarning)
            return None

        if self._bitwidth_assignment_mode == BitwidthAssignmentMode.STRICT:
            weight_qconfigs_in_trace_order = \
                self._filter_configs_by_grouped_weight_quantizers(weight_qconfigs_in_trace_order,
                                                                  weight_quantizer_ids_in_execution_order,
                                                                  self._groups_of_adjacent_quantizers,
                                                                  traces_order)
        if not weight_qconfigs_in_trace_order:
            warnings.warn('No bits configurations are left after removing inconsistent groups of weight quantizers'
                          ' with adjacent activation quantizers!', RuntimeWarning)
            return self._algo.get_quantizer_setup_for_current_state()

        flops_bits_per_config = self.get_flops_bits_per_config(weight_qconfigs_in_trace_order, traces_order)
        min_ratio = min(flops_bits_per_config)
        max_ratio = max(flops_bits_per_config)
        if not min_ratio <= self._compression_ratio <= max_ratio:
            raise AttributeError('Invalid compression ratio={}. Should be within range [{:.3f}, {:.3f}]'.format(
                self._compression_ratio, min_ratio, max_ratio))

        perturbations, weight_observers = self.calc_quantization_noise(covering_configurations, traces_order)

        configuration_metric = self.calc_hawq_metric_per_configuration(weight_qconfigs_in_trace_order, perturbations,
                                                                       traces_per_layer, self._init_device)

        config_index = self.choose_configuration(configuration_metric, flops_bits_per_config)
        chosen_config_in_traces_order = weight_qconfigs_in_trace_order[config_index]
        chosen_config_in_execution_order = traces_order.get_execution_order_config(chosen_config_in_traces_order)
        biwidth_per_weightable_layer = [qconfig.bits for qconfig in chosen_config_in_execution_order]
        nncf_logger.info('Chosen HAWQ configuration with ratio={:.2f}, config per weightable layer={}'.format(
            flops_bits_per_config[config_index], biwidth_per_weightable_layer))
        nncf_logger.debug('Order of the weightable layers in the HAWQ configuration (in descending order of average '
                          'Hessian traces) ={}'.format(traces_order))

        final_quantizer_setup = self.set_chosen_config(chosen_config_in_traces_order, traces_order)
        if is_debug() or self._dump_hawq_data:
            hawq_debugger = HAWQDebugger(weight_qconfigs_in_trace_order,
                                         perturbations,
                                         covering_configurations,
                                         weight_observers, traces_per_layer, self._bits)
            hawq_debugger.dump_metric_MB(configuration_metric)
            hawq_debugger.dump_metric_flops(configuration_metric, flops_bits_per_config, config_index)
            hawq_debugger.dump_avg_traces()
            hawq_debugger.dump_density_of_quantization_noise()
            hawq_debugger.dump_perturbations_ratio()
            hawq_debugger.dump_bitwidth_graph(self._algo, self._model, self._groups_of_adjacent_quantizers)
        str_bw = [str(element) for element in self.get_bitwidth_per_scope(final_quantizer_setup)]
        nncf_logger.info('\n'.join(['\n\"bitwidth_per_scope\": [', ',\n'.join(str_bw), ']']))

        self._model.to(original_device)

        _ = self.get_metric_per_layer(chosen_config_in_execution_order, perturbations,
                                                             traces_per_layer)
        return final_quantizer_setup

    @staticmethod
    def _merge_constraints_for_adjacent_quantizers(groups_of_adjacent_quantizers: GroupsOfAdjacentQuantizers,
                                                   hw_precision_constraints: HardwareQuantizationConstraints) -> \
            HardwareQuantizationConstraints:
        if not hw_precision_constraints:
            return None
        retval = deepcopy(hw_precision_constraints)
        for group in groups_of_adjacent_quantizers:
            all_bits_sets = []
            quantizer_ids = []
            all_quantizers = group.weight_quantizers + group.activation_quantizers
            for quantizer_id, _ in all_quantizers:
                bitwidths_vs_qconfigs = retval.get_bitwidth_vs_qconfigs_dict(quantizer_id)
                bits = set(bitwidths_vs_qconfigs.keys())
                all_bits_sets.append(bits)
                quantizer_ids.append(quantizer_id)
            minimal_set_bits = set.intersection(*all_bits_sets)
            if not minimal_set_bits:
                raise RuntimeError(
                    'No bits configurations are left after removing inconsistent groups of weight quantizers'
                    ' with adjacent activation quantizers!')
            for quantizer_id in quantizer_ids:
                qconfigs = retval.get(quantizer_id)
                filtered_qconfigs = []
                for qconf in qconfigs:
                    if qconf.bits in minimal_set_bits:
                        filtered_qconfigs.append(qconf)
                retval.replace(quantizer_id, filtered_qconfigs)
        return retval

    def get_flops_bits_per_config(self, configurations_in_trace_order: List[List[QuantizerConfig]],
                                  traces_order: TracesOrder) -> List[float]:
        skipped = self._quantizers_handler.get_skipped_weight_quantizers_per_id()
        flops_bits_per_config = []
        for configuration in configurations_in_trace_order:
            execution_order_config = traces_order.get_execution_order_config(configuration)
            bit_sequence = [qc.bits for qc in execution_order_config]
            flops_bits_per_config.append(
                self.flops_counter.ratio_for_bits_configuration(bit_sequence, skipped))
        return flops_bits_per_config

    def get_bitwidth_per_scope(self, quantizer_setup: SingleConfigQuantizerSetup) -> List[List[Union[int, str]]]:
        scope_vs_bitwidth = {}
        for qp in quantizer_setup.quantization_points.values():
            scope_vs_bitwidth[str(qp.insertion_point)] = qp.qconfig.bits
        sorted_scope_vs_bitwidth = OrderedDict(sorted(scope_vs_bitwidth.items(), key=lambda x: x[0]))
        full_bitwidth_per_scope = []
        for scope, bitwidth in sorted_scope_vs_bitwidth.items():
            full_bitwidth_per_scope.append([bitwidth, scope])
        return full_bitwidth_per_scope

    class ParamsToRestore(NamedTuple):
        originally_disabled_gradients: List[str]
        skipped_gradients_to_enable: List[Tuple[nn.Module, str]]

    @staticmethod
    def disable_all_gradients_except_weights_of_quantized_modules(
            quantizers_switcher: QuantizersSwitcher,
            weight_quantizers: Dict[WeightQuantizerId, WeightQuantizerInfo],
            model: nn.Module,
            scopes_of_skipped_weight_quantizers: List[Scope] = None) -> ParamsToRestore: # pylint: disable=undefined-variable
        """
        Disables gradients of all parameters, except for layers that have quantizers for weights, which wasn't skipped
        because of single precision constraints.
        :param quantizers_switcher: object that is responsible for enabling and disabling quantizers
        :param weight_quantizers: modules with quantized weights per scope
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
        for wq_id, wq_info in weight_quantizers.items():
            quantized_module = wq_info.quantized_module
            scope = wq_id.get_scope()
            is_skipped = False
            for skipped_weight_quantizer_scope in scopes_of_skipped_weight_quantizers:
                if skipped_weight_quantizer_scope in scope:
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
        for wq_id in weight_quantizers.values():
            quantized_module = wq_id.quantized_module
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
            self._algo.weight_quantizers,
            self._model,
            self._quantizers_handler.get_scope_of_skipped_weight_quantizers())

        trace_estimator = HessianTraceEstimator(self._model, criterion_fn, criterion, self._init_device,
                                                self._data_loader, self._num_data_points)
        try:
            avg_traces = trace_estimator.get_average_traces(max_iter=iter_number, tolerance=tolerance)
        except RuntimeError as error:
            if "cuda out of memory" in error.args[0].lower():
                raise RuntimeError('Failed to estimate average Hessian traces within precision initialization. Specify '
                                   'a smaller batch size via --batch-size-init option in the NNCF samples or register '
                                   'a data loader with a smaller batch size. Refer to '
                                   '`NNCFConfig.register_extra_structs` and the `QuantizationPrecisionInitArgs`'
                                   ' class') from error
            raise error

        self.restore_disabled_gradients(quantizers_switcher, self._model, self._algo.weight_quantizers,
                                        params_to_restore)

        return TracesPerLayer(avg_traces)

    @staticmethod
    def restore_disabled_gradients(quantizers_switcher: QuantizersSwitcher,
                                   model: nn.Module,
                                   weight_quantizers: Dict[WeightQuantizerId, WeightQuantizerInfo],
                                   params_to_restore: ParamsToRestore):
        """
        Restore requires_grad property of all parameters back, except for ones that were originally disabled
        :param quantizers_switcher: object that is responsible for enabling and disabling quantizers
        :param model: model to access all parameters
        :param weight_quantizers: modules with quantized weights per scope
        :param params_to_restore: storage names of the parameters that should restore reguires_grad property
        """
        for wq_info in weight_quantizers.values():
            quantized_module = wq_info.quantized_module
            for param_name, param in quantized_module.named_parameters():
                if (quantized_module, param_name) in params_to_restore.skipped_gradients_to_enable:
                    param.requires_grad = True

        for param_name, param in model.named_parameters():
            if param_name not in params_to_restore.originally_disabled_gradients:
                param.requires_grad = True
        quantizers_switcher.enable_quantizers()

    def get_configs_constrained_by_traces_order(self, traces_order: TracesOrder) -> \
            Tuple[List[ConfigurationForHAWQToEvaluate], List[CoveringConfigurationForQuantNoiseCalculation]]:
        configuration_space_in_trace_order = []  # type: List[List[QuantizerConfig]]
        trace_order_indices_of_defaulted_configs = set()  # type: Set[int]
        quantizer_ids_in_exec_order = list(self._weight_quantizations_by_execution_order.keys())
        assert len(quantizer_ids_in_exec_order) == len(traces_order)
        for trace_idx in range(len(traces_order)):
            exec_idx = traces_order.get_execution_index_by_traces_index(trace_idx)
            qid = quantizer_ids_in_exec_order[exec_idx]
            default_qconfig = self._weight_quantizations_by_execution_order[qid].get_current_config()
            qconfig_constraints = []
            if self._hw_precision_constraints:
                qconfig_constraints = self._hw_precision_constraints.get(qid)
            if qconfig_constraints:
                configuration_space_in_trace_order.append(qconfig_constraints)
            else:
                configuration_space_in_trace_order.append([default_qconfig])
                trace_order_indices_of_defaulted_configs.add(trace_idx)

        matcher = TraceOrderBitwidthMatcher(self._bits, traces_order)
        return matcher.get_qconfig_sequences_constrained_by_trace_order(configuration_space_in_trace_order,
                                                                        trace_order_indices_of_defaulted_configs)


    @staticmethod
    def _filter_configs_by_precision_constraints(bits_configurations: List[List[int]],
                                                 hw_precision_constraints: HardwareQuantizationConstraints,
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

    def _get_weight_qp_ids_in_trace_order(self, traces_order: TracesOrder) -> List[Set[QuantizationPointId]]:
        quant_module_ids = list(self._weight_quantizations_by_execution_order.keys())
        qp_ids_in_trace_order = []
        for trace_idx in range(len(traces_order)):
            exec_idx = traces_order.get_execution_index_by_traces_index(trace_idx)
            quant_module_id = quant_module_ids[exec_idx]
            qp_ids_in_trace_order.append(self._algo.module_id_to_qp_id_translation_dict[quant_module_id])
        return qp_ids_in_trace_order

    def _apply_weight_configuration_to_quantizer_setup(self,
                                                       configuration: CoveringConfigurationForQuantNoiseCalculation,
                                                       qp_ids_in_trace_order: List[Set[QuantizationPointId]],
                                                       quantizer_setup: SingleConfigQuantizerSetup) -> \
            SingleConfigQuantizerSetup:
        retval = deepcopy(quantizer_setup)
        assert len(configuration) == len(qp_ids_in_trace_order)
        for trace_idx, qp_id_set in enumerate(qp_ids_in_trace_order):
            for qp_id in qp_id_set:
                retval.quantization_points[qp_id].qconfig = deepcopy(configuration[trace_idx])
        return retval

    def calc_quantization_noise(self, configurations_to_run: List[CoveringConfigurationForQuantNoiseCalculation],
                                traces_order: TracesOrder) -> Tuple[Perturbations, List[List[PerturbationObserver]]]:
        perturbations = Perturbations()
        qp_ids_in_trace_order = self._get_weight_qp_ids_in_trace_order(traces_order)
        ctrl = self._algo
        observers_for_all_configurations = []  # type: List[List[PerturbationObserver]]
        for configuration in configurations_to_run:
            quantizer_setup_to_run = self._apply_weight_configuration_to_quantizer_setup(
                configuration,
                qp_ids_in_trace_order,
                ctrl.get_quantizer_setup_for_current_state())
            ctrl, model = ctrl.apply_new_quantizer_setup(
                quantizer_setup_to_run)  # type: Tuple[ExperimentalQuantizationController, NNCFNetwork]

            hook_handles = []
            observers = []
            for qp_id_set in qp_ids_in_trace_order:
                for qp_id in qp_id_set:
                    wq_id = ctrl.setup_to_module_id_translation_dict[qp_id]
                    wq_module = ctrl.weight_quantizers[wq_id].quantizer_module_ref
                    observer = PerturbationObserver(self._init_device)
                    hook_handles.append(wq_module.register_forward_hook(observer.calc_perturbation))
                    observers.append(observer)

            model.do_dummy_forward(force_eval=True)

            for i, observer in enumerate(observers):
                perturbations.add(layer_id=traces_order.get_execution_index_by_traces_index(i),
                                  qconfig=configuration[i],
                                  perturbation=observer.get_observation().to(self._init_device))

            for handle in hook_handles:
                handle.remove()
            observers_for_all_configurations.append(observers)

        return perturbations, observers_for_all_configurations

    @staticmethod
    def calc_hawq_metric_per_configuration(trace_ordered_configurations: List[List[QuantizerConfig]],
                                           perturbations: Perturbations,
                                           traces_per_layer: TracesPerLayer, device) -> List[Tensor]:
        configuration_metric = []
        for trace_ordered_configuration in trace_ordered_configurations:
            hawq_metric = torch.Tensor([0]).to(device)
            for trace_index, qconfig in enumerate(trace_ordered_configuration):
                execution_index = traces_per_layer.traces_order.get_execution_index_by_traces_index(trace_index)
                hawq_metric += traces_per_layer.get_by_trace_index(trace_index) * perturbations.get(
                    layer_id=execution_index, qconfig=qconfig)
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

    def set_chosen_config(self, weight_qconfigs_in_traces_order: ConfigurationForHAWQToEvaluate,
                          traces_order: TracesOrder) -> SingleConfigQuantizerSetup:
        qp_ids_in_trace_order = self._get_weight_qp_ids_in_trace_order(traces_order)
        ctrl = self._algo

        quantizer_setup_to_set = self._apply_weight_configuration_to_quantizer_setup(
            weight_qconfigs_in_traces_order,
            qp_ids_in_trace_order,
            ctrl.get_quantizer_setup_for_current_state())
        if quantizer_setup_to_set.shared_input_operation_set_groups:
            for group in quantizer_setup_to_set.shared_input_operation_set_groups:
                weight_qp_ids = []
                act_qp_ids = []
                for qp_id in group:
                    qp = quantizer_setup_to_set.quantization_points[qp_id]
                    if qp.is_weight_quantization_point():
                        weight_qp_ids.append(qp_id)
                    elif qp.is_activation_quantization_point():
                        act_qp_ids.append(qp_id)
                weight_qps = [quantizer_setup_to_set.quantization_points[qp_id] for qp_id in weight_qp_ids]
                weight_bitwidth_set = {weight_qp.qconfig.bits for weight_qp in weight_qps}

                if self._bitwidth_assignment_mode == BitwidthAssignmentMode.STRICT:
                    quantizer_setup_to_set = self._set_activations_bitwidth_strictly(quantizer_setup_to_set,
                                                                                     act_qp_ids,
                                                                                     weight_bitwidth_set)
                else:
                    quantizer_setup_to_set = self._set_activation_bitwidth_liberally(quantizer_setup_to_set,
                                                                                     act_qp_ids,
                                                                                     weight_bitwidth_set)
        else:
            # TODO: delete not-consistent pairs of activation and weights for pattern-based approach
            pairs = self._algo.get_weights_activation_quantizers_pairs()
            for pair in pairs:
                wq_ids, aq_id = pair
                aq_qp_ids = ctrl.module_id_to_qp_id_translation_dict[aq_id]
                wq_qp_ids = set()
                for wq_id in wq_ids:
                    wq_qp_id_set = ctrl.module_id_to_qp_id_translation_dict[wq_id]
                    wq_qp_ids.update(list(wq_qp_id_set))

                wq_bits = [quantizer_setup_to_set.quantization_points[wq_qp_id].qconfig.bits for wq_qp_id in wq_qp_ids]
                for aq_qp_id in aq_qp_ids:
                    quantizer_setup_to_set.quantization_points[aq_qp_id].qconfig.bits = max(wq_bits)

        return quantizer_setup_to_set

    def _set_activation_bitwidth_liberally(self, quantizer_setup_to_set: SingleConfigQuantizerSetup,
                                           act_qp_ids: List[QuantizationPointId],
                                           weight_bitwidth_set: Set[int]) -> SingleConfigQuantizerSetup:
        for act_qp_id in act_qp_ids:
            original_quant_module_id = self._original_qp_id_vs_quantizer_module_id_dict[act_qp_id]
            activation_bitwidths_vs_qconfigs = self._hw_precision_constraints.get_bitwidth_vs_qconfigs_dict(
                original_quant_module_id)
            activation_bitwidth_set = set(activation_bitwidths_vs_qconfigs.keys())
            intersection = activation_bitwidth_set.intersection(weight_bitwidth_set)
            target_qp = quantizer_setup_to_set.quantization_points[act_qp_id]
            if activation_bitwidth_set.__len__() == 1:
                target_bits = activation_bitwidth_set.pop()
            elif intersection:
                target_bits = min(intersection)
            elif activation_bitwidth_set:
                target_bits = min(activation_bitwidth_set)
            elif weight_bitwidth_set:
                target_bits = min(weight_bitwidth_set)
            else:
                continue

            if activation_bitwidths_vs_qconfigs:
                target_qp.qconfig = deepcopy(activation_bitwidths_vs_qconfigs[target_bits][0])
            else:
                # The activation has no constraints, so the config in the setup was defaulted
                # and we can simply adjust the bitwidth
                target_qp.qconfig.bits = target_bits

        return quantizer_setup_to_set

    def _set_activations_bitwidth_strictly(self, quantizer_setup_to_set: SingleConfigQuantizerSetup,
                                           act_qp_ids: List[QuantizationPointId],
                                           weight_bitwidth_set: Set[int]) -> SingleConfigQuantizerSetup:
        if len(weight_bitwidth_set) > 1:
            raise RuntimeError('Invalid grouping of weight quantizers')
        all_constraints = set()
        original_quant_module_ids = [self._original_qp_id_vs_quantizer_module_id_dict[act_qp_id]
                                     for act_qp_id in act_qp_ids]
        for act_quant_module_id in original_quant_module_ids:
            all_constraints.update(self._hw_precision_constraints.get_all_unique_bits(act_quant_module_id))
        common_constraints = set(all_constraints)
        for act_quant_module_id in original_quant_module_ids:
            constraint = self._hw_precision_constraints.get_all_unique_bits(act_quant_module_id)
            common_constraints = common_constraints.intersection(constraint)
        if weight_bitwidth_set:
            common_constraints = common_constraints.intersection(weight_bitwidth_set)
        if not common_constraints:
            raise RuntimeError('No hardware compatible bitwidth for activation quantizers')
        for act_qp_id in act_qp_ids:
            quant_id = self._original_qp_id_vs_quantizer_module_id_dict[act_qp_id]
            target_bitwidth = sorted(list(common_constraints))[0]
            bitwidths_vs_qconfigs = self._hw_precision_constraints.get_bitwidth_vs_qconfigs_dict(quant_id)
            qconfig_to_select = bitwidths_vs_qconfigs[target_bitwidth][0]
            quantizer_setup_to_set.quantization_points[act_qp_id].qconfig = qconfig_to_select

        return quantizer_setup_to_set


    def get_metric_per_layer(self, chosen_config_in_execution_order: List[int], perturbations: Perturbations,
                             traces_per_layer: TracesPerLayer):
        metric_per_layer = []
        for i, layer_bits in enumerate(chosen_config_in_execution_order):
            metric_per_layer.append(traces_per_layer.get_by_execution_index(i) * perturbations.get(i, layer_bits))
        ordered_metric_per_layer = [i[0] for i in
                                    sorted(enumerate(metric_per_layer), reverse=True, key=lambda x: x[1])]
        return ordered_metric_per_layer

    @staticmethod
    def _filter_configs_by_grouped_weight_quantizers(trace_ordered_configurations: List[List[QuantizerConfig]],
                                                     weight_quantization_ids_by_execution_order: List[QuantizerId],
                                                     groups_of_adjacent_quantizers: GroupsOfAdjacentQuantizers,
                                                     traces_order: TracesOrder) -> List[List[QuantizerConfig]]:
        """ removes configs where adjacent weight quantizers have different bitwidth. Adjacency is defined by common
        activation quantizers"""
        filtered_bits_configurations = []
        all_grouped_indexes = []
        for group_of_adjacent_quantizers in groups_of_adjacent_quantizers:
            wqs = group_of_adjacent_quantizers.weight_quantizers
            if len(wqs) > 1:
                indexes_of_grouped_wq = []
                for quantizer_id, _ in wqs:
                    if quantizer_id in weight_quantization_ids_by_execution_order:
                        index_by_execution_order = weight_quantization_ids_by_execution_order.index(quantizer_id)
                        indexes_of_grouped_wq.append(index_by_execution_order)
                all_grouped_indexes.append(indexes_of_grouped_wq)

        if not all_grouped_indexes:
            return trace_ordered_configurations

        for qconf_configuration in trace_ordered_configurations:
            execution_ordered_configuration = traces_order.get_execution_order_config(qconf_configuration)
            bit_sequence = [qc.bits for qc in execution_ordered_configuration]
            keep_config = True
            for indexes_of_grouped_wq in all_grouped_indexes:
                grouped_bits = [bit_sequence[index] for index in indexes_of_grouped_wq]
                if grouped_bits[1:] != grouped_bits[:-1]:
                    keep_config = False
                    break
            if keep_config:
                filtered_bits_configurations.append(qconf_configuration)

        return filtered_bits_configurations
