"""
 Copyright (c) 2022 Intel Corporation
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
import json
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Set, Tuple

import torch
import warnings
from bisect import bisect_left
from copy import deepcopy
from operator import itemgetter
from torch import Tensor
from torch import nn
from torch.nn.modules.loss import _Loss

from nncf.common.graph import NNCFNodeName
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.utils.os import safe_open
from nncf.common.utils.debug import is_debug
from nncf.torch.quantization.hessian_trace import HessianTraceEstimator
from nncf.torch.quantization.layers import QuantizersSwitcher
from nncf.torch.quantization.precision_constraints import HardwareQuantizationConstraints
from nncf.torch.quantization.precision_init.adjacent_quantizers import GroupsOfAdjacentQuantizers
from nncf.torch.quantization.precision_init.base_init import BasePrecisionInitParams
from nncf.torch.quantization.precision_init.base_init import BasePrecisionInitializer
from nncf.torch.quantization.precision_init.compression_ratio import CompressionRatioCalculator
from nncf.torch.quantization.precision_init.hawq_debug import HAWQDebugger
from nncf.torch.quantization.precision_init.perturbations import PerturbationObserver
from nncf.torch.quantization.precision_init.perturbations import Perturbations
from nncf.torch.quantization.precision_init.traces_order import TracesOrder
from nncf.torch.quantization.precision_init.traces_order import TracesPerLayer
from nncf.common.quantization.structs import QuantizerId
from nncf.common.quantization.structs import WeightQuantizerId
from nncf.common.quantization.quantizer_setup import QuantizationPointId
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.torch.quantization.structs import WeightQuantizerInfo
from nncf.torch.structures import QuantizationPrecisionInitArgs
from nncf.torch.utils import get_model_device


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
                 bitwidths: List[int] = None,
                 bitwidth_per_scope: List[List] = None,
                 traces_per_layer_path: str = None,
                 num_data_points: int = None,
                 iter_number: int = None,
                 tolerance: float = None,
                 compression_ratio: float = None,
                 dump_hawq_data: bool = None,
                 bitwidth_assignment_mode: BitwidthAssignmentMode = None):
        super().__init__(user_init_args)
        self.bitwidths = bitwidths
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
            bitwidths=hawq_init_config_dict.get('bits', [2, 4, 8]),
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


QConfigSequenceForHAWQToEvaluate = List[QuantizerConfig]
CoveringQConfigSequenceForQuantNoiseCalculation = List[QuantizerConfig]


class TraceOrderBitwidthMatcher:
    def __init__(self, available_bitwidths: List[int], traces_order: TracesOrder):
        self._available_bitwidths = available_bitwidths
        self._traces_order = traces_order
        self._bitwidth_sequences = self.get_all_non_decreasing_bitwidth_sequences()

    def get_all_non_decreasing_bitwidth_sequences(self) -> List[List[int]]:
        sequences = []
        bitwidths_ = deepcopy(self._available_bitwidths)
        seq_len = len(self._traces_order)
        if seq_len == 0:
            return sequences
        bitwidths = sorted(bitwidths_)
        m = len(bitwidths)
        L = seq_len
        for j in range(1, m + 1):
            for combo_bitwidths in itertools.combinations(bitwidths, j):
                for combo_partitions in itertools.combinations(list(range(1, L)), j - 1):
                    bit_config = []
                    prev_p = 0
                    for (p, b) in zip(combo_partitions + (L,), combo_bitwidths):
                        bit_config += [b] * (p - prev_p)
                        prev_p = p
                    sequences.append(bit_config)
        return sequences

    @staticmethod
    def _select_first_closest_bitwidth_qconfig(qconf_list: List[QuantizerConfig],
                                               target_bitwidth: int) -> QuantizerConfig:
        bw_diffs = [abs(qc.num_bits - target_bitwidth) for qc in qconf_list]
        _, min_idx = min((val, idx) for (idx, val) in enumerate(bw_diffs))
        return qconf_list[min_idx]

    def _deduplicate(self, qconf_sequences_to_search: List[QConfigSequenceForHAWQToEvaluate]) -> \
            List[QConfigSequenceForHAWQToEvaluate]:
        tupled_sequence = [tuple(seq) for seq in qconf_sequences_to_search]
        odict = OrderedDict.fromkeys(tupled_sequence)
        deduped_tupled_sequence = list(odict.keys())
        return [list(tup) for tup in deduped_tupled_sequence]

    @staticmethod
    def _generate_covering_qconfig_sequences(observed_qconfs: List[Dict[QuantizerConfig, QuantizerConfig]]):
        covering_qconfig_sequences = []  # type: List[CoveringQConfigSequenceForQuantNoiseCalculation]
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
            covering_conf = []  # type: CoveringQConfigSequenceForQuantNoiseCalculation
            for qconfs_for_trace_idx in optimized_observed_qconfs:
                if i < len(qconfs_for_trace_idx):
                    covering_conf.append(qconfs_for_trace_idx[i])
                else:
                    covering_conf.append(qconfs_for_trace_idx[-1])
            covering_qconfig_sequences.append(covering_conf)
        return covering_qconfig_sequences

    def get_qconfig_sequences_constrained_by_trace_order(self,
                                                         possible_qconfigs_sequence_in_trace_order: List[
                                                             List[QuantizerConfig]],
                                                         indices_for_bitwidth_adjustment_only: Set[int]) -> \
            Tuple[List[QConfigSequenceForHAWQToEvaluate], List[CoveringQConfigSequenceForQuantNoiseCalculation]]:
        """
        The 'constraint' is so that the each qconfig sequence should have non-decreasing bitwidths. It
        might be impossible to apply this constraint for a given qconfig space (consider [[2], [6, 8], [4]]).
        In such a case, for trace order index positions where it was impossible to select a bitwidth so that the entire
        sequence is non-decreasing, the bitwidth closest to this target will be chosen instead.
        """
        if len(possible_qconfigs_sequence_in_trace_order) != len(self._traces_order):
            raise ValueError("The size of the qconfig space and the traces do not match!")
        retval = []  # type: List[QConfigSequenceForHAWQToEvaluate]
        observed_qconfs_in_retval = [OrderedDict()
                                     for _ in range(len(self._traces_order))]
        for bitwidth_sequence in self._bitwidth_sequences:
            current_qconfig_sequence_in_trace_order = []  # type: QConfigSequenceForHAWQToEvaluate
            for trace_idx, bitwidth in enumerate(bitwidth_sequence):

                if trace_idx in indices_for_bitwidth_adjustment_only:
                    bitwidth_adjusted_default_qconfig = deepcopy(
                        possible_qconfigs_sequence_in_trace_order[trace_idx][0])
                    bitwidth_adjusted_default_qconfig.num_bits = bitwidth
                    qconfig = bitwidth_adjusted_default_qconfig
                else:
                    # TODO: do a selection based on strategy ("exhaustive" = add all available configurations,
                    # "preset" = do a selection based on a certain preset, "first" = select first match (as below),
                    # "custom" = use a custom selection function to be passed as arg to the HAWQ initializer
                    # OR: do non-bitwidth disambiguation higher up the stack, make sure that the qconfig
                    # space at this spot only has 1 qconfig option for each bitwidth.
                    possible_qconfigs_for_current_trace_idx = possible_qconfigs_sequence_in_trace_order[trace_idx]
                    first_closest_qconfig = self._select_first_closest_bitwidth_qconfig(
                        possible_qconfigs_for_current_trace_idx, bitwidth)
                    qconfig = deepcopy(first_closest_qconfig)

                current_qconfig_sequence_in_trace_order.append(qconfig)
                observed_qconfs_in_retval[trace_idx][qconfig] = qconfig
            retval.append(current_qconfig_sequence_in_trace_order)
        return self._deduplicate(retval), self._generate_covering_qconfig_sequences(observed_qconfs_in_retval)


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
        self._bitwidths = self._hw_precision_constraints.get_all_unique_bitwidths() \
            if self._hw_precision_constraints else params.bitwidths
        self._init_device = init_args.device
        if self._init_device is None:
            self._init_device = get_model_device(self._model)
        current_quantizer_setup = self._algo.get_quantizer_setup_for_current_state()
        flops_per_module = self._model.get_flops_per_module()
        self._compression_ratio_calculator = CompressionRatioCalculator(
            flops_per_module, current_quantizer_setup,
            self._groups_of_adjacent_quantizers.weight_qp_id_per_activation_qp_id)
        self._dump_hawq_data = params.dump_hawq_data
        self._original_qp_id_vs_quantizer_module_id_dict = deepcopy(algo.setup_to_module_id_translation_dict)

    def apply_init(self) -> SingleConfigQuantizerSetup:
        if not self._weight_quantizations_by_execution_order:
            return self._algo.get_quantizer_setup_for_current_state()

        original_device = get_model_device(self._model)
        self._model.to(self._init_device)

        traces_per_layer = self._calc_traces(self._criterion_fn, self._criterion, self._iter_number, self._tolerance)
        if not traces_per_layer:
            raise RuntimeError('Failed to calculate hessian traces!')

        traces_order = traces_per_layer.traces_order
        weight_qconfig_sequences_in_trace_order, covering_qconfig_sequences = \
            self.get_qconfig_sequences_constrained_by_traces_order(traces_order)

        weight_quantizer_ids_in_execution_order = list(self._weight_quantizations_by_execution_order.keys())

        if not weight_qconfig_sequences_in_trace_order:
            warnings.warn('All bitwidths configurations are incompatible with HW Config!', RuntimeWarning)
            return None

        weight_qconfig_sequences_in_trace_order = \
            self._filter_qconfig_sequences_by_excessive_bitwidth(weight_qconfig_sequences_in_trace_order)

        if self._bitwidth_assignment_mode == BitwidthAssignmentMode.STRICT:
            weight_qconfig_sequences_in_trace_order = \
                self._filter_qconfig_sequences_by_grouped_weight_quantizers(weight_qconfig_sequences_in_trace_order,
                                                                            weight_quantizer_ids_in_execution_order,
                                                                            self._groups_of_adjacent_quantizers,
                                                                            traces_order)
        if not weight_qconfig_sequences_in_trace_order:
            warnings.warn('No bitwidths configurations are left after removing inconsistent groups of weight quantizers'
                          ' with adjacent activation quantizers!', RuntimeWarning)
            return self._algo.get_quantizer_setup_for_current_state()

        compression_ratio_per_qconfig = self.get_compression_ratio_per_qconfig_sequence(
            weight_qconfig_sequences_in_trace_order,
            traces_order)
        min_ratio = min(compression_ratio_per_qconfig)
        max_ratio = max(compression_ratio_per_qconfig)
        if not min_ratio <= self._compression_ratio <= max_ratio:
            raise AttributeError('Invalid compression ratio={}. Should be within range [{:.3f}, {:.3f}]'.format(
                self._compression_ratio, min_ratio, max_ratio))

        perturbations, weight_observers = self.calc_quantization_noise(covering_qconfig_sequences, traces_order)

        metric_per_qconfig_sequence = self.calc_hawq_metric_per_qconfig_sequence(
            weight_qconfig_sequences_in_trace_order, perturbations,
            traces_per_layer, self._init_device)

        qconfig_sequence_index = self.choose_qconfig_sequence(
            metric_per_qconfig_sequence, compression_ratio_per_qconfig, self._compression_ratio)
        chosen_qconfig_sequence_in_traces_order = weight_qconfig_sequences_in_trace_order[qconfig_sequence_index]
        chosen_qconfig_sequence_in_execution_order = traces_order.get_execution_order_configs(
            chosen_qconfig_sequence_in_traces_order)
        bitwidth_sequence = [qconfig.num_bits for qconfig in chosen_qconfig_sequence_in_execution_order]
        nncf_logger.info('Chosen HAWQ bitwidth sequence with ratio={:.2f}, bitwidth per weightable layer={}'.format(
            compression_ratio_per_qconfig[qconfig_sequence_index], bitwidth_sequence))
        nncf_logger.debug('Order of the weightable layers in the HAWQ bitwidth sequence (in descending order of average'
                          ' Hessian traces) ={}'.format(traces_order))

        final_quantizer_setup = self.get_quantizer_setup_for_qconfig_sequence(chosen_qconfig_sequence_in_traces_order,
                                                                              traces_order)
        if is_debug() or self._dump_hawq_data:
            hawq_debugger = HAWQDebugger(weight_qconfig_sequences_in_trace_order,
                                         perturbations,
                                         weight_observers, traces_per_layer, self._bitwidths)
            hawq_debugger.dump_metric_MB(metric_per_qconfig_sequence)
            hawq_debugger.dump_metric_flops(
                metric_per_qconfig_sequence, compression_ratio_per_qconfig, qconfig_sequence_index)
            hawq_debugger.dump_avg_traces()
            hawq_debugger.dump_density_of_quantization_noise()
            hawq_debugger.dump_perturbations_ratio()
            new_ctrl, new_model = self._algo.apply_new_quantizer_setup(final_quantizer_setup)
            groups_of_adjacent_quantizers = new_ctrl.groups_of_adjacent_quantizers
            hawq_debugger.dump_bitwidth_graph(new_ctrl, new_model, groups_of_adjacent_quantizers)
        bitwidth_per_scope = self.get_bitwidth_per_scope(final_quantizer_setup)
        from nncf.common.utils.debug import DEBUG_LOG_DIR
        Path(DEBUG_LOG_DIR).mkdir(parents=True, exist_ok=True)
        with safe_open(Path(DEBUG_LOG_DIR) / 'bitwidth_per_scope.json', "w") as outfile:
            json.dump({'bitwidth_per_scope': bitwidth_per_scope}, outfile, indent=4, sort_keys=False)
        self._model.to(original_device)
        return final_quantizer_setup

    @staticmethod
    def _merge_constraints_for_adjacent_quantizers(groups_of_adjacent_quantizers: GroupsOfAdjacentQuantizers,
                                                   hw_precision_constraints: HardwareQuantizationConstraints) -> \
            HardwareQuantizationConstraints:
        if not hw_precision_constraints:
            return None
        retval = deepcopy(hw_precision_constraints)
        for group in groups_of_adjacent_quantizers:
            all_bitwidths_sets = []
            quantizer_ids = []
            all_quantizers = group.weight_quantizers + group.activation_quantizers
            for quantizer_id, _ in all_quantizers:
                bitwidths_vs_qconfig_sequence = retval.get_bitwidth_vs_qconfigs_dict(quantizer_id)
                bitwidths = set(bitwidths_vs_qconfig_sequence.keys())
                all_bitwidths_sets.append(bitwidths)
                quantizer_ids.append(quantizer_id)
            minimal_set_bitwidths = set.intersection(*all_bitwidths_sets)
            if not minimal_set_bitwidths:
                raise RuntimeError(
                    'No bitwidths configurations are left after removing inconsistent groups of weight quantizers'
                    ' with adjacent activation quantizers!')
            for quantizer_id in quantizer_ids:
                qconfig_sequence = retval.get(quantizer_id)
                filtered_qconfig_sequence = []
                for qconf in qconfig_sequence:
                    if qconf.num_bits in minimal_set_bitwidths:
                        filtered_qconfig_sequence.append(qconf)
                retval.replace(quantizer_id, filtered_qconfig_sequence)
        return retval

    def get_compression_ratio_per_qconfig_sequence(self,
                                                   qconfig_sequences_in_trace_order: List[
                                                       QConfigSequenceForHAWQToEvaluate],
                                                   traces_order: TracesOrder) -> List[float]:
        compression_ratio_per_qconfig = []
        for qconfig_sequence in qconfig_sequences_in_trace_order:
            quantizer_setup = self.get_quantizer_setup_for_qconfig_sequence(qconfig_sequence, traces_order)
            compression_ratio = self._compression_ratio_calculator.run_for_quantizer_setup(quantizer_setup)
            compression_ratio_per_qconfig.append(compression_ratio)
        return compression_ratio_per_qconfig

    class ParamsToRestore(NamedTuple):
        originally_disabled_gradients: List[str]
        skipped_gradients_to_enable: List[Tuple[nn.Module, str]]

    @staticmethod
    def disable_all_gradients_except_weights_of_quantized_modules(
            quantizers_switcher: QuantizersSwitcher,
            weight_quantizers: Dict[WeightQuantizerId, WeightQuantizerInfo],
            model: nn.Module,
            skipped_quantized_weight_node_names: List[NNCFNodeName] = None) -> ParamsToRestore:
        """
        Disables gradients of all parameters, except for layers that have quantizers for weights, which wasn't skipped
        because of single precision constraints.
        :param quantizers_switcher: object that is responsible for enabling and disabling quantizers
        :param weight_quantizers: modules with quantized weights per scope
        :param model: model to access all parameters
        :param skipped_quantized_weight_node_names: list of weighted nodes that have a single precision
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
            target_node_name = wq_id.target_node_name
            is_skipped = False
            for skipped_wt_node_name in skipped_quantized_weight_node_names:
                if skipped_wt_node_name == target_node_name:
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
            self._quantizers_handler.get_skipped_quantized_weight_node_names())

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

    def get_qconfig_sequences_constrained_by_traces_order(self, traces_order: TracesOrder) -> \
            Tuple[List[QConfigSequenceForHAWQToEvaluate], List[CoveringQConfigSequenceForQuantNoiseCalculation]]:
        possible_qconfigs_sequence_in_trace_order = []  # type: List[List[QuantizerConfig]]
        trace_order_indices_of_defaulted_qconfig_sequence = set()  # type: Set[int]
        quantizer_ids_in_exec_order = list(self._weight_quantizations_by_execution_order.keys())
        assert len(quantizer_ids_in_exec_order) == len(traces_order)
        for trace_idx in range(len(traces_order)):
            exec_idx = traces_order.get_execution_index_by_traces_index(trace_idx)
            qid = quantizer_ids_in_exec_order[exec_idx]
            default_qconfig = self._weight_quantizations_by_execution_order[qid].get_quantizer_config()
            qconfig_constraints = []
            if self._hw_precision_constraints:
                qconfig_constraints = self._hw_precision_constraints.get(qid)
            if qconfig_constraints:
                possible_qconfigs_sequence_in_trace_order.append(qconfig_constraints)
            else:
                possible_qconfigs_sequence_in_trace_order.append([default_qconfig])
                trace_order_indices_of_defaulted_qconfig_sequence.add(trace_idx)

        matcher = TraceOrderBitwidthMatcher(self._bitwidths, traces_order)
        return matcher.get_qconfig_sequences_constrained_by_trace_order(
            possible_qconfigs_sequence_in_trace_order, trace_order_indices_of_defaulted_qconfig_sequence)

    def _get_weight_qp_ids_in_trace_order(self, traces_order: TracesOrder) -> List[Set[QuantizationPointId]]:
        quant_module_ids = list(self._weight_quantizations_by_execution_order.keys())
        qp_ids_in_trace_order = []
        for trace_idx in range(len(traces_order)):
            exec_idx = traces_order.get_execution_index_by_traces_index(trace_idx)
            quant_module_id = quant_module_ids[exec_idx]
            qp_ids_in_trace_order.append(self._algo.module_id_to_qp_id_translation_dict[quant_module_id])
        return qp_ids_in_trace_order

    @staticmethod
    def _apply_qconfig_sequence_to_quantizer_setup(qconfig_sequence: CoveringQConfigSequenceForQuantNoiseCalculation,
                                                   qp_ids_in_trace_order: List[Set[QuantizationPointId]],
                                                   quantizer_setup: SingleConfigQuantizerSetup) -> \
            SingleConfigQuantizerSetup:
        retval = deepcopy(quantizer_setup)
        assert len(qconfig_sequence) == len(qp_ids_in_trace_order)
        for trace_idx, qp_id_set in enumerate(qp_ids_in_trace_order):
            for qp_id in qp_id_set:
                retval.quantization_points[qp_id].qconfig = deepcopy(qconfig_sequence[trace_idx])
        return retval

    def calc_quantization_noise(self, qconfig_sequences_to_run: List[CoveringQConfigSequenceForQuantNoiseCalculation],
                                traces_order: TracesOrder) -> Tuple[Perturbations, List[List[PerturbationObserver]]]:
        perturbations = Perturbations()
        qp_ids_in_trace_order = self._get_weight_qp_ids_in_trace_order(traces_order)
        ctrl = self._algo
        observers_for_all_qconfig_sequences = []  # type: List[List[PerturbationObserver]]
        for qconfig_sequence in qconfig_sequences_to_run:
            quantizer_setup_to_run = self._apply_qconfig_sequence_to_quantizer_setup(
                qconfig_sequence,
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
                                  qconfig=qconfig_sequence[i],
                                  perturbation=observer.get_observation().to(self._init_device))

            for handle in hook_handles:
                handle.remove()
            observers_for_all_qconfig_sequences.append(observers)

        return perturbations, observers_for_all_qconfig_sequences

    @staticmethod
    def calc_hawq_metric_per_qconfig_sequence(qconfig_sequences_in_trace_order: List[QConfigSequenceForHAWQToEvaluate],
                                              perturbations: Perturbations,
                                              traces_per_layer: TracesPerLayer, device) -> List[Tensor]:
        metric_per_qconfig_sequence = []
        for qconfig_sequence_in_trace_order in qconfig_sequences_in_trace_order:
            hawq_metric = torch.Tensor([0]).to(device)
            for trace_index, qconfig in enumerate(qconfig_sequence_in_trace_order):
                execution_index = traces_per_layer.traces_order.get_execution_index_by_traces_index(trace_index)
                hawq_metric += traces_per_layer.get_by_trace_index(trace_index) * perturbations.get(
                    layer_id=execution_index, qconfig=qconfig)
            metric_per_qconfig_sequence.append(hawq_metric)
        return metric_per_qconfig_sequence

    @staticmethod
    def choose_qconfig_sequence(metric_per_qconfig_sequences: List[Tensor],
                                compression_ratio_per_qconfig: List[float],
                                compression_ratio) -> int:
        num_qconfig_sequences = len(metric_per_qconfig_sequences)

        sorted_compression_ratio_per_qconfig = sorted(compression_ratio_per_qconfig)
        indexes_of_sorted_compression_ratio = [x[0] for x in
                                               sorted(enumerate(compression_ratio_per_qconfig), reverse=False,
                                                      key=lambda x: x[1])]

        boundary_index = bisect_left(sorted_compression_ratio_per_qconfig, compression_ratio)
        indexes_to_check = [indexes_of_sorted_compression_ratio[i] for i in
                            range(boundary_index, num_qconfig_sequences)]
        best_metric = min(list(itemgetter(*indexes_to_check)(metric_per_qconfig_sequences)))
        best_qconfig_sequence_index = metric_per_qconfig_sequences.index(best_metric)
        return best_qconfig_sequence_index

    def get_quantizer_setup_for_qconfig_sequence(self,
                                                 qconfig_sequence_in_traces_order: QConfigSequenceForHAWQToEvaluate,
                                                 traces_order: TracesOrder) -> SingleConfigQuantizerSetup:
        wqp_ids_in_trace_order = self._get_weight_qp_ids_in_trace_order(traces_order)
        ctrl = self._algo

        quantizer_setup_to_set = self._apply_qconfig_sequence_to_quantizer_setup(
            qconfig_sequence_in_traces_order,
            wqp_ids_in_trace_order,
            ctrl.get_quantizer_setup_for_current_state())

        assert quantizer_setup_to_set.shared_input_operation_set_groups
        for group in quantizer_setup_to_set.shared_input_operation_set_groups.values():
            weight_qp_ids = []
            act_qp_ids = []
            for qp_id in group:
                qp = quantizer_setup_to_set.quantization_points[qp_id]
                if qp.is_weight_quantization_point():
                    weight_qp_ids.append(qp_id)
                elif qp.is_activation_quantization_point():
                    act_qp_ids.append(qp_id)
            weight_qps = [quantizer_setup_to_set.quantization_points[qp_id] for qp_id in weight_qp_ids]
            weight_bitwidth_set = {weight_qp.qconfig.num_bits for weight_qp in weight_qps}

            if self._bitwidth_assignment_mode == BitwidthAssignmentMode.STRICT:
                quantizer_setup_to_set = self._set_activations_bitwidth_strictly(quantizer_setup_to_set,
                                                                                 act_qp_ids,
                                                                                 weight_bitwidth_set)
            else:
                quantizer_setup_to_set = self._set_activation_bitwidth_liberally(quantizer_setup_to_set,
                                                                                 act_qp_ids,
                                                                                 weight_bitwidth_set)

        return quantizer_setup_to_set

    def _set_activation_bitwidth_liberally(self, quantizer_setup_to_set: SingleConfigQuantizerSetup,
                                           act_qp_ids: List[QuantizationPointId],
                                           weight_bitwidth_set: Set[int]) -> SingleConfigQuantizerSetup:
        for act_qp_id in act_qp_ids:
            original_quant_module_id = self._original_qp_id_vs_quantizer_module_id_dict[act_qp_id]
            activation_bitwidths_vs_qconfig_sequence = self._hw_precision_constraints.get_bitwidth_vs_qconfigs_dict(
                original_quant_module_id)
            activation_bitwidth_set = set(activation_bitwidths_vs_qconfig_sequence.keys())
            intersection = activation_bitwidth_set.intersection(weight_bitwidth_set)
            target_qp = quantizer_setup_to_set.quantization_points[act_qp_id]
            if activation_bitwidth_set.__len__() == 1:
                target_bitwidth = activation_bitwidth_set.pop()
            elif intersection:
                target_bitwidth = min(intersection)
            elif activation_bitwidth_set:
                target_bitwidth = min(activation_bitwidth_set)
            elif weight_bitwidth_set:
                target_bitwidth = min(weight_bitwidth_set)
            else:
                continue

            if activation_bitwidths_vs_qconfig_sequence:
                target_qp.qconfig = deepcopy(activation_bitwidths_vs_qconfig_sequence[target_bitwidth][0])
            else:
                # The activation has no constraints, so the config in the setup was defaulted
                # and we can simply adjust the bitwidth
                target_qp.qconfig.num_bits = target_bitwidth

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
            all_constraints.update(self._hw_precision_constraints.get_all_unique_bitwidths(act_quant_module_id))
        common_constraints = set(all_constraints)
        for act_quant_module_id in original_quant_module_ids:
            constraint = self._hw_precision_constraints.get_all_unique_bitwidths(act_quant_module_id)
            common_constraints = common_constraints.intersection(constraint)
        if weight_bitwidth_set:
            common_constraints = common_constraints.intersection(weight_bitwidth_set)
        if not common_constraints:
            raise RuntimeError('No hardware compatible bitwidth for activation quantizers')
        for act_qp_id in act_qp_ids:
            quant_id = self._original_qp_id_vs_quantizer_module_id_dict[act_qp_id]
            target_bitwidth = sorted(list(common_constraints))[0]
            bitwidths_vs_qconfig_sequence = self._hw_precision_constraints.get_bitwidth_vs_qconfigs_dict(
                quant_id)
            qconfig_to_select = bitwidths_vs_qconfig_sequence[target_bitwidth][0]
            quantizer_setup_to_set.quantization_points[act_qp_id].qconfig = qconfig_to_select

        return quantizer_setup_to_set

    @staticmethod
    def _filter_qconfig_sequences_by_grouped_weight_quantizers(
            trace_ordered_qconfig_sequences: List[QConfigSequenceForHAWQToEvaluate],
            weight_quantization_ids_by_execution_order: List[QuantizerId],
            groups_of_adjacent_quantizers: GroupsOfAdjacentQuantizers,
            traces_order: TracesOrder) -> List[QConfigSequenceForHAWQToEvaluate]:
        """
        Removes configs where adjacent weight quantizers have different bitwidth. Adjacency is defined by common
        activation quantizers
        """
        filtered_qconfig_sequences = []
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
            return trace_ordered_qconfig_sequences

        for qconfig_sequence in trace_ordered_qconfig_sequences:
            execution_ordered_qconfig_sequence = traces_order.get_execution_order_configs(qconfig_sequence)
            bitwidth_sequence = [qconfig.num_bits for qconfig in execution_ordered_qconfig_sequence]
            keep_config = True
            for indexes_of_grouped_wq in all_grouped_indexes:
                grouped_bits = [bitwidth_sequence[index] for index in indexes_of_grouped_wq]
                if grouped_bits[1:] != grouped_bits[:-1]:
                    keep_config = False
                    break
            if keep_config:
                filtered_qconfig_sequences.append(qconfig_sequence)

        return filtered_qconfig_sequences

    def _filter_qconfig_sequences_by_excessive_bitwidth(self,
                                                        weight_qconfig_sequences_in_trace_order: List[
                                                            QConfigSequenceForHAWQToEvaluate]) \
            -> List[QConfigSequenceForHAWQToEvaluate]:
        result = weight_qconfig_sequences_in_trace_order
        if self._hw_precision_constraints:
            all_weight_bitwidths = set()
            for wq_id in self._algo.weight_quantizers:
                all_weight_bitwidths.update(self._hw_precision_constraints.get_all_unique_bitwidths(wq_id))

            all_activation_bitwidths = set()
            for aq_id in self._algo.non_weight_quantizers:
                all_activation_bitwidths.update(self._hw_precision_constraints.get_all_unique_bitwidths(aq_id))

            excessive_weight_bitwidths = all_weight_bitwidths - all_activation_bitwidths

            def filter_fn(qconfig_sequence: QConfigSequenceForHAWQToEvaluate):
                all_qconfig_bitwidths = set(map(lambda qconfig: qconfig.num_bits, qconfig_sequence))
                return any(map(lambda x: x not in all_qconfig_bitwidths, excessive_weight_bitwidths))

            if excessive_weight_bitwidths:
                result = list(filter(filter_fn, weight_qconfig_sequences_in_trace_order))
        return result
