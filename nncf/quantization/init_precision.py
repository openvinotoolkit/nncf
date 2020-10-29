import itertools
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict, Union, Tuple, NamedTuple

import os
import torch
import warnings

from bisect import bisect_left
from operator import itemgetter
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss

from nncf.debug import is_debug
from nncf.dynamic_graph.context import no_nncf_trace, Scope
from nncf.nncf_logger import logger as nncf_logger
from nncf.nncf_network import NNCFNetwork, CompressionModuleType
from nncf.quantization.layers import QUANTIZATION_MODULES, QuantizersSwitcher, BaseQuantizer
from .hessian_trace import HessianTraceEstimator
from .hw_precision_constraints import HWPrecisionConstraints
from .quantizer_id import QuantizerId
from ..dynamic_graph.graph import NNCFGraph
from ..dynamic_graph.transform_graph import is_nncf_module
from ..layers import NNCFConv2d
from ..structures import QuantizationPrecisionInitArgs
from ..utils import in_scope_list, get_all_modules_by_type


class ManualPrecisionInitializer:
    def __init__(self, algo: 'QuantizationController', config: 'NNCFConfig',
                 init_args: QuantizationPrecisionInitArgs = None):
        self._algo = algo
        self._model = self._algo._model  # type: NNCFNetwork
        all_quantizers = algo.all_quantizations
        self._bitwidth_per_scope = config.get('bitwidth_per_scope', {})  # type: List[List]
        self._hw_precision_constraints = algo._hw_precision_constraints
        self.original_precisions = {q_id: quantizer.num_bits for q_id, quantizer in all_quantizers.items()}
        self._quantizers_handler = WeightQuantizersHandler(self._model, all_quantizers,
                                                           self._hw_precision_constraints)

        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        self._ordered_weight_quantizations = self._quantizers_handler.get_ordered_weight_quantizers_per_id()

        self._all_quantizers_per_scope = get_all_modules_by_type(
            self._model.get_compression_modules_by_type(CompressionModuleType.ACTIVATION_QUANTIZER), quantization_types)
        self._all_quantizers_per_scope.update(get_all_modules_by_type(
            self._model.get_compression_modules_by_type(CompressionModuleType.FUNCTION_QUANTIZER), quantization_types))
        self._all_quantizers_per_scope.update(self._quantizers_handler.get_all_ordered_weight_quantizers_per_scope())

    def apply_init(self):
        for pair in self._bitwidth_per_scope:
            if len(pair) != 2:
                raise ValueError('Invalid format of bitwidth per scope: [int, str] is expected')
            bitwidth = pair[0]
            scope_name = pair[1]
            is_matched = False
            for scope, quantizer in self._all_quantizers_per_scope.items():
                if in_scope_list(str(scope), scope_name):
                    quantizer.num_bits = bitwidth
                    is_matched = True
            if not is_matched:
                raise ValueError(
                    'Invalid scope name `{}`, failed to assign bitwidth {} to it'.format(scope_name, bitwidth))


class PerturbationObserver:
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.perturbation = None
        self.numels = None

    def calc_perturbation(self, module, inputs: torch.Tensor, output: torch.Tensor):
        input_ = inputs[0] if isinstance(inputs, tuple) else inputs
        with no_nncf_trace():
            self.perturbation = torch.norm(input_ - output, p=2) ** 2
            self.numels = input_.size().numel()
            self.input_norm = torch.norm(input_, p=2) ** 2

    def reset(self):
        self.perturbation = None
        self.numels = None

    def get_observation(self):
        return self.perturbation

    def get_numels(self):
        return self.numels

    def get_input_norm(self):
        return self.input_norm


class Perturbations:
    def __init__(self):
        self._perturbations = {}  # type: Dict[int, Dict[int, Tensor]]

    def add(self, layer_id: int, bitwidth: int, perturbation: Tensor):
        if layer_id in self._perturbations:
            self._perturbations[layer_id].update({bitwidth: perturbation})
        else:
            self._perturbations[layer_id] = {bitwidth: perturbation}

    def get(self, layer_id: int, bitwidth: int) -> Tensor:
        layer_perturbations = self._perturbations[layer_id]
        return layer_perturbations[bitwidth]

    def get_all(self) -> Dict[int, Dict[int, Tensor]]:
        return self._perturbations


class TracesPerLayer:
    def __init__(self, traces_per_layer: Tensor):
        self._traces_per_layer = traces_per_layer
        self._traces_order = [i[0] for i in
                              sorted(enumerate(traces_per_layer), reverse=False, key=lambda x: x[1])]

    def get(self, index: int) -> Tensor:
        return self._traces_per_layer[index]

    def get_order_of_traces(self) -> List[int]:
        return self._traces_order

    def get_all(self) -> Tensor:
        return self._traces_per_layer

    def __bool__(self):
        return bool(self._traces_order)


class HAWQPrecisionInitializer(ManualPrecisionInitializer):
    def __init__(self, algo: 'QuantizationController', config: 'NNCFConfig',
                 init_args: QuantizationPrecisionInitArgs):
        super().__init__(algo, config, init_args)
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

    def apply_init(self):
        bits_configurations = None
        if len(self._quantizers_handler.get_ordered_weight_quantizers_per_id()) != 0:
            original_device = next(self._model.parameters()).device
            self._model.to(self._init_device)

            traces_per_layer = self._calc_traces(self._criterion, self._iter_number, self._tolerance)
            if not traces_per_layer:
                raise RuntimeError('Failed to calculate hessian traces!')

            traces_order = traces_per_layer.get_order_of_traces()
            num_weights = len(self._ordered_weight_quantizations)
            bits_configurations = self.get_configs_constrained_by_order(self._bits, num_weights)

            ordered_weight_quantization_ids = list(self._ordered_weight_quantizations.keys())
            bits_configurations = self._filter_configs_by_precision_constraints(bits_configurations,
                                                                                self._hw_precision_constraints,
                                                                                ordered_weight_quantization_ids,
                                                                                traces_order)
        ordered_metric_per_layer = None
        if not bits_configurations:
            warnings.warn('All bits configurations are incompatible with HW Config!', RuntimeWarning)
        else:
            perturbations, weight_observers = self.calc_quantization_noise()

            configuration_metric = self.calc_hawq_metric_per_configuration(bits_configurations, perturbations,
                                                                           traces_per_layer, self._init_device)

            flops_bits_per_config = self.get_flops_bits_per_config(bits_configurations, traces_order)
            config_index = self.choose_configuration(configuration_metric, flops_bits_per_config)
            chosen_config_per_layer = bits_configurations[config_index]
            chosen_config_per_layer = self.get_ordered_config(chosen_config_per_layer, traces_order)
            nncf_logger.info('Chosen HAWQ configuration with ratio={:.2f}, bitwidth per weightable layer={}'.format(
                flops_bits_per_config[config_index], chosen_config_per_layer))
            nncf_logger.debug('Order of the weightable layers in the HAWQ configuration={}'.format(traces_order))

            self.set_chosen_config(chosen_config_per_layer)

            if is_debug():
                hawq_debugger = HAWQDebugger(bits_configurations, perturbations,
                                             weight_observers, traces_per_layer, self._bits)
                hawq_debugger.dump_metric_MB(configuration_metric)
                hawq_debugger.dump_metric_flops(configuration_metric, flops_bits_per_config, config_index)
                hawq_debugger.dump_avg_traces()
                hawq_debugger.dump_density_of_quantization_noise()
                hawq_debugger.dump_perturbations_ratio()
                hawq_debugger.dump_bitwidth_graph(self._algo, self._model)

            self._model.rebuild_graph()
            str_bw = [str(element) for element in self.get_bitwidth_per_scope()]
            nncf_logger.info('\n'.join(['\n\"bitwidth_per_scope\": [', ',\n'.join(str_bw), ']']))

            self._model.to(original_device)

            ordered_metric_per_layer = self.get_metric_per_layer(chosen_config_per_layer,
                                                                 perturbations,
                                                                 traces_per_layer)
        return ordered_metric_per_layer

    def get_flops_bits_per_config(self, bits_configurations, traces_order):
        skipped = self._quantizers_handler.get_skipped_weight_quantizers_per_id()
        flops_bits_per_config = []
        for bits_config in bits_configurations:
            bits_config = self.get_ordered_config(bits_config, traces_order)
            flops_bits_per_config.append(self.flops_counter.ratio_for_bits_configuration(bits_config, skipped))
        return flops_bits_per_config

    def get_bitwidth_per_scope(self) -> List[List[Union[int, str]]]:
        sorted_quantizers = OrderedDict(sorted(self._all_quantizers_per_scope.items(), key=lambda x: str(x[0])))
        full_bitwidth_per_scope = []
        for scope, quantizer in sorted_quantizers.items():
            quantizer_id = self._quantizers_handler.get_id(quantizer)
            if quantizer.num_bits != self.original_precisions[quantizer_id]:
                full_bitwidth_per_scope.append([quantizer.num_bits, str(scope)])
        return full_bitwidth_per_scope

    @staticmethod
    def get_ordered_config(bit_configuration: List[int], order: List[int]) -> List[int]:
        ordered_config = [0] * len(bit_configuration)
        for i, bitwidth in enumerate(bit_configuration):
            ordered_config[order[i]] = bitwidth
        return ordered_config

    class ParamsToRestore(NamedTuple):
        originally_disabled_gradients: List[str]
        skipped_gradients_to_enable: List[Tuple[nn.Module, str]]

    @staticmethod
    def disable_all_gradients_except_weights_of_quantized_modules(
            quantizers_switcher: QuantizersSwitcher,
            quantized_weight_modules_registry: Dict[str, torch.nn.Module],
            model: nn.Module,
            scopes_of_skipped_weight_quantizers: List['Scope'] = None) -> ParamsToRestore: # pylint: disable=undefined-variable
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

    def _calc_traces(self, criterion: _Loss, iter_number: int, tolerance: float) -> TracesPerLayer:
        if self._traces_per_layer_path:
            return TracesPerLayer(torch.load(self._traces_per_layer_path).to(self._init_device))

        quantizers_switcher = QuantizersSwitcher(list(self._all_quantizers_per_scope.values()))
        params_to_restore = self.disable_all_gradients_except_weights_of_quantized_modules(
            quantizers_switcher,
            self._algo.quantized_weight_modules_registry,
            self._model,
            self._quantizers_handler.get_scope_of_skipped_weight_quantizers())

        trace_estimator = HessianTraceEstimator(self._model, criterion, self._init_device, self._data_loader,
                                                self._num_data_points)
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
    def get_configs_constrained_by_order(bits_: List[int], num_layers: int) -> List[List[int]]:
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
                                                 traces_order: List[int]) -> List[List[int]]:
        if not hw_precision_constraints:
            return bits_configurations

        filtered_bits_configurations = []
        for bits_configuration in bits_configurations:
            is_all_bitwidth_compatible = True
            for i, bitwidth in enumerate(bits_configuration):
                weight_id = ordered_weight_ids[traces_order[i]]
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
        for module in self._ordered_weight_quantizations.values():
            observer = PerturbationObserver(self._init_device)
            hook_handles.append(module.register_forward_hook(observer.calc_perturbation))
            observers.append(observer)

        perturbations = Perturbations()
        for b in self._bits:
            for wi in self._ordered_weight_quantizations.values():
                wi.num_bits = b

            self._model.do_dummy_forward(force_eval=True)

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
            for i, layer_bits in enumerate(bits_config):
                order = traces_per_layer.get_order_of_traces()[i]
                hawq_metric += traces_per_layer.get(order) * perturbations.get(layer_id=order,
                                                                               bitwidth=layer_bits)
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

    def set_chosen_config(self, weight_bits_per_layer: List[int]):
        for wq, bits in zip(self._ordered_weight_quantizations.values(), weight_bits_per_layer):
            wq.num_bits = bits
        pairs = self._algo.get_weights_activation_quantizers_pairs()
        for pair in pairs:
            wqs, aq = pair
            aq.num_bits = max([wq.num_bits for wq in wqs])

    def get_metric_per_layer(self, chosen_config_per_layer: List[int], perturbations: Perturbations,
                             traces_per_layer: TracesPerLayer):
        metric_per_layer = []
        for i, layer_bits in enumerate(chosen_config_per_layer):
            metric_per_layer.append(traces_per_layer.get(i) * perturbations.get(i, layer_bits))
        ordered_metric_per_layer = [i[0] for i in
                                    sorted(enumerate(metric_per_layer), reverse=True, key=lambda x: x[1])]
        return ordered_metric_per_layer


class WeightQuantizersHandler:
    """
    Defines weight quantizers for precision initialization in the order of execution.
    """

    def __init__(self, model, all_quantizers: Dict[QuantizerId, BaseQuantizer], constraints: HWPrecisionConstraints):
        self._quantizer_address_to_id_mapping = {id(quantizer): q_id for q_id, quantizer in all_quantizers.items()}
        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        weight_module_dict = model.get_nncf_wrapped_model()
        self._ordered_weight_quantizers_per_scope = get_all_modules_by_type(weight_module_dict, quantization_types)
        ordered_weight_quantization_list = []
        self._scopes_of_skipped_weight_quantizers = []
        self._skipped_weight_quantizers = {}
        for scope, quantizer in self._ordered_weight_quantizers_per_scope.items():
            address = id(quantizer)
            if quantizer.is_weights:
                quantizer_id = self._quantizer_address_to_id_mapping[address]
                # no need to init quantizer with single precision constraint
                if len(constraints.get(quantizer_id)) != 1:
                    ordered_weight_quantization_list.append((quantizer_id, quantizer))
                else:
                    self._scopes_of_skipped_weight_quantizers.append(scope)
                    self._skipped_weight_quantizers[quantizer_id] = quantizer
        self._ordered_weight_quantizations = OrderedDict(ordered_weight_quantization_list)

    def get_scope_of_skipped_weight_quantizers(self) -> List['Scope']:
        return self._scopes_of_skipped_weight_quantizers

    def get_all_ordered_weight_quantizers_per_scope(self) -> Dict['Scope', BaseQuantizer]:
        return self._ordered_weight_quantizers_per_scope

    def get_ordered_weight_quantizers_per_id(self) -> Dict[QuantizerId, BaseQuantizer]:
        return self._ordered_weight_quantizations

    def get_id(self, quantizer: BaseQuantizer) -> Dict[int, QuantizerId]:
        address = id(quantizer)
        return self._quantizer_address_to_id_mapping[address]

    def get_skipped_weight_quantizers_per_id(self) -> Dict[QuantizerId, BaseQuantizer]:
        return self._skipped_weight_quantizers


class CompressionRatioCalculator:
    """
    Calculates compression ratio - ratio between bits complexity of fully INT8 model and mixed-precision lower-bit one.
    Bit complexity of the model is a sum of bit complexities for each quantized layer, which are a multiplication of
    FLOPS for the layer by number of bits for its quantization. The compression ratio can be used for estimation of
    performance boost for quantized model.
    """
    DEFAULT_NUMBER_OF_BITS = 8

    def __init__(self, model, quantizers_handler: WeightQuantizersHandler):
        flops_count_per_module_name = model.get_flops_per_module()

        self._ordered_weight_quantizations = quantizers_handler.get_ordered_weight_quantizers_per_id()

        self.ops_per_quantizer_id = {}
        for name, module in model.named_modules():
            curr_ops = flops_count_per_module_name.get(name, 0)
            if is_nncf_module(module):
                quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
                all_quantizers_in_module = get_all_modules_by_type(module, quantization_types)
                for quantizer in all_quantizers_in_module.values():
                    if quantizer.is_weights:
                        quantizer_id = quantizers_handler.get_id(quantizer)
                        self.ops_per_quantizer_id[quantizer_id] = curr_ops

        self.total_ops_count = sum(v for v in self.ops_per_quantizer_id.values()) * self.DEFAULT_NUMBER_OF_BITS

    def ratio_for_bits_configuration(self, bits_config: List[int],
                                     skipped: Dict[QuantizerId, BaseQuantizer] = None) -> float:
        """
        Calculates compression ratio for a given bits configuration

        Args:
            bits_config: list of bits for each weight quantization
            skipped: quantizers that were skipped from bitwidth initialization, since their bitwidth is determined
            unambiguously based on constraints of the HW config

        Returns:
            compression ratio of mixed-precision model by relation to fully INT8
        """
        quantizer_ops = 0
        for num_bits, (quantizer_id, quantizer) in zip(bits_config, self._ordered_weight_quantizations.items()):
            quantizer_ops += num_bits * self.ops_per_quantizer_id[quantizer_id]
        if skipped:
            for quantizer_id, quantizer in skipped.items():
                quantizer_ops += quantizer.num_bits * self.ops_per_quantizer_id[quantizer_id]

        return self.total_ops_count / quantizer_ops

    def ratio_limits(self, bits: List[int], order: List[int] = None, constraints: HWPrecisionConstraints = None,
                     skipped: Dict[QuantizerId, BaseQuantizer] = None) -> (float, float):
        """
        Calculates minimum and maximum compression ratio.

        Args:
            bits: list of all available bits for weight quantization
            order: defines the order in which bits are assigned for quantizers
            constraints: precision constraints defined by HW config
            skipped: quantizers that were skipped from bitwidth initialization, since their bitwidth is determined
            unambiguously based on constraints of the HW config

        Returns:
            minimum and maximum compression ratio
        """
        config_len = len(self._ordered_weight_quantizations)
        min_config = [min(bits)] * config_len
        max_config = [max(bits)] * config_len
        if not order:
            order = list(range(config_len))
        if constraints:
            for i, quantizer_id in enumerate(self._ordered_weight_quantizations):
                bit_constraints = constraints.get(quantizer_id)
                if bit_constraints:
                    min_config[order[i]] = min(bit_constraints)
                    max_config[order[i]] = max(bit_constraints)

        min_config = HAWQPrecisionInitializer.get_ordered_config(min_config, order)
        max_config = HAWQPrecisionInitializer.get_ordered_config(max_config, order)

        max_ratio = self.ratio_for_bits_configuration(min_config, skipped)
        min_ratio = self.ratio_for_bits_configuration(max_config, skipped)
        return min_ratio, max_ratio


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

        self._traces_order = traces_per_layer.get_order_of_traces()
        self._traces_per_layer = traces_per_layer.get_all()

        num_of_weights = []
        norm_of_weights = []
        for i in range(self._num_weights):
            order = self._traces_order[i]
            num_of_weights.append(weight_observers[order].get_numels())
            norm_of_weights.append(weight_observers[order].get_input_norm())
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

    @staticmethod
    def get_bitwidth_graph(algo_ctrl, model, all_quantizers_per_full_scope) -> NNCFGraph:
        nncf_graph = model.get_graph()
        for node_key in nncf_graph.get_all_node_keys():
            node = nncf_graph.get_nx_node_by_key(node_key)
            node_id = node[NNCFGraph.ID_NODE_ATTR]
            color = ''
            if node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR]:
                operator_name = node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].operator_name
                scope = node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].input_agnostic.scope_in_model
                module = model.get_module_by_scope(scope)
                if isinstance(module, NNCFConv2d):
                    color = 'blue'
                    if module.groups == module.in_channels:
                        operator_name = 'DW_Conv2d'
                        color = 'purple'

                node['label'] = '_#'.join([operator_name, str(node_id)])
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
                node_id = post_hooked_module_node[NNCFGraph.ID_NODE_ATTR]
                post_hooked_module_node['label'] = '_#'.join([operator_name, str(node_id)])

                for next_nx_node_key in nncf_graph.get_successors(post_hooked_nx_node_key):
                    activation_fq_node = nncf_graph.get_nx_node_by_key(next_nx_node_key)
                    bits = non_weight_quantizers[quantizer_id].quantizer_module_ref.num_bits

                    activation_fq_node['color'] = bits_color_map[bits]
                    node_id = activation_fq_node[NNCFGraph.ID_NODE_ATTR]
                    activation_fq_node['label'] = '{}_bit__AFQ_#{}'.format(bits, str(node_id))

        for scope, quantizer in all_quantizers_per_full_scope.items():
            if quantizer.is_weights:
                node = nncf_graph.find_node_in_nx_graph_by_scope(scope)
                if not node:
                    raise AttributeError('Failed to get node by scope={}'.format(str(scope)))
                if node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR]:
                    bits = quantizer.num_bits
                    node_id = node[NNCFGraph.ID_NODE_ATTR]
                    node['label'] = '{}_bit__WFQ_#{}'.format(bits, str(node_id))
                    node['color'] = bits_color_map[bits]
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
                order = self._traces_order[i]
                qnoise += self._perturbations.get(layer_id=order, bitwidth=layer_bits)
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

    def dump_bitwidth_graph(self, algo_ctrl: 'QuantizationController', model: 'NNCFNetwork'):
        all_quantizers_per_full_scope = self.get_all_quantizers_per_full_scope(model)
        graph = self.get_bitwidth_graph(algo_ctrl, model, all_quantizers_per_full_scope)
        graph.dump_graph(self._dump_dir / Path('bitwidth_graph.dot'))


class PrecisionInitializerFactory:
    @staticmethod
    def create(init_type: str):
        if init_type == "manual":
            return ManualPrecisionInitializer
        if init_type == "hawq":
            return HAWQPrecisionInitializer
        raise NotImplementedError
