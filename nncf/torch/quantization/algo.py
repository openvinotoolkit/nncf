"""
 Copyright (c) 2019-2021 Intel Corporation
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

import shutil
# pylint:disable=too-many-lines
from collections import Counter
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from string import Template
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import networkx as nx
import numpy as np
import torch
from torch import nn

from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.api.compression import CompressionStage
from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.common.graph import MODEL_INPUT_OP_NAME
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.hardware.config import HWConfig
from nncf.common.hardware.config import HWConfigType
from nncf.common.quantization.statistics import QuantizationStatistics
from nncf.common.quantization.structs import NonWeightQuantizerId
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.structs import QuantizationConstraints
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.quantization.structs import QuantizerId
from nncf.common.quantization.structs import WeightQuantizerId
from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.helpers import matches_any
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.utils.os import safe_open
from nncf.config import NNCFConfig
from nncf.config.extractors import extract_bn_adaptation_init_params
from nncf.config.extractors import extract_range_init_params
from nncf.torch.algo_selector import COMPRESSION_ALGORITHMS
from nncf.torch.algo_selector import ZeroCompressionLoss
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.debug import CallCountTracker
from nncf.torch.debug import DebugInterface
from nncf.torch.debug import is_debug
from nncf.torch.dynamic_graph.context import TracingContext
from nncf.torch.graph.operator_metatypes import Conv2dMetatype
from nncf.torch.graph.operator_metatypes import DepthwiseConv2dSubtype
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.commands import TransformationPriority
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.initialization import SimpleDataLoaderRunner
from nncf.torch.module_operations import UpdatePaddingValue
from nncf.torch.nncf_network import EXTERNAL_QUANTIZERS_STORAGE_NAME
from nncf.torch.nncf_network import ExtraCompressionModuleType
from nncf.torch.nncf_network import InsertionPointGraph
from nncf.torch.nncf_network import InsertionPointGraphNodeType
from nncf.torch.nncf_network import LoadStateListener
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.adjust_padding import AdjustPaddingArgs
from nncf.torch.quantization.adjust_padding import CalculatePaddingAdjustment
from nncf.torch.quantization.init_precision import PrecisionInitializerFactory
from nncf.torch.quantization.init_range import DataLoaderRangeInitializeRunner
from nncf.torch.quantization.init_range import PTRangeInitParams
from nncf.torch.quantization.init_range import StatCollectorGenerator
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.layers import QuantizationMode
from nncf.torch.quantization.layers import QuantizerConfig
from nncf.torch.quantization.layers import QuantizerExportMode
from nncf.torch.quantization.layers import QuantizersSwitcher
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.quantization.metrics import BitwidthDistributionStatisticsCollector
from nncf.torch.quantization.metrics import MemoryConsumptionStatisticsCollector
from nncf.torch.quantization.metrics import QuantizationShareBuildTimeInfo
from nncf.torch.quantization.metrics import QuantizationShareStatisticsCollector
from nncf.torch.quantization.metrics import ShareEdgesQuantizedDataPathStatisticsCollector
from nncf.torch.quantization.node_matcher import PTOperatorMetatypeNodeMatcher
from nncf.torch.quantization.precision_constraints import HardwareQuantizationConstraints
from nncf.torch.quantization.precision_init.adjacent_quantizers import GroupsOfAdjacentQuantizers
from nncf.torch.quantization.precision_init.autoq_init import AutoQPrecisionInitParams
from nncf.torch.quantization.precision_init.base_init import BasePrecisionInitParams
from nncf.torch.quantization.precision_init.hawq_init import HAWQPrecisionInitParams
from nncf.torch.quantization.precision_init.manual_init import ManualPrecisionInitParams
from nncf.torch.quantization.quantizer_propagation import QuantizerPropagationSolver
from nncf.torch.quantization.quantizer_propagation import QuantizerPropagationStateGraph
from nncf.torch.quantization.quantizer_setup import MultiConfigQuantizerSetup
from nncf.torch.quantization.quantizer_setup import QuantizationPointId
from nncf.torch.quantization.quantizer_setup import QuantizerSetupBase
from nncf.torch.quantization.quantizer_setup import SingleConfigQuantizationPoint
from nncf.torch.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.torch.quantization.schedulers import QUANTIZATION_SCHEDULERS
from nncf.torch.quantization.structs import NonWeightQuantizerInfo
from nncf.torch.quantization.structs import WeightQuantizerInfo
from nncf.torch.structures import AutoQPrecisionInitArgs
from nncf.torch.structures import QuantizationPrecisionInitArgs
from nncf.torch.tensor_statistics.algo import TensorStatisticsCollectionBuilder
from nncf.torch.tensor_statistics.collectors import ReductionShape
from nncf.torch.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.torch.tensor_statistics.statistics import TensorStatistic
from nncf.torch.utils import get_scale_shape
from nncf.torch.utils import get_state_dict_names_with_modules
from nncf.torch.utils import is_main_process


class QuantizerSetupGeneratorBase:
    DEFAULT_QUANTIZER_CONFIG = QuantizerConfig(num_bits=8,
                                               mode=QuantizationMode.SYMMETRIC,
                                               signedness_to_force=None,
                                               per_channel=False)

    def __init__(self, quant_config: NNCFConfig,
                 target_model: NNCFNetwork,
                 precision_init_type: str = None,
                 precision_init_params: BasePrecisionInitParams = None,
                 range_init_params: PTRangeInitParams = None):
        self._target_model = target_model
        self._quantization_config = quant_config

        self._quantize_inputs = self._quantization_config.get('quantize_inputs', True)
        self._quantize_outputs = self._quantization_config.get('quantize_outputs', False)

        self.ignored_scopes = self._quantization_config.get('ignored_scopes')
        self.target_scopes = self._quantization_config.get('target_scopes')

        self.global_quantizer_constraints = {}  # type: Dict[QuantizerGroup, QuantizationConstraints]
        self._ignored_scopes_per_group = {}  # type: Dict[QuantizerGroup, List[str]]
        self._target_scopes_per_group = {}  # type: Dict[QuantizerGroup, List[str]]

        for quantizer_group in QuantizerGroup:
            self._parse_group_params(self._quantization_config, quantizer_group)

        self._precision_init_type = precision_init_type
        self._precision_init_params = precision_init_params
        self._range_init_params = range_init_params
        self._num_potential_quantized_weights = len(self._target_model.get_nncf_modules())

    def generate_setup(self) -> SingleConfigQuantizerSetup:
        raise NotImplementedError

    def get_build_time_metric_infos(self):
        raise NotImplementedError

    def _parse_group_params(self, quant_config: 'NNCFConfig', quantizer_group: QuantizerGroup):
        group_name = quantizer_group.value
        params_dict = quant_config.get(group_name, {})
        self.global_quantizer_constraints[quantizer_group] = QuantizationConstraints.from_config_dict(params_dict)
        self._ignored_scopes_per_group[quantizer_group] = params_dict.get('ignored_scopes')
        self._target_scopes_per_group[quantizer_group] = params_dict.get('target_scopes')

    @staticmethod
    def get_scoped_quantizer_config(base_config: QuantizerConfig,
                                    parent_module_scope_str: str,
                                    scope_overrides: Dict = None) -> QuantizerConfig:
        qconfig = deepcopy(base_config)
        if scope_overrides is None:
            scope_overrides = {}
        for overridden_scope in scope_overrides.keys():
            if matches_any(parent_module_scope_str, overridden_scope):
                config_overrides = scope_overrides[overridden_scope]
                if config_overrides.get("bits") is not None:
                    qconfig.num_bits = config_overrides["bits"]
                if config_overrides.get("mode") is not None:
                    qconfig.mode = config_overrides["mode"]
                if config_overrides.get("per_channel") is not None:
                    qconfig.per_channel = config_overrides["per_channel"]
                if config_overrides.get("signed") is not None:
                    qconfig.signedness_to_force = config_overrides["signed"]
        return qconfig

    def _get_default_qconfig(self, constraints: QuantizationConstraints = None):
        qconfig = deepcopy(self.DEFAULT_QUANTIZER_CONFIG)
        if constraints is not None:
            qconfig = constraints.apply_constraints_to(qconfig)
        return qconfig

    def _should_consider_scope_for_group(self, node_name: NNCFNodeName, group: QuantizerGroup) -> bool:
        if self.target_scopes is not None or self._target_scopes_per_group[group] is not None:
            if matches_any(node_name, self.target_scopes):
                return True
            if matches_any(node_name, self._target_scopes_per_group[group]):
                return True

            return False

        if matches_any(node_name, self.ignored_scopes):
            return False
        if matches_any(node_name, self._ignored_scopes_per_group[group]):
            return False

        return True

    def _filter_by_ignored_algo(self, nodes: List[NNCFNode]) -> List[NNCFNode]:
        retval = []
        for node in nodes:
            if 'quantization' in node.ignored_algorithms:
                continue
            retval.append(node)
        return retval

    def _filter_by_weight_ignored_target_scopes(self, weighted_nodes: List[NNCFNode]) -> List[NNCFNode]:
        retval = []
        for node in weighted_nodes:
            if not self._should_consider_scope_for_group(node.node_name, QuantizerGroup.WEIGHTS):
                nncf_logger.info("Ignored adding Weight quantizer for: {}".format(node.node_name))
                continue
            retval.append(node)
        return retval

    def _assign_qconfig_lists_to_modules(self, weighted_nodes: List[NNCFNode]) -> \
        Dict[NNCFNode, List[QuantizerConfig]]:
        raise NotImplementedError

    def get_quantizable_module_nodes(self) -> List[QuantizableWeightedLayerNode]:
        weighted_nodes = self._target_model.get_weighted_original_graph_nodes()
        quantized_modules_with_potential_qconfig = []

        weighted_nodes = self._filter_by_ignored_algo(weighted_nodes)
        weighted_nodes = self._filter_by_weight_ignored_target_scopes(weighted_nodes)
        weighted_node_vs_qconfig_list = self._assign_qconfig_lists_to_modules(weighted_nodes)

        for node, qconfig_list in weighted_node_vs_qconfig_list.items():
            if qconfig_list is not None:
                qconfig_list_copy = deepcopy(qconfig_list)
                quantized_modules_with_potential_qconfig.append(QuantizableWeightedLayerNode(node,
                                                                                             qconfig_list_copy))
        return quantized_modules_with_potential_qconfig


class IQuantizerSetupDisambiguator:
    def select_final_quantizer_setup(self, multi_config_setup: MultiConfigQuantizerSetup) -> SingleConfigQuantizerSetup:
        raise NotImplementedError


class DefaultQuantizerSetupDisambiguator(IQuantizerSetupDisambiguator):
    def __init__(self, target_model: NNCFNetwork,
                 precision_init_type: str = None,
                 precision_init_params: BasePrecisionInitParams = None,
                 range_init_params: PTRangeInitParams = None,
                 override_bit_options_with_precision_init: bool = False,
                 hw_config: HWConfig = None):
        self._precision_init_type = precision_init_type
        self._precision_init_params = precision_init_params
        self._range_init_params = range_init_params
        self._target_model = target_model
        self._override_bit_options_with_precision_init = override_bit_options_with_precision_init
        self.hw_config = hw_config

    @staticmethod
    def select_first_qconfig_with_bitwidth_variants_for_each_point(
            multi_config_setup: MultiConfigQuantizerSetup) -> MultiConfigQuantizerSetup:
        new_setup = deepcopy(multi_config_setup)
        for qp_id, qp in multi_config_setup.quantization_points.items():
            main_qconfig = qp.possible_qconfigs[0]
            constrained_qconfig_list = [main_qconfig]
            if len(qp.possible_qconfigs) > 1:
                constrained_qconfig_list += list(filter(main_qconfig.is_a_bitwidth_variant, qp.possible_qconfigs[1:]))
            new_setup.quantization_points[qp_id].possible_qconfigs = constrained_qconfig_list
        return new_setup

    def select_final_quantizer_setup(self, multi_config_setup: MultiConfigQuantizerSetup) -> SingleConfigQuantizerSetup:
        if self._precision_init_type is not None:
            with self._target_model.temporary_clean_view() as intermediate_model:
                stats = QuantizationBuilder.get_statistics_for_quantizer_setup(intermediate_model,
                                                                               multi_config_setup,
                                                                               self._range_init_params)
                bitwidth_varying_only_multi_setup = \
                    self.select_first_qconfig_with_bitwidth_variants_for_each_point(multi_config_setup)

                init_setup = bitwidth_varying_only_multi_setup.select_first_qconfig_for_each_point()
                intermediate_builder = ExperimentalQuantizationBuilder(bitwidth_varying_only_multi_setup,
                                                                       init_setup,
                                                                       stats, hw_config=self.hw_config)
                intermediate_builder.apply_to(intermediate_model)
                intermediate_ctrl = intermediate_builder.build_controller(intermediate_model)

                # intermediate_ctrl.init_range()
                hw_constraints = HardwareQuantizationConstraints()
                if not self._override_bit_options_with_precision_init:
                    for qp_id, qp in multi_config_setup.quantization_points.items():
                        quantizer_module_id = intermediate_ctrl.setup_to_module_id_translation_dict[qp_id]
                        hw_constraints.add(quantizer_module_id, qp.possible_qconfigs)
                final_quantizer_setup = intermediate_ctrl.init_precision(self._precision_init_type,
                                                                         self._precision_init_params,
                                                                         hw_constraints)
        else:
            final_quantizer_setup = multi_config_setup.select_first_qconfig_for_each_point()
        return final_quantizer_setup


class PropagationBasedQuantizerSetupGenerator(QuantizerSetupGeneratorBase):
    def __init__(self, quant_config: NNCFConfig, target_model: NNCFNetwork,
                 hw_config: HWConfig = None,
                 precision_init_type: str = None,
                 precision_init_params: BasePrecisionInitParams = None,
                 range_init_params: PTRangeInitParams = None,
                 debug_interface: 'QuantizationDebugInterface' = None):
        super().__init__(quant_config, target_model, precision_init_type, precision_init_params, range_init_params)
        self._quantizable_subgraph_patterns = quant_config.get('quantizable_subgraph_patterns', None)

        self.hw_config = hw_config

        self._hw_precision_constraints = HardwareQuantizationConstraints()
        self._debug_interface = debug_interface
        self._num_potential_quantized_activations = 0

        act_config = quant_config.get(QuantizerGroup.ACTIVATIONS.value, {})
        self._unified_scale_ops = act_config.get('unified_scale_ops')

    def generate_setup(self) -> SingleConfigQuantizerSetup:
        quantizable_module_nodes = self.get_quantizable_module_nodes()

        insertion_point_graph = self._target_model.get_insertion_point_graph()
        if self._debug_interface:
            self._debug_interface.visualize_insertion_point_graph(insertion_point_graph)
        prop_graph_solver = QuantizerPropagationSolver(
            ignored_scopes=self.ignored_scopes,
            target_scopes=self.target_scopes,
            debug_interface=self._debug_interface,
            hw_config=self.hw_config,
            default_qconfig_list=[self._get_default_qconfig(
                constraints=self.global_quantizer_constraints[
                    QuantizerGroup.ACTIVATIONS])],
            quantizable_module_nodes=quantizable_module_nodes,
            scope_overrides=self._quantization_config.get("scope_overrides", {}),
            global_constraints=self.global_quantizer_constraints,
            additional_unified_scale_op_scopes=self._unified_scale_ops,
            quantize_outputs=self._quantize_outputs)

        merged_ip_graph = insertion_point_graph.get_ip_graph_with_merged_hw_optimized_operations(
            self.hw_config,
            additional_patterns=self._quantizable_subgraph_patterns)
        quantization_proposal = prop_graph_solver.run_on_ip_graph(merged_ip_graph)
        self._num_potential_quantized_activations = prop_graph_solver.get_num_potential_quantized_activations()

        quantizer_setup = deepcopy(quantization_proposal.quantizer_setup)
        quantization_proposal.quantizer_setup = quantizer_setup

        disambiguator = DefaultQuantizerSetupDisambiguator(
            self._target_model,
            self._precision_init_type,
            self._precision_init_params,
            self._range_init_params,
            override_bit_options_with_precision_init=self.hw_config is None,
            hw_config=self.hw_config)

        single_config_quantizer_setup = disambiguator.select_final_quantizer_setup(
            quantization_proposal.quantizer_setup)

        finalized_proposal = quantization_proposal.finalize(single_config_quantizer_setup,
                                                            strict=self.hw_config is not None)
        finalized_quantizer_setup = prop_graph_solver.get_final_quantizer_setup(finalized_proposal)
        finalized_quantizer_setup = self._handle_quantize_inputs_option(finalized_quantizer_setup)
        return finalized_quantizer_setup


    def _assign_qconfig_lists_to_modules(self, nodes_with_weights: List[NNCFNode]) -> Dict[NNCFNode, QuantizerConfig]:
        retval = {}  # type: Dict[NNCFNode, QuantizerConfig]
        global_constraints = self.global_quantizer_constraints[QuantizerGroup.WEIGHTS]
        default_qconfig = self._get_default_qconfig(constraints=global_constraints)
        scope_overrides_dict = self._quantization_config.get("scope_overrides", {})
        weight_scope_overrides_dict = scope_overrides_dict.get("weights")
        if self.hw_config is not None:
            meta_vs_qconfig_map = self.hw_config.get_metatype_vs_quantizer_configs_map(for_weights=True)
        for node in nodes_with_weights:
            qconfig_for_current_scope = self.get_scoped_quantizer_config(default_qconfig,
                                                                         node.node_name,
                                                                         weight_scope_overrides_dict)
            if self.hw_config is None:
                qconfig_list = [qconfig_for_current_scope]
            else:
                metatype = PTOperatorMetatypeNodeMatcher.match(node)
                qconfig_list = meta_vs_qconfig_map[metatype]
                if HWConfig.is_wildcard_quantization(qconfig_list):  # Empty list = wildcard quantization
                    qconfig_list = [default_qconfig]
                elif HWConfig.is_qconf_list_corresponding_to_unspecified_op(qconfig_list):
                    continue  # The module will not have its weights quantized
                try:
                    local_constraints = global_constraints
                    for overridden_scope, scoped_override_dict in scope_overrides_dict.items():
                        if matches_any(node.node_name, overridden_scope):
                            scope_constraints = QuantizationConstraints.from_config_dict(scoped_override_dict)
                            local_constraints = local_constraints.get_updated_constraints(scope_constraints)
                    qconfig_list = local_constraints.constrain_qconfig_list(qconfig_list)

                except RuntimeError as e:
                    err_msg = "Quantization parameter constraints specified in NNCF config are incompatible with HW "
                    err_msg += "capabilities as specified in HW config type '{}'. ".format(self.hw_config.target_device)
                    err_msg += "First conflicting quantizer location: {}".format(str(node.node_name))
                    raise RuntimeError(err_msg) from e

            retval[node] = qconfig_list
        return retval

    def _handle_quantize_inputs_option(self, quantizer_setup: SingleConfigQuantizerSetup) -> SingleConfigQuantizerSetup:
        nncf_graph = self._target_model.get_original_graph()
        qp_ids_to_discard = []
        for qp_id, qp in quantizer_setup.quantization_points.items():
            if qp.is_activation_quantization_point():
                insertion_point = qp.insertion_point
                target_node = nncf_graph.get_node_by_name(insertion_point.target_node_name)
                if not self._quantize_inputs and target_node.node_type == MODEL_INPUT_OP_NAME:
                    qp_ids_to_discard.append(qp_id)
        for qp_id in qp_ids_to_discard:
            quantizer_setup.discard(qp_id, keep_shared_input_qps=True)
        return quantizer_setup

    def get_build_time_metric_infos(self):
        return QuantizationShareBuildTimeInfo(self._num_potential_quantized_activations,
                                              self._num_potential_quantized_weights)


@COMPRESSION_ALGORITHMS.register('quantization')
class QuantizationBuilder(PTCompressionAlgorithmBuilder):
    def __init__(self, config, should_init: bool = True):
        super().__init__(config, should_init)
        self._debug_interface = QuantizationDebugInterface() if is_debug() else None
        self._weight_quantizers = OrderedDict()  # Quantizers applied via UpdateWeights
        self._non_weight_quantizers = OrderedDict()  # All the other quantizers
        self._processed_insertion_points = set()  # type: Set[PTTargetPoint]
        self._groups_of_adjacent_quantizers = GroupsOfAdjacentQuantizers()  # type: GroupsOfAdjacentQuantizers
        self._setup_to_module_id_translation_dict = {}  # type: Dict[QuantizationPointId, QuantizerId]
        self.eval_ops_exec_ctx = []
        self._build_time_metric_infos = None
        self.hw_config = None

        # can be False to disable setting of adjust padding operations on precision init, because it may add unnecessary
        # noise on model evaluation (e.g. in AutoQ)
        self._should_setup_adjust_pad_ops = True

        hw_config_type = self.config.get("hw_config_type")
        if hw_config_type is not None:
            hw_config_path = HWConfig.get_path_to_hw_config(hw_config_type)
            self.hw_config = HWConfig.from_json(hw_config_path)

        self._range_init_params = None
        self._precision_init_type = None
        self._precision_init_params = None
        if self.should_init:
            self._parse_init_params()
        else:
            # TODO: remove it! It workarounds checkpoint loading for mixed precision model by forcing manual init
            self._force_manual_precision_init()

        self._use_logarithm_scale_per_group = {}  # type: Dict[QuantizerGroup, bool]

        for quantizer_group in QuantizerGroup:
            group_name = quantizer_group.value
            params_dict = self.config.get(group_name, {})
            self._use_logarithm_scale_per_group[quantizer_group] = params_dict.get('logarithm_scale', False)

        self._disable_saturation_fix = self.config.get('disable_saturation_fix', False)
        self._device_for_callable_obj_creation = 'cpu'

    def _force_manual_precision_init(self):
        init_config = self.config.get('initializer', {})
        init_precision_config = init_config.get('precision', None)
        if init_precision_config is not None:
            precision_init_type = init_precision_config.get('type', 'manual')
            if precision_init_type == 'manual':
                # range init is needed for correct setting of Adjust Padding ops as it considers sign of FQ
                self._range_init_params = self._parse_range_init_params()
                self.should_init = True
                self._precision_init_type = precision_init_type
                self._precision_init_params = ManualPrecisionInitParams.from_config(init_precision_config)

    def _parse_init_params(self):
        self._range_init_params = self._parse_range_init_params()
        self._precision_init_type, self._precision_init_params = self._parse_precision_init_params(
            self.config.get('initializer', {}))

    def _parse_range_init_params(self) -> Optional[PTRangeInitParams]:
        range_init_params = extract_range_init_params(self.config)
        return PTRangeInitParams(**range_init_params) if range_init_params is not None else None

    def _parse_precision_init_params(self, initializer_config: Dict) -> Tuple[str, BasePrecisionInitParams]:
        init_precision_config = initializer_config.get('precision', None)
        if not init_precision_config:
            return None, None
        precision_init_type = init_precision_config.get('type', 'manual')
        if precision_init_type == 'hawq':
            try:
                precision_init_args = self.config.get_extra_struct(QuantizationPrecisionInitArgs)
            except KeyError as e:
                raise ValueError(
                    'Specified non-manual precision initialization in the NNCF config, '
                    'but the initializing data loader and loss criterion are not provided as an extra struct. '
                    'Refer to `NNCFConfig.register_extra_structs` and the `QuantizationPrecisionInitArgs` '
                    'class') from e
            precision_init_params = HAWQPrecisionInitParams.from_config(
                init_precision_config,
                precision_init_args
            )
        elif precision_init_type == "autoq":
            if self.hw_config is not None and self.hw_config.target_device != HWConfigType.VPU.value:
                raise ValueError("Unsupported device ({}). Automatic Precision Initialization only supports for "
                                 "target_device NONE or VPU".format(self.hw_config.target_device))
            try:
                precision_init_args = self.config.get_extra_struct(AutoQPrecisionInitArgs)
            except KeyError as e:
                raise ValueError('Specified Automated precision initialization in the NNCF config, '
                                 'but the initializing data loader and loss criterion are not provided as an extra '
                                 'struct. Refer to `NNCFConfig.register_extra_structs` and the '
                                 '`AutoQPrecisionInitArgs` class') from e

            hw_config_type = None
            if self.hw_config is not None:
                hw_config_type = HWConfigType.from_str(self.hw_config.target_device)
            precision_init_params = AutoQPrecisionInitParams.from_config(init_precision_config,
                                                                         precision_init_args,
                                                                         hw_config_type)
        else:
            precision_init_params = ManualPrecisionInitParams.from_config(init_precision_config)

        return precision_init_type, precision_init_params

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        # TODO (vshampor): a simpler solution would be to always create callables on CPU and
        # to move these to model-specific device upon actual application, but would this impact
        # the time required to create a compressed model?
        self._device_for_callable_obj_creation = next(target_model.parameters()).device
        target_model_graph = target_model.get_original_graph()
        target_model.register_compression_module_type(ExtraCompressionModuleType.EXTERNAL_QUANTIZER)
        single_config_quantizer_setup = self._get_quantizer_setup(target_model)
        minmax_values_for_range_init = {}
        if is_main_process() and self.should_init:
            stats_for_range_init = self._get_statistics_for_final_range_init(target_model,
                                                                             single_config_quantizer_setup,
                                                                             self._range_init_params)
            minmax_values_for_range_init = single_config_quantizer_setup.get_minmax_values(stats_for_range_init,
                                                                                           target_model_graph)
        insertion_commands, setup_to_module_id_translation_dict = \
            self._build_insertion_commands_list_for_quantizer_setup(single_config_quantizer_setup,
                                                                    target_model,
                                                                    minmax_values_for_range_init)

        transformation_layout = PTTransformationLayout()
        for command in insertion_commands:
            transformation_layout.register(command)

        self._setup_to_module_id_translation_dict = setup_to_module_id_translation_dict
        all_quantizations = {}
        all_quantizations.update({k: v.quantizer_module_ref for k, v in self._weight_quantizers.items()})
        all_quantizations.update({k: v.quantizer_module_ref for k, v in self._non_weight_quantizers.items()})
        self._groups_of_adjacent_quantizers.parse_from_quantizer_setup(all_quantizations, single_config_quantizer_setup,
                                                                       setup_to_module_id_translation_dict)

        # NOTE: Order of activations must be the same to correctly broadcast parameters (e.g. scales) in distributed
        # mode (see call of `_dist_broadcast_coalesced` in torch/nn/parallel/distributed.py for more details)
        # pylint: disable=protected-access
        target_model.sort_compression_modules(ExtraCompressionModuleType.EXTERNAL_QUANTIZER)

        if self._debug_interface is not None:
            target_model.debug_interface.add_interface(self._debug_interface)

        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        all_quantizations = get_state_dict_names_with_modules(target_model, quantization_types)
        target_model._load_listener = LoadStateListener(target_model, all_quantizations)

        return transformation_layout

    @staticmethod
    def get_statistics_for_quantizer_setup(target_model: NNCFNetwork,
                                           quantizer_setup: QuantizerSetupBase,
                                           range_init_params: PTRangeInitParams) \
        -> Dict[PTTargetPoint, Dict[ReductionShape, TensorStatistic]]:
        if range_init_params is None:
            return {}
        observation_points_vs_collectors_dict = StatCollectorGenerator. \
            generate_collectors_for_range_init_statistics_collection(target_model.get_original_graph(),
                                                                     quantizer_setup,
                                                                     range_init_params)

        with target_model.temporary_clean_view() as intermediate_model:
            stat_builder = TensorStatisticsCollectionBuilder(NNCFConfig(),
                                                             observation_points_vs_collectors_dict)
            stat_builder.apply_to(intermediate_model)
            stat_ctrl = stat_builder.build_controller(intermediate_model)
            runner = SimpleDataLoaderRunner(intermediate_model, range_init_params.device)
            runner.progressbar_description = 'Collecting tensor statistics'
            runner.run(range_init_params.init_range_data_loader,
                       range_init_params.get_max_num_init_steps())

        retval = {}
        for ip, collector in stat_ctrl.ip_vs_collector_dict.items():
            retval[ip] = collector.get_statistics()
        return retval

    def _get_statistics_for_final_range_init(self, target_model: NNCFNetwork,
                                             quantizer_setup: QuantizerSetupBase,
                                             range_init_params: PTRangeInitParams) \
            -> Dict[PTTargetPoint, Dict[ReductionShape, TensorStatistic]]:
        return self.get_statistics_for_quantizer_setup(target_model, quantizer_setup, range_init_params)

    def _get_quantizer_setup(self, target_model: NNCFNetwork) -> SingleConfigQuantizerSetup:
        setup_generator = PropagationBasedQuantizerSetupGenerator(self.config,
                                                                  target_model,
                                                                  self.hw_config,
                                                                  self._precision_init_type,
                                                                  self._precision_init_params,
                                                                  self._range_init_params,
                                                                  self._debug_interface)
        single_config_quantizer_setup = setup_generator.generate_setup()
        self._build_time_metric_infos = setup_generator.get_build_time_metric_infos()
        return single_config_quantizer_setup

    def build_controller(self, target_model: NNCFNetwork) -> PTCompressionAlgorithmController:
        return QuantizationController(target_model,
                                      self.config,
                                      self._debug_interface,
                                      self._weight_quantizers,
                                      self._non_weight_quantizers,
                                      self._groups_of_adjacent_quantizers,
                                      build_time_metric_info=self._build_time_metric_infos,
                                      build_time_range_init_params=self._range_init_params)

    def __create_quantize_module(self, quantizer_spec: PTQuantizerSpec):
        quantizer_cls = QUANTIZATION_MODULES.get(quantizer_spec.mode)
        return quantizer_cls(quantizer_spec)

    @staticmethod
    def _get_adjust_padding_args(
            target_model_graph: NNCFGraph,
            quantization_point: SingleConfigQuantizationPoint,
            activation_quantizer: BaseQuantizer,
            quantization_points: List[SingleConfigQuantizationPoint]) -> List[AdjustPaddingArgs]:
        result = []
        for op_node_name in quantization_point.directly_quantized_operator_node_names:
            weight_bitwidth = None
            for qp in quantization_points:
                is_weight = qp.is_weight_quantization_point()
                if is_weight and (qp.insertion_point.target_node_name == op_node_name):
                    weight_bitwidth = qp.qconfig.num_bits
                    break
            if weight_bitwidth:
                is_applicable = False
                target_node = target_model_graph.get_node_by_name(op_node_name)
                if target_node.metatype in [Conv2dMetatype, DepthwiseConv2dSubtype]:
                    layer_attrs = target_node.layer_attributes  # type: ConvolutionLayerAttributes
                    padding_values = set(layer_attrs.padding_values)
                    padding_enabled = len(padding_values) >= 1 and padding_values.pop()
                    if padding_enabled:
                        symmetric = isinstance(activation_quantizer, SymmetricQuantizer)
                        per_tensor = not activation_quantizer.per_channel
                        a_int4 = activation_quantizer.num_bits == 4
                        w_int24 = weight_bitwidth <= 4
                        unsigned = not activation_quantizer.signed
                        is_applicable = symmetric and per_tensor and a_int4 and w_int24 and unsigned
                if is_applicable:
                    result.append(AdjustPaddingArgs(weight_bitwidth, activation_quantizer, op_node_name))
        return result

    def _add_adjust_padding_ops(self, adjust_padding_args: List[AdjustPaddingArgs]):
        commands = []
        for args in adjust_padding_args:
            ap = CalculatePaddingAdjustment(args.activation_quantizer)
            op = UpdatePaddingValue(ap).to(self._device_for_callable_obj_creation)
            insertion_point = PTTargetPoint(target_type=TargetType.PRE_LAYER_OPERATION,
                                            target_node_name=args.module_op_node_name)
            nncf_logger.warning('Padding will be adjusted for {}'.format(args.module_op_node_name))
            commands.append(PTInsertionCommand(insertion_point, op, TransformationPriority.DEFAULT_PRIORITY))
        return commands

    class ExternalQuantizerCallHook:
        """
        Cannot simply register the quantizer module as a callable hook, since we need to call
        a thread-local version of the quantizer module during base module execution.
        """

        def __init__(self, context: TracingContext, quantizer_storage_key: str,
                     debug_interface: 'QuantizationDebugInterface' = None):
            self.compressed_context = context
            self.quantizer_storage_key = quantizer_storage_key
            self.debug_interface = debug_interface

        def __call__(self, *args, **kwargs):
            if self.debug_interface is not None:
                self.debug_interface.register_activation_quantize_call(str(self.quantizer_storage_key))
            replica = self.compressed_context.base_module_thread_local_replica
            storage = getattr(replica, EXTERNAL_QUANTIZERS_STORAGE_NAME)
            return storage[self.quantizer_storage_key](*args, **kwargs)

    def _build_insertion_commands_list_for_quantizer_setup(self,
                                                           quantizer_setup: SingleConfigQuantizerSetup,
                                                           target_model: NNCFNetwork,
                                                           minmax_values_for_range_init: Dict[
                                                               QuantizationPointId, MinMaxTensorStatistic]) -> \
            Tuple[List[PTInsertionCommand], Dict[QuantizationPointId, QuantizerId]]:
        insertion_commands = []
        qp_id_vs_quant_module_id_dict = {}  # type: Dict[QuantizationPointId, QuantizerId]
        target_model_graph = target_model.get_original_graph()
        non_unified_scales_quantization_point_ids = set(quantizer_setup.quantization_points.keys())

        for unified_scales_group in quantizer_setup.unified_scale_groups.values():
            for us_qp_id in unified_scales_group:
                non_unified_scales_quantization_point_ids.discard(us_qp_id)

            quant_module_id, commands = self._build_commands_for_single_unified_scale_group(
                target_model,
                quantizer_setup,
                unified_scales_group,
                minmax_values_for_range_init)
            for us_qp_id in unified_scales_group:
                qp_id_vs_quant_module_id_dict[us_qp_id] = quant_module_id
            insertion_commands += commands

        for qp_id in non_unified_scales_quantization_point_ids:
            qp = quantizer_setup.quantization_points[qp_id]
            ip = qp.insertion_point
            qconfig = quantizer_setup.quantization_points[qp_id].qconfig

            range_init_minmax_values = None
            if minmax_values_for_range_init:
                minmax_stat = minmax_values_for_range_init[qp_id] if qp_id in minmax_values_for_range_init else None
                if minmax_stat is None:
                    nncf_logger.warning("Tensor statistics for location {} were not collected! The corresponding "
                                        "quantizer range will not be initialized!".format(ip))
                else:
                    range_init_minmax_values = (minmax_stat.min_values, minmax_stat.max_values)

            quantizer_module_id, commands = self._quantize_at_points_by_single_module(target_model,
                                                                                      [ip, ],
                                                                                      qconfig,
                                                                                      range_init_minmax_values)

            qp_id_vs_quant_module_id_dict[qp_id] = quantizer_module_id
            insertion_commands += commands

        adjust_padding_args = self._collect_adjust_padding_args(non_unified_scales_quantization_point_ids,
                                                                qp_id_vs_quant_module_id_dict, quantizer_setup,
                                                                target_model_graph)

        commands = self._add_adjust_padding_ops(adjust_padding_args)
        if commands:
            insertion_commands += commands

        return insertion_commands, qp_id_vs_quant_module_id_dict

    def _collect_adjust_padding_args(self,
                                     non_unified_scales_quantization_point_ids: Set[QuantizationPointId],
                                     qp_id_vs_quant_module_id_dict: Dict[QuantizationPointId, QuantizerId],
                                     quantizer_setup: SingleConfigQuantizerSetup,
                                     target_model_graph: NNCFGraph) -> List[AdjustPaddingArgs]:
        def weight_qp_filter_fn(qp_id_):
            qp_ = quantizer_setup.quantization_points[qp_id_]
            return qp_.is_weight_quantization_point()

        weight_qps = list(filter(weight_qp_filter_fn, non_unified_scales_quantization_point_ids))
        adjust_padding_args = []
        adjust_padding_operation_set = set()
        if self.hw_config is not None:
            adjust_padding_operation_set = self.hw_config.get_operations_with_adjusted_paddings()
        for wqp_id in weight_qps:
            wqp = quantizer_setup.quantization_points[wqp_id]
            ip = wqp.insertion_point
            target_node = target_model_graph.get_node_by_name(ip.target_node_name)

            op_type = PTOperatorMetatypeNodeMatcher.match(target_node)
            is_adjust_padding_applicable = op_type in adjust_padding_operation_set
            if self._should_setup_adjust_pad_ops and is_adjust_padding_applicable:
                gid = quantizer_setup.get_shared_inputs_group_id(wqp_id)
                shared_input_group = quantizer_setup.shared_input_operation_set_groups[gid]

                def is_qp_quantizing_same_op_as_wqp(qp_id_):
                    qp_ = quantizer_setup.quantization_points[qp_id_]
                    node_matched = target_node.node_name in qp_.directly_quantized_operator_node_names
                    return qp_.is_activation_quantization_point() and node_matched

                for qp_id in filter(is_qp_quantizing_same_op_as_wqp, shared_input_group):
                    quantizer_module_id = qp_id_vs_quant_module_id_dict[qp_id]
                    activation_quantizer = self._non_weight_quantizers[quantizer_module_id].quantizer_module_ref
                    args = self._get_adjust_padding_args(target_model_graph,
                                                         wqp, activation_quantizer,
                                                         list(quantizer_setup.quantization_points.values()))
                    if args:
                        adjust_padding_args.extend(args)
        return adjust_padding_args

    def _build_commands_for_single_unified_scale_group(self,
                                                       target_model: NNCFNetwork,
                                                       quantizer_setup: SingleConfigQuantizerSetup,
                                                       unified_scales_group: Set[QuantizationPointId],
                                                       minmax_values_for_range_init: Dict[QuantizationPointId,
                                                                                          MinMaxTensorStatistic]) -> \
            Tuple[QuantizerId, List[PTInsertionCommand]]:
        qp_ids_list_for_current_group = list(unified_scales_group)

        # The primary insertion point (to be associated with the actual quantizer module, not just hooks to it)
        # will be determined based on the string representation of said insertion point, to avoid random selection.
        # Weight insertion points are given priority.
        weight_qp_ids = [qp_id for qp_id in qp_ids_list_for_current_group
            if quantizer_setup.quantization_points[qp_id].is_weight_quantization_point()]
        act_qp_ids = [qp_id for qp_id in qp_ids_list_for_current_group
            if quantizer_setup.quantization_points[qp_id].is_activation_quantization_point()]
        ip_str_repr_key_lambda = lambda x: str(quantizer_setup.quantization_points[x].insertion_point)
        sorted_wqp_ids = sorted(weight_qp_ids, key=ip_str_repr_key_lambda)
        sorted_aqp_ids = sorted(act_qp_ids, key=ip_str_repr_key_lambda)
        sorted_qp_ids = sorted_wqp_ids + sorted_aqp_ids

        primary_qp_id = sorted_qp_ids[0]
        linked_qp_ids = sorted_qp_ids[1:]
        qconfig = quantizer_setup.quantization_points[primary_qp_id].qconfig
        linked_qconfigs = [quantizer_setup.quantization_points[qp_id].qconfig for qp_id in linked_qp_ids]
        for linked_qconfig in linked_qconfigs:
            if not qconfig.compatible_with_a_unified_scale_linked_qconfig(linked_qconfig):
                raise RuntimeError("The quantizer configurations for unified scale quantization points should"
                                   "be identical!")

        range_init_minmax_values = None
        if minmax_values_for_range_init:
            # Hopefully this will suffice.
            # TODO: gather unified statistic by linking stat collectors_and_modules_to_init instead
            min_values = None
            max_values = None
            for qp_id in sorted_qp_ids:
                minmax_stat = minmax_values_for_range_init[qp_id] if qp_id in minmax_values_for_range_init else None
                if minmax_stat is None:
                    nncf_logger.warning("Tensor statistics for location {} were not collected! The corresponding "
                                        "quantizer range will not be initialized!".format(
                        quantizer_setup.quantization_points[qp_id].insertion_point))
                    continue

                if min_values is None:
                    min_values = minmax_stat.min_values
                else:
                    min_values = torch.min(min_values, minmax_stat.min_values)

                if max_values is None:
                    max_values = minmax_stat.max_values
                else:
                    max_values = torch.max(max_values, minmax_stat.max_values)
            if min_values is not None and max_values is not None:
                range_init_minmax_values = min_values, max_values

        insertion_points = [quantizer_setup.quantization_points[qp_id].insertion_point for qp_id in sorted_qp_ids]
        quantizer_module_id, commands = self._quantize_at_points_by_single_module(target_model,
                                                                                  insertion_points,
                                                                                  qconfig,
                                                                                  range_init_minmax_values)
        return quantizer_module_id, commands

    def _select_final_qconfig(self, quantizer_config_list: List[QuantizerConfig]) -> QuantizerConfig:
        # Quantizer config list entries should arrive in the same order as they are listed
        # in the HW config, where they are sorted by descending order of priority
        return quantizer_config_list[0]

    def _quantize_at_points_by_single_module(self, target_model: NNCFNetwork,
                                             insertion_points: List[PTTargetPoint],
                                             qconfig: QuantizerConfig,
                                             range_init_minmax_values: Tuple[torch.Tensor, torch.Tensor] = None) -> \
            Tuple[QuantizerId, List[PTInsertionCommand]]:
        """
        Will generate insertion commands for quantization at possibly multiple points
        in the network using one and the same trainable quantizer module. The trainable
        quantizer module will be saved either inside the weightable module which weights
        it quantizes (for single-point weight quantization), or in a NNCFNetwork wrapper
        module (i.e. in a storage external to the original module).
        :param: target_model - the model to be quantized.
        :param: insertion_points - a list of target points for quantization using one
        quantizer module
        :param: qconfig - the QuantizerConfig for the resulting quantizer module
        :param: range_init_minmax_values - a pair of minimum and maximum values of input statistics
        for initializing the quantizer's trainable parameters
        :return: A tuple with the identifier of the new quantizer module and a list of
        insertion commands registering this module for quantization at spots described by
        insertion_points.
        """
        #pylint:disable=too-many-branches
        #pylint:disable=too-many-statements

        target_model_graph = target_model.get_original_graph()
        if not insertion_points:
            raise RuntimeError("No insertion points to put quantizers into!")

        def is_weights(ip: PTTargetPoint) -> bool:
            return ip.target_type is TargetType.OPERATION_WITH_WEIGHTS

        # The scale shapes for all insertion points must match, otherwise it is impossible to quantize them all
        # using a single module
        scale_shapes = []  # type: List[List[int]]
        for ip in insertion_points:
            if is_weights(ip):
                target_node = target_model_graph.get_node_by_name(ip.target_node_name)
                layer_attributes = target_node.layer_attributes
                scale_shape = get_scale_shape(layer_attributes.get_weight_shape(), is_weights=True,
                                              per_channel=qconfig.per_channel)
                scale_shapes.append(scale_shape)
            else:
                input_shape = target_model_graph.get_input_shape_for_insertion_point(ip)
                scale_shapes.append(get_scale_shape(list(input_shape),
                                                    is_weights=False, per_channel=qconfig.per_channel))
        if not all([shape == scale_shapes[0] for shape in scale_shapes]):
            raise RuntimeError("Scale shapes for the insertion points do not match!")
        scale_shape = scale_shapes[0]

        primary_ip = insertion_points[0]
        if is_weights(primary_ip):
            use_logarithm_scale = self._use_logarithm_scale_per_group[QuantizerGroup.WEIGHTS]
            narrow_range = True
        else:
            use_logarithm_scale = self._use_logarithm_scale_per_group[QuantizerGroup.ACTIVATIONS]
            narrow_range = False

        compression_lr_multiplier = self.config.get('compression_lr_multiplier', None)

        half_range = False
        if self.hw_config and not self._disable_saturation_fix and is_weights(primary_ip):
            if self.hw_config.target_device in ['CPU', 'ANY'] and qconfig.num_bits == 8:
                nncf_logger.warning('A saturation issue fix will be applied. '
                                    'Now all weight quantizers will effectively use only 7 bits out of 8 bits. '
                                    'This resolves the saturation issue problem on AVX2 and AVX-512 machines. '
                                    'Please take a look at the documentation for a detailed information.')
                half_range = True

        qspec = PTQuantizerSpec.from_config(qconfig,
                                            narrow_range=narrow_range,
                                            scale_shape=tuple(scale_shape),
                                            logarithm_scale=use_logarithm_scale,
                                            half_range=half_range,
                                            compression_lr_multiplier=compression_lr_multiplier)
        quantizer = self.__create_quantize_module(qspec).to(self._device_for_callable_obj_creation)
        if range_init_minmax_values is not None:
            quantizer.apply_minmax_init(min_values=range_init_minmax_values[0],
                                        max_values=range_init_minmax_values[1],
                                        log_module_name=str(primary_ip))

        qids = []  # type: List[QuantizerId]
        for ip in insertion_points:
            if is_weights(ip):
                qids.append(WeightQuantizerId(ip.target_node_name))
            else:
                qids.append(NonWeightQuantizerId(ip.target_node_name, ip.input_port_id))

        serialized_insertions_list = [str(x) for x in qids]
        external_quantizer_storage_key = ";".join(serialized_insertions_list)
        if len(insertion_points) > 1:
            nncf_logger.info(
                "Processing linked quantizer group:\n {}\n".format("\n".join(serialized_insertions_list)))

        if is_weights(primary_ip):
            primary_qid = WeightQuantizerId(primary_ip.target_node_name)
            self._weight_quantizers[primary_qid] = WeightQuantizerInfo(quantizer,
                                                                       target_model.get_containing_module(
                                                                           primary_ip.target_node_name
                                                                       ),
                                                                       insertion_points)
        else:
            primary_qid = NonWeightQuantizerId(primary_ip.target_node_name, primary_ip.input_port_id)
            self._non_weight_quantizers[primary_qid] = \
                NonWeightQuantizerInfo(quantizer, insertion_points)

        if not (is_weights(primary_ip) and len(insertion_points) == 1):
            assert external_quantizer_storage_key not in target_model.get_compression_modules_by_type(
                ExtraCompressionModuleType.EXTERNAL_QUANTIZER)

            target_model.add_compression_module(external_quantizer_storage_key, quantizer,
                                                ExtraCompressionModuleType.EXTERNAL_QUANTIZER)

        insertion_commands = []
        for curr_insertion_point in insertion_points:
            if curr_insertion_point in self._processed_insertion_points:
                raise RuntimeError(
                    "Insertion point {} already quantized!".format(str(curr_insertion_point))
                )
            self._processed_insertion_points.add(curr_insertion_point)

            if is_weights(curr_insertion_point):
                nncf_logger.info("Performing {}{} weight quantization for: {}".format(
                    "signed" if quantizer.signed else "unsigned",
                    " logarithm_scale" if quantizer.is_using_log_scale_storage else "",
                    str(curr_insertion_point)))
                if len(insertion_points) == 1:
                    # For backward compatibility, if only one weight is quantized by a single quantizer,
                    # insert UpdateWeight ops with a genuine quantizer module
                    callable_obj = quantizer
                else:
                    # Otherwise use external quantizer module storage since the quantization points will have to
                    # share the single module and this would be impossible for multiple weight quantizer sharing if
                    # the corresponding UpdateWeights operations contained real modules (these would simply get copied
                    # by PyTorch internals)
                    callable_obj = self.ExternalQuantizerCallHook(target_model.get_tracing_context(),
                                                                  external_quantizer_storage_key,
                                                                  self._debug_interface)
            else:
                nncf_logger.info("Performing {}{} activation quantization for: {}".format(
                    "signed" if quantizer.signed else "unsigned",
                    " logarithm_scale" if quantizer.is_using_log_scale_storage else "",
                    str(curr_insertion_point)
                ))
                # Hooks will be identical for each affected op_address in the linked scenario
                # - will call one and the same quantizer
                callable_obj = self.ExternalQuantizerCallHook(target_model.get_tracing_context(),
                                                              external_quantizer_storage_key,
                                                              self._debug_interface)

            insertion_commands.append(PTInsertionCommand(curr_insertion_point,
                                                         callable_obj,
                                                         TransformationPriority.QUANTIZATION_PRIORITY))
        return primary_qid, insertion_commands

    def _are_frozen_layers_allowed(self) -> Tuple[bool, str]:
        message_template = Template('Frozen layers are$denial allowed for $algo_prefix quantization')
        bits = set()
        bits.update({wq.quantizer_module_ref.num_bits for wq in self._weight_quantizers.values()})
        bits.update({nwq.quantizer_module_ref.num_bits for nwq in self._non_weight_quantizers.values()})

        if self._precision_init_params or len(bits) > 1:
            return False, message_template.substitute(denial=' not', algo_prefix='mixed precision')

        if len(bits) == 1:
            bitwidth = bits.pop()
            algo_prefix = f'INT{bitwidth}'
            if bitwidth == 8:
                return True, message_template.substitute(denial='', algo_prefix=algo_prefix)
            return False, message_template.substitute(denial=' not', algo_prefix=algo_prefix)
        return True, message_template.substitute(denial='', algo_prefix='empty')

    def initialize(self, model: NNCFNetwork) -> None:
        if is_main_process() and self.should_init:
            bn_adaptation = BatchnormAdaptationAlgorithm(
                **extract_bn_adaptation_init_params(self.config))
            bn_adaptation.run(model)


class QuantizationControllerBase(PTCompressionAlgorithmController):
    def enable_activation_quantization(self):
        raise NotImplementedError

    def enable_weight_quantization(self):
        raise NotImplementedError

    def disable_activation_quantization(self):
        raise NotImplementedError

    def disable_weight_quantization(self):
        raise NotImplementedError

    def init_range(self):
        raise NotImplementedError


class QuantizationController(QuantizationControllerBase):
    def __init__(self, target_model: NNCFNetwork,
                 quantization_config: 'NNCFConfig',
                 debug_interface: 'QuantizationDebugInterface',
                 weight_quantizers: Dict[WeightQuantizerId, WeightQuantizerInfo],
                 non_weight_quantizers: Dict[NonWeightQuantizerId, NonWeightQuantizerInfo],
                 groups_of_adjacent_quantizers: GroupsOfAdjacentQuantizers,
                 collect_compression_metrics: bool = True,
                 build_time_metric_info: QuantizationShareBuildTimeInfo = None,
                 build_time_range_init_params: PTRangeInitParams = None):
        super().__init__(target_model)
        self._loss = ZeroCompressionLoss(next(target_model.parameters()).device)
        self._scheduler = BaseCompressionScheduler()
        self.debug_interface = debug_interface
        self.quantization_config = quantization_config
        self._collect_compression_metrics = collect_compression_metrics
        self._build_time_range_init_params = build_time_range_init_params

        self.weight_quantizers = weight_quantizers  # type: Dict[WeightQuantizerId, WeightQuantizerInfo]
        self.non_weight_quantizers = non_weight_quantizers  # type: Dict[NonWeightQuantizerId, NonWeightQuantizerInfo]
        self.all_quantizations = OrderedDict()  # type: Dict[QuantizerId, BaseQuantizer]
        self.all_quantizations.update({k: v.quantizer_module_ref for k, v in self.weight_quantizers.items()})
        self.all_quantizations.update({k: v.quantizer_module_ref for k, v in self.non_weight_quantizers.items()})
        self._distributed = False
        self._groups_of_adjacent_quantizers = groups_of_adjacent_quantizers
        self._bn_adaptation = None

        should_export_to_onnx_qdq = quantization_config.get("export_to_onnx_standard_ops",
                                                            False)
        if should_export_to_onnx_qdq:
            export_mode = QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS
        else:
            export_mode = QuantizerExportMode.FAKE_QUANTIZE

        for quantizer in self.all_quantizations.values():  # type: BaseQuantizer
            quantizer.set_export_mode(export_mode)

        if self._collect_compression_metrics:
            self.metric_store = {}
            # These metrics are collected here and are updated when the method .statistics() is called
            self.non_stable_metric_collectors = [
                QuantizationShareStatisticsCollector(
                    self.weight_quantizers, self.non_weight_quantizers, build_time_metric_info
                ),
                BitwidthDistributionStatisticsCollector(
                    self.weight_quantizers, self.non_weight_quantizers
                ),
                MemoryConsumptionStatisticsCollector(
                    target_model, self.weight_quantizers, self.non_weight_quantizers
                ),
            ]
            # These metrics are collected once here and are not updated when the method .statistics() is called
            self.stable_metric_collectors = [ShareEdgesQuantizedDataPathStatisticsCollector(target_model, self)]
            self.update_metric_store(True)

        params = quantization_config.get('params', None)
        self.is_staged_scheduler = bool(params)


        # Staged scheduler must be created after initialized to prevent extra logic with disabled quantizations
        if self.is_staged_scheduler:
            scheduler_cls = QUANTIZATION_SCHEDULERS.get("staged")
            self._scheduler = scheduler_cls(self, params)

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    @property
    def groups_of_adjacent_quantizers(self) -> GroupsOfAdjacentQuantizers:
        return self._groups_of_adjacent_quantizers

    def prepare_for_export(self):
        for quantizer_id, quantizer in self.all_quantizations.items():
            if not quantizer.is_enabled_quantization():
                nncf_logger.warning('Disabled quantization on export to ONNX: {}'.format(quantizer_id))

    def update_metric_store(self, do_all: bool = False):
        for collector in self.non_stable_metric_collectors:
            self.metric_store[collector.NAME_STR] = collector.collect()
        if do_all:
            for collector in self.stable_metric_collectors:
                self.metric_store[collector.NAME_STR] = collector.collect()

    def distributed(self):
        self._distributed = True
        self._broadcast_initialized_params_for_each_quantizer()

    def _broadcast_initialized_params_for_each_quantizer(self):
        # NOTE: Order of quantization modules must be the same on GPUs to correctly broadcast num_bits
        sorted_quantizers = OrderedDict(sorted(self.all_quantizations.items(), key=lambda x: str(x[0])))
        for quantizer in sorted_quantizers.values():  # type: BaseQuantizer
            quantizer.broadcast_initialized_params()

    def _do_runtime_range_init(self, range_init_params: PTRangeInitParams):
        modules_to_init = OrderedDict()
        for wq_id, wq_info in self.weight_quantizers.items():
            group = QuantizerGroup.WEIGHTS
            init_config = range_init_params.get_init_config_for_scope_and_group(wq_id, group)
            modules_to_init[str(wq_id)] = (wq_info.quantizer_module_ref, init_config)

        for aq_id, aq_info in self.non_weight_quantizers.items():
            group = QuantizerGroup.ACTIVATIONS
            init_config = range_init_params.get_init_config_for_scope_and_group(aq_id, group)
            modules_to_init[str(aq_id)] = (aq_info.quantizer_module_ref, init_config)

        # NOTE: Order of modules must be the same to correctly broadcast parameters (e.g. input_low
        # and input_range)
        modules_to_init = OrderedDict(sorted(modules_to_init.items()))
        self.modules_to_range_init = modules_to_init
        runner = DataLoaderRangeInitializeRunner(self._model, modules_to_init, range_init_params.device)

        quantizers = [module for module, config in modules_to_init.values()]
        quantizers_switcher = QuantizersSwitcher(quantizers)
        # bypass quantization to collect statistics from floating point model
        quantizers_switcher.disable_quantizers()
        runner.run(range_init_params.init_range_data_loader,
                   range_init_params.get_max_num_init_steps())
        quantizers_switcher.enable_quantizers()

        self._model.rebuild_graph()

    def compression_stage(self) -> CompressionStage:
        if self.is_staged_scheduler:
            return self.scheduler.compression_stage()
        return CompressionStage.FULLY_COMPRESSED

    def init_precision(self,
                       precision_init_type: str,
                       precision_init_params: BasePrecisionInitParams,
                       precision_constraints: HardwareQuantizationConstraints) -> SingleConfigQuantizerSetup:
        """
        Precision initialization happens based on an measure of layer sensitivity to perturbations. The measure is
        calculated by average Hessian trace estimation for each layer using Hutchinson algorithm.
        """
        init_impl = PrecisionInitializerFactory.create(precision_init_type)
        initializer = init_impl(self, precision_init_params, precision_constraints)
        nncf_logger.info("Initialization of quantization precisions")
        return initializer.apply_init()

    def init_range(self, range_init_params: PTRangeInitParams = None):
        """
        Tracks input statistics for quantizers in the model and sets ranges of the quantizers to correspond to
        minimum and maximum input tensor levels observed.
        :param range_init_params: specifies parameters for this range initialization call; if None, the parameters
        that were used during compressed model creation will be used.
        """
        if range_init_params is None:
            if self._build_time_range_init_params is None:
                nncf_logger.warning("Requested a quantization controller to do range initialization without params, but"
                                    " the build time range initialization was not supplied with params as well - range "
                                    "initialization will not be done")
                return
            range_init_params = self._build_time_range_init_params

        self._do_runtime_range_init(range_init_params)

        if self._distributed:
            self._broadcast_initialized_params_for_each_quantizer()

    def enable_activation_quantization(self):
        for m in self.non_weight_quantizers.values():
            m.quantizer_module_ref.enable_quantization()

    def enable_weight_quantization(self):
        for m in self.weight_quantizers.values():
            m.quantizer_module_ref.enable_quantization()

    def disable_activation_quantization(self):
        for m in self.non_weight_quantizers.values():
            m.quantizer_module_ref.disable_quantization()

    def disable_weight_quantization(self):
        for m in self.weight_quantizers.values():
            m.quantizer_module_ref.disable_quantization()

    def statistics(self, quickly_collected_only=False) -> NNCFStatistics:
        num_enabled_quantization = len([1 for q in self.all_quantizations.values() if q.is_enabled_quantization()])
        multiplier = 100 / len(self.all_quantizations)
        ratio_of_enabled_quantizations = num_enabled_quantization * multiplier

        stats = QuantizationStatistics(ratio_of_enabled_quantizations)
        if self._collect_compression_metrics and not quickly_collected_only:
            self.update_metric_store()
            stats = QuantizationStatistics(ratio_of_enabled_quantizations, **self.metric_store)

        nncf_stats = NNCFStatistics()
        nncf_stats.register('quantization', stats)
        return nncf_stats


class QuantizationDebugInterface(DebugInterface):
    QUANTIZERS_IN_NNCF_MODULES_TRACKER_NAME = 'quantized_modules'
    ACTIVATION_QUANTIZERS_TRACKER_NAME = 'activation_quantizers'

    def __init__(self):
        self.call_trackers = {
            self.QUANTIZERS_IN_NNCF_MODULES_TRACKER_NAME: CallCountTracker(
                QuantizationDebugInterface.QUANTIZERS_IN_NNCF_MODULES_TRACKER_NAME),
            self.ACTIVATION_QUANTIZERS_TRACKER_NAME: CallCountTracker(
                QuantizationDebugInterface.ACTIVATION_QUANTIZERS_TRACKER_NAME),
        }
        self.graph_size = 0

        from nncf.torch.debug import DEBUG_LOG_DIR
        self.dump_dir = Path(DEBUG_LOG_DIR) / Path("debug_dumps")
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.scale_dump_dir = self.dump_dir / Path("scale")
        if self.scale_dump_dir.exists():
            shutil.rmtree(str(self.scale_dump_dir))
        self.scale_dump_dir.mkdir(parents=True, exist_ok=True)
        self.prop_graph_dump_dir = self.dump_dir / Path("quant_prop")
        if self.prop_graph_dump_dir.exists():
            shutil.rmtree(str(self.prop_graph_dump_dir))
        self.forward_call_count = 0
        self._strict_forward = False

    def init_actual(self, owner_model: NNCFNetwork):
        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        quantizers_in_nncf_modules = owner_model.get_modules_in_nncf_modules_by_type(quantization_types)
        nncf_module_quantizations_id_list = [str(scope) for scope in
                                             quantizers_in_nncf_modules.keys()]  # type: List[str]

        activation_quantizer_id_list = owner_model.get_compression_modules_by_type(
            ExtraCompressionModuleType.EXTERNAL_QUANTIZER).keys()  # type: List[str]
        self.call_trackers[self.QUANTIZERS_IN_NNCF_MODULES_TRACKER_NAME].init_with_key_list(
            nncf_module_quantizations_id_list)
        self.call_trackers[self.ACTIVATION_QUANTIZERS_TRACKER_NAME].init_with_key_list(
            activation_quantizer_id_list)
        self._strict_forward = True

    def pre_forward_actions(self, module: 'NNCFNetwork'):
        self.reset_counters()

    def post_forward_actions(self, module: 'NNCFNetwork'):
        self.register_forward_call()
        # pylint:disable=protected-access
        ctx = module.get_tracing_context()
        self.set_graph_size(ctx.graph.get_nodes_count())

        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        nncf_module_quantizations = module.get_modules_in_nncf_modules_by_type(
            quantization_types)  # type: Dict['Scope', nn.Module]

        for qm_scope, qm_module in nncf_module_quantizations.items():
            # Important - this will not work for DataParallel since it copies the
            # entire parent module for each thread and the `call_count` attributes
            # are incremented for thread local copies of `qm_module`, which are not
            # the same as the primary copies of `qm_module` iterated over at this point
            self.register_quantizer_module_call(str(qm_scope), qm_module.call_count)
            self.dump_scale(qm_module.get_trainable_params(), str(qm_scope))
            qm_module.reset_call_counter()
        self.print_call_stats()

        call_dict = ctx.get_node_call_counter_dict()
        total_calls = sum(call_dict.values())
        nncf_logger.debug("{} nodes called out of total {}".format(total_calls,
                                                                   ctx.graph.get_nodes_count()))
        if self._strict_forward:
            for tracker in self.call_trackers.values():
                if tracker.get_never_called_keys():
                    # This will always trigger for DataParallel - disregard or disable debug mode
                    # for DataParallel runs
                    raise RuntimeError("{} has never called modules: {}!".format(
                        tracker.name, tracker.get_never_called_keys()))

    def dump_scale(self, quantizer_scale_params: Dict[str, torch.Tensor], quantizer_name: str):
        import re
        quantizer_normalized_name = re.sub(r'[^\w\-_\. ]', '_', quantizer_name)
        for scale_param_name, scale_param in quantizer_scale_params.items():
            fname = "{}_{}.txt".format(quantizer_normalized_name, scale_param_name)
            with safe_open(self.scale_dump_dir / fname, "ab") as file:
                np.savetxt(file, scale_param.cpu().numpy().flatten())

    def reset_counters(self):
        for tracker in self.call_trackers.values():
            tracker.reset()

    def register_quantizer_module_call(self, key, counts=None):
        self.call_trackers[self.QUANTIZERS_IN_NNCF_MODULES_TRACKER_NAME].register_call(key, counts)

    def register_activation_quantize_call(self, key: str):
        self.call_trackers[self.ACTIVATION_QUANTIZERS_TRACKER_NAME].register_call(key)

    def print_call_stats(self):
        nncf_logger.debug(" Graph size: {} nodes".format(self.graph_size))
        for tracker in self.call_trackers.values():
            msg = " {} tracker:".format(tracker.name)
            msg += " {} total calls;".format(tracker.get_total_call_count())

            never_called = tracker.get_never_called_keys()
            if never_called:
                msg += " {} entries never called;".format(len(never_called))

            overcalled = tracker.get_overcalled_keys_with_call_counts()
            if overcalled:
                msg += " {} entries called more than once;".format(len(overcalled))
            nncf_logger.debug(msg)

    def set_graph_size(self, new_size):
        if new_size != self.graph_size:
            nncf_logger.debug('\n')
            nncf_logger.debug(
                " warning - graph size has changed from {} to {} since last forward".format(self.graph_size,
                                                                                            new_size))
        self.graph_size = new_size

    def register_forward_call(self):
        self.forward_call_count += 1

    def visualize_quantizer_propagation(self,
                                        prop_solver: QuantizerPropagationSolver,
                                        prop_graph: QuantizerPropagationStateGraph,
                                        iteration: str):
        self.prop_graph_dump_dir.mkdir(parents=True, exist_ok=True)
        fname = "quant_prop_iter_{}.dot".format(iteration)
        prop_solver.debug_visualize(prop_graph,
                                    self.prop_graph_dump_dir / Path(fname))

    def visualize_insertion_point_graph(self, insertion_point_graph: InsertionPointGraph):
        out_graph = nx.MultiDiGraph()
        for node_key, node in insertion_point_graph.nodes.items():
            if node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.INSERTION_POINT:
                insertion_point_data = node[
                    InsertionPointGraph.INSERTION_POINT_DATA_NODE_ATTR]  # type: PTTargetPoint
                label = "IP: {}".format(insertion_point_data.target_type)
                if insertion_point_data.input_port_id is not None:
                    label += " port " + str(insertion_point_data.input_port_id)
                out_graph.add_node(node_key, label=label, color="red")
            elif node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.OPERATOR:
                out_graph.add_node(node_key)
            else:
                raise RuntimeError("Invalid InsertionPointGraph node!")
        for u, v in insertion_point_graph.edges:
            out_graph.add_edge(u, v)

        for node_key, node in insertion_point_graph.nodes.items():
            if node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.OPERATOR:
                for ip_node_key in node[InsertionPointGraph.ASSOCIATED_IP_NODE_KEYS_NODE_ATTR]:
                    out_graph.add_edge(node_key, ip_node_key, style="dashed", headport='e', tailport='e')

        nx.drawing.nx_pydot.write_dot(out_graph, self.dump_dir / Path("insertion_point_graph.dot"))


class ExperimentalQuantizationBuilder(QuantizationBuilder):
    def __init__(self, quantizer_setup: MultiConfigQuantizerSetup,
                 initial_quantizer_setup: SingleConfigQuantizerSetup,
                 tensor_stats_for_all_setup_variations: Dict[PTTargetPoint, Dict[ReductionShape, TensorStatistic]],
                 hw_config: HWConfig = None):
        should_init = bool(tensor_stats_for_all_setup_variations)
        super().__init__(NNCFConfig(), should_init=should_init)
        self._initial_quantizer_setup = initial_quantizer_setup
        self._quantizer_setup = quantizer_setup
        self._tensor_stats = tensor_stats_for_all_setup_variations
        self._should_setup_adjust_pad_ops = False
        self.hw_config = hw_config

    def _handle_frozen_layers(self, target_model: NNCFNetwork):
        pass

    def _get_quantizer_setup(self, target_model: NNCFNetwork) -> SingleConfigQuantizerSetup:
        return self._initial_quantizer_setup

    def _get_statistics_for_final_range_init(self,
                                             target_model: NNCFNetwork,
                                             quantizer_setup: QuantizerSetupBase,
                                             range_init_params: PTRangeInitParams) -> Dict[
        PTTargetPoint, Dict[ReductionShape, TensorStatistic]]:
        return self._tensor_stats

    def build_controller(self, target_model: NNCFNetwork) -> 'ExperimentalQuantizationController':
        groups_of_adjacent_quantizers = GroupsOfAdjacentQuantizers()
        all_quantizations = {}  # type: Dict[QuantizerId, BaseQuantizer]
        all_quantizations.update({k: v.quantizer_module_ref for k, v in self._weight_quantizers.items()})
        all_quantizations.update({k: v.quantizer_module_ref for k, v in self._non_weight_quantizers.items()})

        groups_of_adjacent_quantizers.parse_from_quantizer_setup(all_quantizations,
                                                                 self._quantizer_setup,
                                                                 self._setup_to_module_id_translation_dict)

        build_time_metric_infos = QuantizationShareBuildTimeInfo(len(self._non_weight_quantizers),
                                                                 len(self._weight_quantizers))

        return ExperimentalQuantizationController(target_model,
                                                  self._weight_quantizers,
                                                  self._non_weight_quantizers,
                                                  groups_of_adjacent_quantizers,
                                                  self._quantizer_setup,
                                                  self._initial_quantizer_setup,
                                                  self._setup_to_module_id_translation_dict,
                                                  self._tensor_stats,
                                                  build_time_metric_infos,
                                                  self._should_setup_adjust_pad_ops,
                                                  self.hw_config)

    def initialize(self, model: NNCFNetwork) -> None:
        pass


class ExperimentalQuantizationController(QuantizationController):
    def __init__(self, target_model: NNCFNetwork,
                 weight_quantizers: Dict[WeightQuantizerId, WeightQuantizerInfo],
                 non_weight_quantizers: Dict[NonWeightQuantizerId, NonWeightQuantizerInfo],
                 groups_of_adjacent_quantizers: GroupsOfAdjacentQuantizers,
                 quantizer_setup: MultiConfigQuantizerSetup,
                 initial_quantizer_setup: SingleConfigQuantizerSetup,
                 setup_to_module_id_translation_dict: Dict[QuantizationPointId, QuantizerId],
                 tensor_stats: Dict[PTTargetPoint, Dict[ReductionShape, TensorStatistic]],
                 build_time_metric_info: QuantizationShareBuildTimeInfo,
                 should_setup_adjust_pad_ops=False,
                 hw_config: HWConfig = None):
        super().__init__(target_model,
                         NNCFConfig(),
                         debug_interface=None,
                         weight_quantizers=weight_quantizers,
                         non_weight_quantizers=non_weight_quantizers,
                         groups_of_adjacent_quantizers=groups_of_adjacent_quantizers,
                         collect_compression_metrics=True,
                         build_time_metric_info=build_time_metric_info)
        self._target_model_ref = target_model
        self._should_setup_adjust_pad_ops = should_setup_adjust_pad_ops
        self._quantizer_setup = quantizer_setup
        self._initial_quantizer_setup = initial_quantizer_setup
        self._tensor_stats = tensor_stats
        self.setup_to_module_id_translation_dict = setup_to_module_id_translation_dict
        self.module_id_to_qp_id_translation_dict = {}  # type: Dict[QuantizerId, Set[QuantizationPointId]]
        for qp_id, qid in self.setup_to_module_id_translation_dict.items():
            if qid in self.module_id_to_qp_id_translation_dict:
                self.module_id_to_qp_id_translation_dict[qid].add(qp_id)
            else:
                self.module_id_to_qp_id_translation_dict[qid] = {qp_id}
        self.hw_config = hw_config

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    def get_quantizer_setup_for_current_state(self) -> SingleConfigQuantizerSetup:
        qpid_vs_selected_qconfig = {}
        for qp_id in self._initial_quantizer_setup.quantization_points:
            quant_module_id = self.setup_to_module_id_translation_dict[qp_id]
            quant_module = self.all_quantizations[quant_module_id]
            qconfig = quant_module.get_quantizer_config()
            qpid_vs_selected_qconfig[qp_id] = qconfig
        return self._quantizer_setup.select_qconfigs(qpid_vs_selected_qconfig, strict=False)

    def is_new_setup_requires_regeneration(self, quantizer_setup: SingleConfigQuantizerSetup) -> bool:
        current_setup = self.get_quantizer_setup_for_current_state()
        if Counter(current_setup.quantization_points.keys()) != Counter(quantizer_setup.quantization_points.keys()):
            raise ValueError("The new setup is inconsistent with the original parameter space!")
        for qp_id, qp in quantizer_setup.quantization_points.items():
            current_qconfig = current_setup.quantization_points[qp_id].qconfig
            new_qconfig = quantizer_setup.quantization_points[qp_id].qconfig
            new_padding_adjust_applicable = CalculatePaddingAdjustment.is_config_applicable(new_qconfig)
            current_padding_adjust_applicable = CalculatePaddingAdjustment.is_config_applicable(current_qconfig)
            need_padding_regeneration = \
                self._should_setup_adjust_pad_ops and \
                qp.is_activation_quantization_point() and \
                new_padding_adjust_applicable != current_padding_adjust_applicable
            if current_qconfig.per_channel != new_qconfig.per_channel or \
                    (new_qconfig.signedness_to_force is not None and
                     current_qconfig.signedness_to_force != new_qconfig.signedness_to_force) or \
                    current_qconfig.mode != new_qconfig.mode or \
                    need_padding_regeneration:
                return True
        return False

    def apply_new_quantizer_setup(self, quantizer_setup: SingleConfigQuantizerSetup) -> Tuple[
            'ExperimentalQuantizationController', NNCFNetwork]:
        if not self.is_new_setup_requires_regeneration(quantizer_setup):
            for qp_id, qp in quantizer_setup.quantization_points.items():
                quant_module_id = self.setup_to_module_id_translation_dict[qp_id]
                quant_module = self.all_quantizations[quant_module_id]
                quant_module.num_bits = qp.qconfig.num_bits
            return self, self._target_model_ref
        new_model = self._target_model_ref.get_clean_shallow_copy()
        new_builder = ExperimentalQuantizationBuilder(self._quantizer_setup,
                                                      initial_quantizer_setup=quantizer_setup,
                                                      tensor_stats_for_all_setup_variations=self._tensor_stats,
                                                      hw_config=self.hw_config)
        new_builder.apply_to(new_model)
        new_ctrl = new_builder.build_controller(new_model)
        return new_ctrl, new_model
