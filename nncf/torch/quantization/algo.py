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

"""
Contains builder and controller class definitions for the quantization algorithm.
"""

from collections import Counter
from collections import OrderedDict
from copy import deepcopy
from enum import IntEnum
from string import Template
from typing import Any, Optional

import torch

import nncf
from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.api.compression import CompressionStage
from nncf.common.deprecation import warning_deprecated
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.definitions import MODEL_INPUT_OP_NAME
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import WeightedLayerAttributes
from nncf.common.graph.patterns.manager import PatternsManager
from nncf.common.graph.patterns.manager import TargetDevice
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.utils import get_first_nodes_of_type
from nncf.common.graph.utils import get_target_dim_for_compression_legacy
from nncf.common.graph.utils import get_weight_shape_legacy
from nncf.common.hardware.config import HWConfig
from nncf.common.hardware.config import HWConfigType
from nncf.common.hardware.config import get_hw_config_type
from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.common.logging import nncf_logger
from nncf.common.quantization.config_assignment import assign_qconfig_lists_to_modules
from nncf.common.quantization.quantizer_propagation.structs import IgnoreReason
from nncf.common.quantization.quantizer_setup import DEFAULT_QUANTIZER_CONFIG
from nncf.common.quantization.quantizer_setup import MultiConfigQuantizerSetup
from nncf.common.quantization.quantizer_setup import QuantizationPointId
from nncf.common.quantization.quantizer_setup import QuantizerSetupBase
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.quantization.structs import NonWeightQuantizerId
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.structs import QuantizationConstraints
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.quantization.structs import QuantizerId
from nncf.common.quantization.structs import WeightQuantizerId
from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.statistics import NNCFStatistics
from nncf.common.tensor_statistics.collectors import ReductionAxes
from nncf.common.utils.api_marker import api
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import copy_model
from nncf.common.utils.debug import is_debug
from nncf.config import NNCFConfig
from nncf.config.extractors import extract_algo_specific_config
from nncf.config.extractors import extract_bn_adaptation_init_params
from nncf.config.extractors import extract_range_init_params
from nncf.config.schemata.algo.quantization import PRECISION_INIT_TYPES_VS_DESCRIPTION
from nncf.config.schemata.defaults import QUANTIZATION_EXPORT_TO_ONNX_STANDARD_OPS
from nncf.config.schemata.defaults import QUANTIZATION_LOGARITHM_SCALE
from nncf.config.schemata.defaults import QUANTIZATION_OVERFLOW_FIX
from nncf.config.schemata.defaults import QUANTIZATION_PRESET
from nncf.config.schemata.defaults import QUANTIZE_INPUTS
from nncf.config.schemata.defaults import QUANTIZE_OUTPUTS
from nncf.experimental.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.experimental.common.tensor_statistics.statistics import TensorStatistic
from nncf.parameters import StripFormat
from nncf.torch.algo_selector import PT_COMPRESSION_ALGORITHMS
from nncf.torch.algo_selector import ZeroCompressionLoss
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import ELEMENTWISE_OPERATIONS
from nncf.torch.graph.operator_metatypes import UNIFICATION_PRODUCING_METATYPES
from nncf.torch.graph.operator_metatypes import PTCatMetatype
from nncf.torch.graph.operator_metatypes import PTModuleConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTModuleDepthwiseConv2dSubtype
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.commands import TransformationPriority
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.hardware.config import PTHWConfig
from nncf.torch.initialization import SimpleDataLoaderRunner
from nncf.torch.module_operations import UpdatePaddingValue
from nncf.torch.nncf_network import LoadStateListener
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.adjust_padding import AdjustPaddingArgs
from nncf.torch.quantization.adjust_padding import CalculatePaddingAdjustment
from nncf.torch.quantization.base_ctrl import QuantizationControllerBase
from nncf.torch.quantization.debug_interface import QuantizationDebugInterface
from nncf.torch.quantization.default_quantization import DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT
from nncf.torch.quantization.default_quantization import QUANTIZATION_LAYER_METATYPES
from nncf.torch.quantization.external_quantizer import ExternalQuantizerCallHook
from nncf.torch.quantization.init_precision import PrecisionInitializerFactory
from nncf.torch.quantization.init_range import DataLoaderRangeInitializeRunner
from nncf.torch.quantization.init_range import PTRangeInitParams
from nncf.torch.quantization.init_range import StatCollectorGenerator
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import PTQuantizationPoint
from nncf.torch.quantization.layers import PTQuantizerSetup
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import QuantizerConfig
from nncf.torch.quantization.layers import QuantizerExportMode
from nncf.torch.quantization.layers import QuantizersSwitcher
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.quantization.layers import get_scale_shape
from nncf.torch.quantization.metrics import MemoryConsumptionStatisticsCollector
from nncf.torch.quantization.metrics import PTQuantizationStatisticsCollector
from nncf.torch.quantization.metrics import QuantizationShareBuildTimeInfo
from nncf.torch.quantization.metrics import ShareEdgesQuantizedDataPathStatisticsCollector
from nncf.torch.quantization.precision_constraints import HardwareQuantizationConstraints
from nncf.torch.quantization.precision_init.adjacent_quantizers import GroupsOfAdjacentQuantizers
from nncf.torch.quantization.precision_init.autoq_init import AutoQPrecisionInitParams
from nncf.torch.quantization.precision_init.base_init import BasePrecisionInitializer
from nncf.torch.quantization.precision_init.base_init import BasePrecisionInitParams
from nncf.torch.quantization.precision_init.hawq_init import HAWQPrecisionInitParams
from nncf.torch.quantization.precision_init.manual_init import ManualPrecisionInitParams
from nncf.torch.quantization.schedulers import QUANTIZATION_SCHEDULERS
from nncf.torch.quantization.strip import strip_quantized_model
from nncf.torch.quantization.structs import NonWeightQuantizerInfo
from nncf.torch.quantization.structs import WeightQuantizerInfo
from nncf.torch.quantization.translator import PTTargetPointTranslator
from nncf.torch.structures import AutoQPrecisionInitArgs
from nncf.torch.structures import QuantizationPrecisionInitArgs
from nncf.torch.tensor_statistics.algo import TensorStatisticsCollectionBuilder
from nncf.torch.tensor_statistics.statistics import pt_convert_stat_to_min_max_tensor_stat
from nncf.torch.utils import get_model_device
from nncf.torch.utils import get_model_dtype
from nncf.torch.utils import get_state_dict_names_with_modules
from nncf.torch.utils import is_main_process
from nncf.torch.utils import training_mode_switcher

QUANTIZER_BUILDER_STATE_VERSION_SAVE_NAME = "version"


class QuantizerBuilderStateVersion(IntEnum):
    # In Quantization builder state SingleConfigQuantizerSetup is being saved as quantizer setup.
    v1 = 1
    # In Quantization builder state PTQuantizerSetup is being saved as quantizer setup.
    v2 = 2

    @staticmethod
    def from_compression_state(compression_state):
        if QUANTIZER_BUILDER_STATE_VERSION_SAVE_NAME in compression_state:
            return compression_state.get(QUANTIZER_BUILDER_STATE_VERSION_SAVE_NAME)
        return QuantizerBuilderStateVersion.v1


class QuantizerSetupGeneratorBase:
    def __init__(
        self,
        quant_config: dict,
        target_model: NNCFNetwork,
        precision_init_type: str = None,
        precision_init_params: BasePrecisionInitParams = None,
        range_init_params: PTRangeInitParams = None,
        hw_config: HWConfig = None,
    ):
        self._target_model = target_model
        self._quantization_config = quant_config
        self.hw_config = hw_config
        self._target_device = None if hw_config is None else hw_config.target_device
        self._quantize_inputs = self._quantization_config.get("quantize_inputs", QUANTIZE_INPUTS)
        self._quantize_outputs = self._quantization_config.get("quantize_outputs", QUANTIZE_OUTPUTS)

        self.ignored_scopes = self._quantization_config.get("ignored_scopes")
        self.target_scopes = self._quantization_config.get("target_scopes")

        self.global_quantizer_constraints: dict[QuantizerGroup, QuantizationConstraints] = {}
        self._ignored_scopes_per_group: dict[QuantizerGroup, list[str]] = {}
        self._target_scopes_per_group: dict[QuantizerGroup, list[str]] = {}

        for quantizer_group in QuantizerGroup:
            self._parse_group_params(self._quantization_config, quantizer_group)

        self._precision_init_type = precision_init_type
        self._precision_init_params = precision_init_params
        self._range_init_params = range_init_params
        self._num_potential_quantized_weights = len(self._target_model.nncf.get_nncf_modules())

    def generate_setup(self) -> SingleConfigQuantizerSetup:
        raise NotImplementedError

    def get_build_time_metric_infos(self):
        raise NotImplementedError

    def _parse_group_params(self, quant_config: dict, quantizer_group: QuantizerGroup):
        group_name = quantizer_group.value
        params_dict = {}
        params_dict_from_config = quant_config.get(group_name, {})
        preset = quant_config.get("preset")
        if self._target_device in ["ANY", "CPU", "GPU"] or (self._target_device is None and preset is not None):
            preset = QuantizationPreset(quant_config.get("preset", QUANTIZATION_PRESET))
            params_dict = preset.get_params_configured_by_preset(quantizer_group)
            overridden_params = params_dict.keys() & params_dict_from_config.keys()
            if overridden_params:
                nncf_logger.info(f"Preset quantizer parameters {overridden_params} explicitly overridden by config.")
        params_dict.update(params_dict_from_config)
        self.global_quantizer_constraints[quantizer_group] = QuantizationConstraints.from_config_dict(params_dict)
        self._ignored_scopes_per_group[quantizer_group] = params_dict_from_config.get("ignored_scopes", [])
        if self.ignored_scopes is not None:
            self._ignored_scopes_per_group[quantizer_group] += self.ignored_scopes
        target_scopes = params_dict_from_config.get("target_scopes")
        if target_scopes is None and self.target_scopes is not None:
            self._target_scopes_per_group[quantizer_group] = self.target_scopes
        else:
            self._target_scopes_per_group[quantizer_group] = target_scopes

    def _get_default_qconfig(self, constraints: QuantizationConstraints = None):
        qconfig = deepcopy(DEFAULT_QUANTIZER_CONFIG)
        if constraints is not None:
            qconfig = constraints.apply_constraints_to(qconfig)
        return qconfig

    def _filter_by_ignored_algo(self, nodes: list[NNCFNode]) -> list[NNCFNode]:
        retval = []
        for node in nodes:
            if "quantization" in node.ignored_algorithms:
                continue
            retval.append(node)
        return retval

    def _assign_qconfig_lists_to_modules(self, weighted_nodes: list[NNCFNode]) -> dict[NNCFNode, list[QuantizerConfig]]:
        raise NotImplementedError

    def get_quantizable_module_nodes(self) -> list[QuantizableWeightedLayerNode]:
        weighted_nodes = self._target_model.nncf.get_original_graph().get_nodes_by_metatypes(
            QUANTIZATION_LAYER_METATYPES
        )
        quantized_modules_with_potential_qconfig = []

        weighted_nodes = self._filter_by_ignored_algo(weighted_nodes)
        weighted_node_vs_qconfig_list = self._assign_qconfig_lists_to_modules(weighted_nodes)

        for node, qconfig_list in weighted_node_vs_qconfig_list.items():
            if qconfig_list is not None:
                qconfig_list_copy = deepcopy(qconfig_list)
                quantized_modules_with_potential_qconfig.append(QuantizableWeightedLayerNode(node, qconfig_list_copy))
        return quantized_modules_with_potential_qconfig


class IQuantizerSetupDisambiguator:
    def select_final_quantizer_setup(self, multi_config_setup: MultiConfigQuantizerSetup) -> SingleConfigQuantizerSetup:
        raise NotImplementedError


class DefaultQuantizerSetupDisambiguator(IQuantizerSetupDisambiguator):
    def __init__(
        self,
        target_model: NNCFNetwork,
        precision_init_type: str = None,
        precision_init_params: BasePrecisionInitParams = None,
        range_init_params: PTRangeInitParams = None,
        override_bit_options_with_precision_init: bool = False,
        hw_config: HWConfig = None,
    ):
        self._precision_init_type = precision_init_type
        self._precision_init_params = precision_init_params
        self._range_init_params = range_init_params
        self._target_model = target_model
        self._override_bit_options_with_precision_init = override_bit_options_with_precision_init
        self.hw_config = hw_config

    @staticmethod
    def select_first_qconfig_with_bitwidth_variants_for_each_point(
        multi_config_setup: MultiConfigQuantizerSetup,
    ) -> MultiConfigQuantizerSetup:
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
            with self._target_model.nncf.temporary_clean_view() as intermediate_model:
                stats = QuantizationBuilder.get_statistics_for_quantizer_setup(
                    intermediate_model, multi_config_setup, self._range_init_params
                )
                bitwidth_varying_only_multi_setup = self.select_first_qconfig_with_bitwidth_variants_for_each_point(
                    multi_config_setup
                )

                init_setup = bitwidth_varying_only_multi_setup.select_first_qconfig_for_each_point()
                intermediate_builder = ExperimentalQuantizationBuilder(
                    bitwidth_varying_only_multi_setup, init_setup, stats, hw_config=self.hw_config
                )
                intermediate_builder.apply_to(intermediate_model)
                intermediate_ctrl = intermediate_builder.build_controller(intermediate_model)

                # intermediate_ctrl.init_range()
                hw_constraints = HardwareQuantizationConstraints()
                if not self._override_bit_options_with_precision_init:
                    for qp_id, qp in multi_config_setup.quantization_points.items():
                        quantizer_module_id = intermediate_ctrl.setup_to_module_id_translation_dict[qp_id]
                        hw_constraints.add(quantizer_module_id, qp.possible_qconfigs)
                final_quantizer_setup = intermediate_ctrl.init_precision(
                    self._precision_init_type, self._precision_init_params, hw_constraints
                )
        else:
            final_quantizer_setup = multi_config_setup.select_first_qconfig_for_each_point()
        return final_quantizer_setup


class PropagationBasedQuantizerSetupGenerator(QuantizerSetupGeneratorBase):
    def __init__(
        self,
        quant_config: dict,
        target_model: NNCFNetwork,
        hw_config: HWConfig = None,
        device: TargetDevice = None,
        precision_init_type: str = None,
        precision_init_params: BasePrecisionInitParams = None,
        range_init_params: PTRangeInitParams = None,
        debug_interface: "QuantizationDebugInterface" = None,
    ):
        super().__init__(
            quant_config, target_model, precision_init_type, precision_init_params, range_init_params, hw_config
        )

        self._pattern_fusing_graph = PatternsManager.get_full_hw_pattern_graph(backend=BackendType.TORCH, device=device)

        self._hw_precision_constraints = HardwareQuantizationConstraints()
        self._debug_interface = debug_interface
        self._num_potential_quantized_activations = 0

        act_config = quant_config.get(QuantizerGroup.ACTIVATIONS.value, {})
        self._unified_scale_ops = act_config.get("unified_scale_ops")

    def generate_setup(self) -> SingleConfigQuantizerSetup:
        quantizable_module_nodes = self.get_quantizable_module_nodes()

        insertion_point_graph = self._target_model.nncf.get_original_insertion_point_graph()
        if self._debug_interface:
            self._debug_interface.visualize_insertion_point_graph(insertion_point_graph)
        from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver

        scales_unification_map = {PTCatMetatype: UNIFICATION_PRODUCING_METATYPES}
        ignored_scopes_for_solver = {
            name: IgnoreReason.USER_REQUESTED for name in self._ignored_scopes_per_group[QuantizerGroup.ACTIVATIONS]
        }
        prop_graph_solver = QuantizerPropagationSolver(
            activation_ignored_scopes=ignored_scopes_for_solver,
            weight_ignored_scopes=self._ignored_scopes_per_group[QuantizerGroup.WEIGHTS],
            activation_target_scopes=self._target_scopes_per_group[QuantizerGroup.ACTIVATIONS],
            weight_target_scopes=self._target_scopes_per_group[QuantizerGroup.WEIGHTS],
            hw_config=self.hw_config,
            default_trait_to_metatype_map=DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT,
            default_qconfig_list=[
                self._get_default_qconfig(constraints=self.global_quantizer_constraints[QuantizerGroup.ACTIVATIONS])
            ],
            quantizable_layer_nodes=quantizable_module_nodes,
            scope_overrides=self._quantization_config.get("scope_overrides", {}),
            global_constraints=self.global_quantizer_constraints,
            additional_unified_scale_op_scopes=self._unified_scale_ops,
            quantize_outputs=self._quantize_outputs,
            scales_unification_map=scales_unification_map,
        )

        merged_ip_graph = insertion_point_graph.get_ip_graph_with_merged_hw_optimized_operations(
            self._pattern_fusing_graph
        )
        quantization_proposal = prop_graph_solver.run_on_ip_graph(merged_ip_graph, ELEMENTWISE_OPERATIONS)
        self._num_potential_quantized_activations = prop_graph_solver.get_num_potential_quantized_activations()

        quantizer_setup = deepcopy(quantization_proposal.quantizer_setup)
        quantization_proposal.quantizer_setup = quantizer_setup

        disambiguator = DefaultQuantizerSetupDisambiguator(
            self._target_model,
            self._precision_init_type,
            self._precision_init_params,
            self._range_init_params,
            override_bit_options_with_precision_init=self.hw_config is None,
            hw_config=self.hw_config,
        )

        single_config_quantizer_setup = disambiguator.select_final_quantizer_setup(
            quantization_proposal.quantizer_setup
        )

        finalized_proposal = quantization_proposal.finalize(
            single_config_quantizer_setup, strict=self.hw_config is not None
        )
        finalized_quantizer_setup = prop_graph_solver.get_final_quantizer_setup(finalized_proposal)
        finalized_quantizer_setup = self._handle_quantize_inputs_option(finalized_quantizer_setup)
        return finalized_quantizer_setup

    def _assign_qconfig_lists_to_modules(self, weighted_nodes: list[NNCFNode]) -> dict[NNCFNode, list[QuantizerConfig]]:
        global_constraints = self.global_quantizer_constraints[QuantizerGroup.WEIGHTS]
        scope_overrides_dict = self._quantization_config.get("scope_overrides", {})
        return assign_qconfig_lists_to_modules(
            weighted_nodes, self._get_default_qconfig(), global_constraints, scope_overrides_dict, self.hw_config
        )

    def _handle_quantize_inputs_option(self, quantizer_setup: SingleConfigQuantizerSetup) -> SingleConfigQuantizerSetup:
        nncf_graph = self._target_model.nncf.get_original_graph()
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
        return QuantizationShareBuildTimeInfo(
            self._num_potential_quantized_activations, self._num_potential_quantized_weights
        )


class QBuilderStateNames:
    BUILD_TIME_METRIC_INFOS = "build_time_metric_infos"
    QUANTIZER_SETUP = "quantizer_setup"


@PT_COMPRESSION_ALGORITHMS.register("quantization")
class QuantizationBuilder(PTCompressionAlgorithmBuilder):
    _state_names = QBuilderStateNames

    def __init__(self, config, should_init: bool = True):
        super().__init__(config, should_init)
        self._debug_interface = QuantizationDebugInterface() if is_debug() else None
        self._weight_quantizers = OrderedDict()  # Quantizers applied via UpdateWeights
        self._non_weight_quantizers = OrderedDict()  # All the other quantizers
        self._quantizers_input_shapes = OrderedDict()
        self._processed_insertion_points: set[PTTargetPoint] = set()
        self._groups_of_adjacent_quantizers: GroupsOfAdjacentQuantizers = GroupsOfAdjacentQuantizers()
        self._setup_to_module_id_translation_dict: dict[QuantizationPointId, QuantizerId] = {}
        self.eval_ops_exec_ctx = []
        self._build_time_metric_infos: Optional[QuantizationShareBuildTimeInfo] = None
        self.hw_config = None
        self._legacy_single_config_quantizer_setup_from_comp_state: Optional[SingleConfigQuantizerSetup] = None
        self._pt_quantizer_setup: Optional[PTQuantizerSetup] = None
        self._minmax_values_for_range_init: Optional[dict[QuantizationPointId, MinMaxTensorStatistic]] = {}

        # can be False to disable setting of adjust padding operations on precision init, because it may add unnecessary
        # noise on model evaluation (e.g. in AutoQ)
        self._should_setup_adjust_pad_ops = True
        hw_config_type = None
        self._target_device = self.config.get("target_device", "ANY")
        hw_config_type = get_hw_config_type(self._target_device)
        if hw_config_type is not None:
            hw_config_path = PTHWConfig.get_path_to_hw_config(hw_config_type)
            self.hw_config = PTHWConfig.from_json(hw_config_path)

        algo_config = self._get_algo_specific_config_section()
        if self._target_device == "NPU" and "preset" in algo_config:
            msg = "The NPU target device does not support presets."
            raise nncf.InternalError(msg)
        if self._target_device == "CPU_SPR":
            msg = "The CPU_SPR target device does not supported."
            raise nncf.InternalError(msg)

        self._range_init_params = None
        self._precision_init_type = None
        self._precision_init_params = None
        if self.should_init:
            self._parse_init_params()

        self._use_logarithm_scale_per_group: dict[QuantizerGroup, bool] = {}

        for quantizer_group in QuantizerGroup:
            group_name = quantizer_group.value
            params_dict = self._algo_config.get(group_name, {})
            self._use_logarithm_scale_per_group[quantizer_group] = params_dict.get(
                "logarithm_scale", QUANTIZATION_LOGARITHM_SCALE
            )

        self._overflow_fix = self._algo_config.get("overflow_fix", QUANTIZATION_OVERFLOW_FIX)
        self._device_for_callable_obj_creation = "cpu"

    def _load_state_without_name(self, state_without_name: dict[str, Any]):
        """
        Initializes object from the state.

        :param state_without_name: Output of `get_state()` method.
        """
        quantizer_setup_state = state_without_name[self._state_names.QUANTIZER_SETUP]
        version = state_without_name.get(QUANTIZER_BUILDER_STATE_VERSION_SAVE_NAME, QuantizerBuilderStateVersion.v1)
        if version == QuantizerBuilderStateVersion.v1:
            self._legacy_single_config_quantizer_setup_from_comp_state = SingleConfigQuantizerSetup.from_state(
                quantizer_setup_state
            )
        else:
            self._pt_quantizer_setup = PTQuantizerSetup.from_state(quantizer_setup_state)
        self._build_time_metric_infos = QuantizationShareBuildTimeInfo.from_state(
            state_without_name[self._state_names.BUILD_TIME_METRIC_INFOS]
        )

    def _get_state_without_name(self) -> dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        build_time_metric_infos_state = {}
        if self._build_time_metric_infos:
            build_time_metric_infos_state = self._build_time_metric_infos.get_state()
        quantizer_setup_state = {}
        if self._pt_quantizer_setup:
            quantizer_setup_state = self._pt_quantizer_setup.get_state()
        return {
            self._state_names.QUANTIZER_SETUP: quantizer_setup_state,
            self._state_names.BUILD_TIME_METRIC_INFOS: build_time_metric_infos_state,
            QUANTIZER_BUILDER_STATE_VERSION_SAVE_NAME: max(QuantizerBuilderStateVersion).value,
        }

    def _parse_init_params(self):
        self._range_init_params = self._parse_range_init_params()
        self._precision_init_type, self._precision_init_params = self._parse_precision_init_params(
            self._algo_config.get("initializer", {})
        )

    def _parse_range_init_params(self) -> Optional[PTRangeInitParams]:
        range_init_params = extract_range_init_params(self.config)
        return PTRangeInitParams(**range_init_params) if range_init_params is not None else None

    def _parse_precision_init_params(self, initializer_config: dict) -> tuple[str, BasePrecisionInitParams]:
        init_precision_config = initializer_config.get("precision")
        if not init_precision_config:
            return None, None
        precision_init_type = init_precision_config.get("type", "manual")
        if precision_init_type not in PRECISION_INIT_TYPES_VS_DESCRIPTION:
            msg = f"Unrecognized precision init type: {precision_init_type}"
            raise nncf.InternalError(msg)
        if precision_init_type == "hawq":
            try:
                precision_init_args = self.config.get_extra_struct(QuantizationPrecisionInitArgs)
            except KeyError as e:
                msg = (
                    "Specified non-manual precision initialization in the NNCF config, "
                    "but the initializing data loader and loss criterion are not provided as an extra struct. "
                    "Refer to `NNCFConfig.register_extra_structs` and the `QuantizationPrecisionInitArgs` "
                    "class"
                )
                raise ValueError(msg) from e
            precision_init_params = HAWQPrecisionInitParams.from_config(init_precision_config, precision_init_args)
        elif precision_init_type == "autoq":
            if self.hw_config is not None and self.hw_config.target_device != HWConfigType.NPU.value:
                msg = (
                    f"Unsupported device ({self.hw_config.target_device})."
                    f" Automatic Precision Initialization only supports for target_device NONE or NPU"
                )
                raise ValueError(msg)
            try:
                precision_init_args = self.config.get_extra_struct(AutoQPrecisionInitArgs)
            except KeyError as e:
                msg = (
                    "Specified Automated precision initialization in the NNCF config, "
                    "but the initializing data loader and loss criterion are not provided as an extra "
                    "struct. Refer to `NNCFConfig.register_extra_structs` and the "
                    "`AutoQPrecisionInitArgs` class"
                )
                raise ValueError(msg) from e

            hw_config_type = None
            if self.hw_config is not None:
                hw_config_type = HWConfigType(self.hw_config.target_device)
            precision_init_params = AutoQPrecisionInitParams.from_config(
                init_precision_config, precision_init_args, hw_config_type
            )
        elif precision_init_type == "manual":
            precision_init_params = ManualPrecisionInitParams.from_config(init_precision_config)
        else:
            msg = f"Unhandled precision init type: {precision_init_type}"
            raise ValueError(msg)
        return precision_init_type, precision_init_params

    def _get_minmax_values_for_quantizer_locations(
        self,
        quantizer_setup: SingleConfigQuantizerSetup,
        tensor_statistics: dict[PTTargetPoint, dict[ReductionAxes, TensorStatistic]],
        target_model_graph: PTNNCFGraph,
    ) -> dict[QuantizationPointId, MinMaxTensorStatistic]:
        retval = {}
        for qp_id, qp in quantizer_setup.quantization_points.items():
            qip = qp.insertion_point
            tp = PTTargetPointTranslator.translate(qip)
            if tp not in tensor_statistics:
                nncf_logger.debug(f"TP {tp} not found in tensor statistics")
                retval[qp_id] = None
            else:
                target_node = target_model_graph.get_node_by_name(tp.target_node_name)
                if qp.is_weight_quantization_point():
                    layer_attrs = target_node.layer_attributes
                    assert isinstance(layer_attrs, WeightedLayerAttributes)
                    input_shape = get_weight_shape_legacy(layer_attrs)
                    channel_idx = get_target_dim_for_compression_legacy(layer_attrs)
                else:
                    input_shape = target_model_graph.get_input_shape_for_insertion_point(qp.insertion_point)
                    channel_idx = 1  # channel dim for activations
                scale_shape = tuple(
                    get_scale_shape(input_shape, qp.is_weight_quantization_point(), qp.qconfig.per_channel, channel_idx)
                )

                if scale_shape not in tensor_statistics[tp]:
                    nncf_logger.debug(f"Did not collect tensor statistics at {tp} for shape {scale_shape}")
                    retval[qp_id] = None
                else:
                    minmax_stat = pt_convert_stat_to_min_max_tensor_stat(tensor_statistics[tp][scale_shape])
                    retval[qp_id] = minmax_stat
        return retval

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        # TODO (vshampor): a simpler solution would be to always create callables on CPU and
        # to move these to model-specific device upon actual application, but would this impact
        # the time required to create a compressed model?
        self._device_for_callable_obj_creation = get_model_device(target_model)
        target_model.nncf.register_compression_module_type(ExtraCompressionModuleType.EXTERNAL_QUANTIZER)
        if self._pt_quantizer_setup is None:
            self._pt_quantizer_setup = self._get_quantizer_setup(target_model)

        (
            insertion_commands,
            setup_to_module_id_translation_dict,
        ) = self._build_insertion_commands_list_for_quantizer_setup(
            self._pt_quantizer_setup, target_model, self._minmax_values_for_range_init
        )

        transformation_layout = PTTransformationLayout()
        for command in insertion_commands:
            transformation_layout.register(command)

        self._setup_to_module_id_translation_dict = setup_to_module_id_translation_dict
        all_quantizations = {}
        all_quantizations.update({k: v.quantizer_module_ref for k, v in self._weight_quantizers.items()})
        all_quantizations.update({k: v.quantizer_module_ref for k, v in self._non_weight_quantizers.items()})
        self._groups_of_adjacent_quantizers.parse_from_quantizer_setup(
            all_quantizations, self._pt_quantizer_setup, setup_to_module_id_translation_dict
        )

        # NOTE: Order of activations must be the same to correctly broadcast parameters (e.g. scales) in distributed
        # mode (see call of `_dist_broadcast_coalesced` in torch/nn/parallel/distributed.py for more details)

        target_model.nncf.sort_compression_modules(ExtraCompressionModuleType.EXTERNAL_QUANTIZER)

        if self._debug_interface is not None:
            target_model.nncf.debug_interface.add_interface(self._debug_interface)

        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        all_quantizations = get_state_dict_names_with_modules(target_model, quantization_types)
        target_model.nncf._load_listener = LoadStateListener(target_model, all_quantizations)

        return transformation_layout

    @staticmethod
    def get_statistics_for_quantizer_setup(
        target_model: NNCFNetwork, quantizer_setup: QuantizerSetupBase, range_init_params: PTRangeInitParams
    ) -> dict[PTTargetPoint, dict[ReductionAxes, TensorStatistic]]:
        if range_init_params is None:
            return {}
        observation_points_vs_collectors_dict = (
            StatCollectorGenerator.generate_collectors_for_range_init_statistics_collection(
                target_model.nncf.get_original_graph(), quantizer_setup, range_init_params
            )
        )

        with target_model.nncf.temporary_clean_view() as intermediate_model:
            stat_builder = TensorStatisticsCollectionBuilder(NNCFConfig(), observation_points_vs_collectors_dict)
            stat_builder.apply_to(intermediate_model)
            stat_ctrl = stat_builder.build_controller(intermediate_model)
            runner = SimpleDataLoaderRunner(intermediate_model, range_init_params.device)
            runner.progressbar_description = "Collecting tensor statistics"
            with training_mode_switcher(intermediate_model, is_training=False):
                # Run statistics collection in eval mode, otherwise it may fail because graph was built in eval mode
                runner.run(range_init_params.init_range_data_loader, range_init_params.get_max_num_init_steps())

        retval = {}
        for ip, rs_vs_collector in stat_ctrl.ip_vs_collector_dict.items():
            retval[ip] = {rs: collector.get_statistics() for rs, collector in rs_vs_collector.items()}
        return retval

    def _get_statistics_for_final_range_init(
        self, target_model: NNCFNetwork, quantizer_setup: QuantizerSetupBase, range_init_params: PTRangeInitParams
    ) -> dict[PTTargetPoint, dict[ReductionAxes, TensorStatistic]]:
        return self.get_statistics_for_quantizer_setup(target_model, quantizer_setup, range_init_params)

    def _get_single_config_quantizer_setup(self, target_model) -> SingleConfigQuantizerSetup:
        setup_generator = PropagationBasedQuantizerSetupGenerator(
            self._algo_config,
            target_model,
            self.hw_config,
            self._target_device,
            self._precision_init_type,
            self._precision_init_params,
            self._range_init_params,
            self._debug_interface,
        )
        single_config_quantizer_setup = setup_generator.generate_setup()
        self._build_time_metric_infos = setup_generator.get_build_time_metric_infos()
        return single_config_quantizer_setup

    def _get_quantizer_setup(self, target_model: NNCFNetwork) -> PTQuantizerSetup:
        if self._legacy_single_config_quantizer_setup_from_comp_state is None:
            single_config_quantizer_setup = self._get_single_config_quantizer_setup(target_model)
        else:
            single_config_quantizer_setup = self._legacy_single_config_quantizer_setup_from_comp_state

        target_model_graph = target_model.nncf.get_original_graph()

        if is_main_process() and self.should_init:
            stats_for_range_init = self._get_statistics_for_final_range_init(
                target_model, single_config_quantizer_setup, self._range_init_params
            )
            self._minmax_values_for_range_init = self._get_minmax_values_for_quantizer_locations(
                single_config_quantizer_setup, stats_for_range_init, target_model_graph
            )

            self._check_and_log_missing_stats_for_setup(
                single_config_quantizer_setup, self._minmax_values_for_range_init
            )

        bitwidth_per_scope = BasePrecisionInitializer.get_bitwidth_per_scope(single_config_quantizer_setup)
        str_bw = [str(element) for element in bitwidth_per_scope]
        nncf_logger.debug("\n".join(['\n"bitwidth_per_scope": [', ",\n".join(str_bw), "]"]))

        setup = PTQuantizerSetup(
            single_config_quantizer_setup.unified_scale_groups,
            single_config_quantizer_setup.shared_input_operation_set_groups,
        )

        for qp_id, qp in single_config_quantizer_setup.quantization_points.items():
            qconfig = qp.qconfig
            insertion_point = qp.insertion_point  # QuantizationInsertionPointBase

            compression_lr_multiplier = self._get_compression_lr_multiplier()

            half_range = False
            if self.hw_config and qp.is_weight_quantization_point():
                target_node = target_model_graph.get_node_by_name(insertion_point.target_node_name)
                if self.hw_config.target_device in ["CPU", "ANY"] and qconfig.num_bits == 8:
                    if self._overflow_fix == "enable":
                        half_range = True
                        quantizers_with_overflow_fix_str = "all weight quantizers"
                    elif self._overflow_fix == "first_layer_only":
                        if target_node in get_first_nodes_of_type(target_model_graph, ["conv2d", "conv3d"]):
                            half_range = True
                            quantizers_with_overflow_fix_str = "first convolution weight quantizers"
                    elif self._overflow_fix != "disable":
                        msg = f"Unknown overflow fix type: {self._overflow_fix}"
                        raise nncf.InternalError(msg)
                    if half_range:
                        nncf_logger.debug(f"Overflow issue fix will be applied to {quantizers_with_overflow_fix_str}")

            if qp.is_weight_quantization_point():
                use_logarithm_scale = self._use_logarithm_scale_per_group[QuantizerGroup.WEIGHTS]
                narrow_range = qconfig.num_bits == 8 and not half_range
            else:
                use_logarithm_scale = self._use_logarithm_scale_per_group[QuantizerGroup.ACTIVATIONS]
                narrow_range = False

            if qp.is_weight_quantization_point():
                target_node = target_model_graph.get_node_by_name(insertion_point.target_node_name)
                layer_attributes = target_node.layer_attributes
                assert isinstance(layer_attributes, WeightedLayerAttributes)
                scale_shape = get_scale_shape(
                    get_weight_shape_legacy(layer_attributes),
                    is_weights=True,
                    per_channel=qconfig.per_channel,
                    channel_idx=get_target_dim_for_compression_legacy(layer_attributes),
                )
            else:
                input_shape = target_model_graph.get_input_shape_for_insertion_point(insertion_point)
                scale_shape = get_scale_shape(list(input_shape), is_weights=False, per_channel=qconfig.per_channel)

            qspec = PTQuantizerSpec.from_config(
                qconfig,
                narrow_range=narrow_range,
                scale_shape=tuple(scale_shape),
                logarithm_scale=use_logarithm_scale,
                half_range=half_range,
                is_quantized_on_export=qp.is_weight_quantization_point(),
                compression_lr_multiplier=compression_lr_multiplier,
            )
            pt_qp = PTQuantizationPoint(
                qspec, PTTargetPointTranslator.translate(insertion_point), qp.directly_quantized_operator_node_names
            )
            setup.add_quantization_point(qp_id, pt_qp)

        return setup

    def _build_controller(self, model: NNCFNetwork) -> PTCompressionAlgorithmController:
        return QuantizationController(
            model,
            self.config,
            self._debug_interface,
            self._weight_quantizers,
            self._non_weight_quantizers,
            self._groups_of_adjacent_quantizers,
            self._quantizers_input_shapes,
            build_time_metric_info=self._build_time_metric_infos,
            build_time_range_init_params=self._range_init_params,
        )

    def __create_quantize_module(self, quantizer_spec: PTQuantizerSpec):
        quantizer_cls = QUANTIZATION_MODULES.get(quantizer_spec.mode)
        return quantizer_cls(quantizer_spec)

    @staticmethod
    def _get_adjust_padding_args(
        target_model_graph: NNCFGraph,
        quantization_point: PTQuantizationPoint,
        activation_quantizer: BaseQuantizer,
        quantization_points: list[PTQuantizationPoint],
    ) -> list[AdjustPaddingArgs]:
        result = []
        for op_node_name in quantization_point.directly_quantized_operator_node_names:
            weight_bitwidth = None
            for qp in quantization_points:
                is_weight = qp.is_weight_quantization_point()
                if is_weight and (qp.target_point.target_node_name == op_node_name):
                    weight_bitwidth = qp.qspec.num_bits
                    break
            if weight_bitwidth:
                is_applicable = False
                target_node = target_model_graph.get_node_by_name(op_node_name)
                if target_node.metatype in [PTModuleConv2dMetatype, PTModuleDepthwiseConv2dSubtype]:
                    layer_attrs = target_node.layer_attributes
                    assert isinstance(layer_attrs, ConvolutionLayerAttributes)
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

    def _add_adjust_padding_ops(self, adjust_padding_args: list[AdjustPaddingArgs]):
        commands = []
        for args in adjust_padding_args:
            ap = CalculatePaddingAdjustment(args.activation_quantizer)
            op = UpdatePaddingValue(ap).to(self._device_for_callable_obj_creation)
            insertion_point = PTTargetPoint(
                target_type=TargetType.PRE_LAYER_OPERATION, target_node_name=args.module_op_node_name
            )
            nncf_logger.debug_once(f"Padding will be adjusted for {args.module_op_node_name}")
            commands.append(PTInsertionCommand(insertion_point, op, TransformationPriority.DEFAULT_PRIORITY))
        return commands

    @staticmethod
    def _check_and_log_missing_stats_for_setup(
        quantizer_setup: SingleConfigQuantizerSetup,
        minmax_values_for_range_init: dict[QuantizationPointId, MinMaxTensorStatistic],
    ):
        tps_with_uncollected_stats = set()
        for qp_id in quantizer_setup.quantization_points:
            if qp_id not in minmax_values_for_range_init:
                tps_with_uncollected_stats.add(quantizer_setup.quantization_points[qp_id].insertion_point)
        if tps_with_uncollected_stats:
            nncf_logger.error("Tensor statistics for the following locations were not collected:")
            for tp in tps_with_uncollected_stats:
                nncf_logger.error(f"\t{tp}")
            nncf_logger.error(
                "The corresponding quantizer range will not be initialized! If the model has "
                "data-dependent control flow branches, make sure that your initializing data loader is "
                "producing data that allows the model cover to all of these branches. If this is not the "
                "case, consider adding the corresponding nodes to `ignored_scopes`."
            )

    def _build_insertion_commands_list_for_quantizer_setup(
        self,
        quantizer_setup: PTQuantizerSetup,
        target_model: NNCFNetwork,
        minmax_values_for_range_init: dict[QuantizationPointId, MinMaxTensorStatistic],
    ) -> tuple[list[PTInsertionCommand], dict[QuantizationPointId, QuantizerId]]:
        insertion_commands = []
        qp_id_vs_quant_module_id_dict: dict[QuantizationPointId, QuantizerId] = {}
        target_model_graph = target_model.nncf.get_original_graph()
        non_unified_scales_quantization_point_ids = set(quantizer_setup.quantization_points.keys())
        already_weight_quantized_shared_layers: dict[str, QuantizerId] = {}

        for unified_scales_group in quantizer_setup.unified_scale_groups.values():
            for us_qp_id in unified_scales_group:
                non_unified_scales_quantization_point_ids.discard(us_qp_id)

            (
                filtered_unified_scales_group,
                shared_weight_quantized_layers_in_group,
            ) = self._remove_shared_layer_weight_quantization_point_duplicates(
                unified_scales_group, quantizer_setup, target_model_graph
            )

            quant_module_id, commands = self._build_commands_for_single_unified_scale_group(
                target_model, quantizer_setup, filtered_unified_scales_group, minmax_values_for_range_init
            )

            for layer_name in shared_weight_quantized_layers_in_group:
                if layer_name in already_weight_quantized_shared_layers:
                    msg = (
                        "Attempted to assign a unified-scale quantizer to a shared layer node that has "
                        "already had its weights quantized by another unified-scale quantizer!"
                    )
                    raise nncf.InternalError(msg)
                already_weight_quantized_shared_layers[layer_name] = quant_module_id

            for us_qp_id in unified_scales_group:
                qp_id_vs_quant_module_id_dict[us_qp_id] = quant_module_id
            insertion_commands += commands

        for qp_id in non_unified_scales_quantization_point_ids:
            qp = quantizer_setup.quantization_points[qp_id]
            nncf_node = target_model_graph.get_node_by_name(qp.target_point.target_node_name)
            if qp.is_weight_quantization_point() and nncf_node.is_shared():
                layer_name = nncf_node.layer_name
                if layer_name in already_weight_quantized_shared_layers:
                    nncf_logger.debug_once(
                        f"Filtering a regular weight quantization point {qp_id} - "
                        f"already quantized as a shared layer {nncf_node.layer_name}",
                    )
                    qp_id_vs_quant_module_id_dict[qp_id] = already_weight_quantized_shared_layers[layer_name]
                    continue

            qspec = quantizer_setup.quantization_points[qp_id].qspec
            tp = quantizer_setup.quantization_points[qp_id].target_point

            range_init_minmax_values = None
            if minmax_values_for_range_init:
                minmax_stat = minmax_values_for_range_init.get(qp_id)
                if minmax_stat is not None:
                    range_init_minmax_values = (minmax_stat.min_values, minmax_stat.max_values)

            quantizer_module_id, commands = self._quantize_at_points_by_single_module(
                target_model,
                [
                    tp,
                ],
                qspec,
                range_init_minmax_values,
            )

            if (
                qp.is_weight_quantization_point()
                and nncf_node.is_shared()
                and nncf_node.layer_name not in already_weight_quantized_shared_layers
            ):
                already_weight_quantized_shared_layers[nncf_node.layer_name] = quantizer_module_id

            qp_id_vs_quant_module_id_dict[qp_id] = quantizer_module_id
            insertion_commands += commands

        adjust_padding_args = self._collect_adjust_padding_args(
            non_unified_scales_quantization_point_ids,
            qp_id_vs_quant_module_id_dict,
            quantizer_setup,
            target_model_graph,
        )

        commands = self._add_adjust_padding_ops(adjust_padding_args)
        if commands:
            insertion_commands += commands

        return insertion_commands, qp_id_vs_quant_module_id_dict

    def _remove_shared_layer_weight_quantization_point_duplicates(
        self,
        unified_scales_group: set[QuantizationPointId],
        quantizer_setup: PTQuantizerSetup,
        target_model_graph: NNCFGraph,
    ) -> tuple[set[QuantizationPointId], set[str]]:
        observed_shared_layer_names = set()
        retval = set()
        for us_qp_id in unified_scales_group:
            qp = quantizer_setup.quantization_points[us_qp_id]
            if qp.is_weight_quantization_point():
                nncf_node = target_model_graph.get_node_by_name(qp.target_point.target_node_name)
                if nncf_node.is_shared():
                    if nncf_node.layer_name not in observed_shared_layer_names:
                        observed_shared_layer_names.add(nncf_node.layer_name)
                    else:
                        nncf_logger.debug_once(
                            f"Filtering a unified-scale weight quantization point {us_qp_id} "
                            f"- already quantized as a shared layer {nncf_node.layer_name}",
                        )
                        continue
            retval.add(us_qp_id)
        return retval, observed_shared_layer_names

    def _collect_adjust_padding_args(
        self,
        non_unified_scales_quantization_point_ids: set[QuantizationPointId],
        qp_id_vs_quant_module_id_dict: dict[QuantizationPointId, QuantizerId],
        quantizer_setup: PTQuantizerSetup,
        target_model_graph: NNCFGraph,
    ) -> list[AdjustPaddingArgs]:
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
            tp = quantizer_setup.quantization_points[wqp_id].target_point
            target_node = target_model_graph.get_node_by_name(tp.target_node_name)

            op_type = target_node.metatype
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
                    args = self._get_adjust_padding_args(
                        target_model_graph,
                        wqp,
                        activation_quantizer,
                        list(quantizer_setup.quantization_points.values()),
                    )
                    if args:
                        adjust_padding_args.extend(args)
        return adjust_padding_args

    def _build_commands_for_single_unified_scale_group(
        self,
        target_model: NNCFNetwork,
        quantizer_setup: PTQuantizerSetup,
        unified_scales_group: set[QuantizationPointId],
        minmax_values_for_range_init: dict[QuantizationPointId, MinMaxTensorStatistic],
    ) -> tuple[QuantizerId, list[PTInsertionCommand]]:
        qp_ids_list_for_current_group = list(unified_scales_group)

        # The primary insertion point (to be associated with the actual quantizer module, not just hooks to it)
        # will be determined based on the string representation of said insertion point, to avoid random selection.
        # Weight insertion points are given priority.
        weight_qp_ids = [
            qp_id
            for qp_id in qp_ids_list_for_current_group
            if quantizer_setup.quantization_points[qp_id].is_weight_quantization_point()
        ]
        act_qp_ids = [
            qp_id
            for qp_id in qp_ids_list_for_current_group
            if quantizer_setup.quantization_points[qp_id].is_activation_quantization_point()
        ]

        def ip_str_repr_key_lambda(x):
            return str(quantizer_setup.quantization_points[x].target_point.target_node_name)

        sorted_wqp_ids = sorted(weight_qp_ids, key=ip_str_repr_key_lambda)
        sorted_aqp_ids = sorted(act_qp_ids, key=ip_str_repr_key_lambda)
        sorted_qp_ids = sorted_wqp_ids + sorted_aqp_ids

        primary_qp_id = sorted_qp_ids[0]
        linked_qp_ids = sorted_qp_ids[1:]
        qspec = quantizer_setup.quantization_points[primary_qp_id].qspec
        linked_qspecs = [quantizer_setup.quantization_points[qp_id].qspec for qp_id in linked_qp_ids]
        for linked_qspec in linked_qspecs:
            if qspec != linked_qspec:
                msg = "The qspecs for unified scale quantization points should be identical!"
                raise nncf.InternalError(msg)

        range_init_minmax_values = None
        if minmax_values_for_range_init:
            # Hopefully this will suffice.
            # TODO: gather unified statistic by linking stat collectors_and_modules_to_init instead
            min_values = None
            max_values = None
            for qp_id in sorted_qp_ids:
                minmax_stat = minmax_values_for_range_init.get(qp_id)
                if minmax_stat is None:
                    continue

                if min_values is None:
                    min_values = minmax_stat.min_values.data
                else:
                    min_values = torch.min(min_values, minmax_stat.min_values.data)

                if max_values is None:
                    max_values = minmax_stat.max_values.data
                else:
                    max_values = torch.max(max_values, minmax_stat.max_values.data)
            if min_values is not None and max_values is not None:
                range_init_minmax_values = min_values, max_values

        target_points = [quantizer_setup.quantization_points[qp_id].target_point for qp_id in sorted_qp_ids]
        quantizer_module_id, commands = self._quantize_at_points_by_single_module(
            target_model, target_points, qspec, range_init_minmax_values
        )
        return quantizer_module_id, commands

    def _select_final_qconfig(self, quantizer_config_list: list[QuantizerConfig]) -> QuantizerConfig:
        # Quantizer config list entries should arrive in the same order as they are listed
        # in the HW config, where they are sorted by descending order of priority
        return quantizer_config_list[0]

    def _quantize_at_points_by_single_module(
        self,
        target_model: NNCFNetwork,
        insertion_points: list[PTTargetPoint],
        qspec: PTQuantizerSpec,
        range_init_minmax_values: tuple[torch.Tensor, torch.Tensor] = None,
    ) -> tuple[QuantizerId, list[PTInsertionCommand]]:
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
        target_model_graph = target_model.nncf.get_original_graph()
        if not insertion_points:
            msg = "No insertion points to put quantizers into!"
            raise nncf.InternalError(msg)

        def is_weights(ip: PTTargetPoint) -> bool:
            return ip.target_type is TargetType.OPERATION_WITH_WEIGHTS

        primary_ip = insertion_points[0]

        quantizer = self.__create_quantize_module(qspec).to(self._device_for_callable_obj_creation)
        if range_init_minmax_values is not None:
            # Need to cast to the model's current dtype since the statistics could have been gathered in an
            # AMP autocast model (and therefore be FP16 since AMP autocast switches precision of activations
            # at forward pass time)
            own_type = get_model_dtype(target_model)
            min_values = range_init_minmax_values[0].data.type(own_type)
            max_values = range_init_minmax_values[1].data.type(own_type)

            quantizer.apply_minmax_init(min_values=min_values, max_values=max_values, log_module_name=str(primary_ip))

        qids: list[QuantizerId] = []
        for ip in insertion_points:
            if is_weights(ip):
                qids.append(WeightQuantizerId(ip.target_node_name))
            else:
                qids.append(NonWeightQuantizerId(ip.target_node_name, ip.input_port_id))

        serialized_insertions_list = [str(x) for x in qids]
        external_quantizer_storage_key = ";".join(serialized_insertions_list)
        if len(insertion_points) > 1:
            linked_quantizers_str = "\n".join(serialized_insertions_list)
            nncf_logger.info_once(f"Scales will be unified for quantizer group:\n{linked_quantizers_str}\n")

        if is_weights(primary_ip):
            primary_qid = WeightQuantizerId(primary_ip.target_node_name)
            self._weight_quantizers[primary_qid] = WeightQuantizerInfo(
                quantizer, target_model.nncf.get_containing_module(primary_ip.target_node_name), insertion_points
            )
            module_node = target_model_graph.get_node_by_name(primary_ip.target_node_name)
            layer_attributes = module_node.layer_attributes
            input_shape = get_weight_shape_legacy(layer_attributes)
            self._quantizers_input_shapes[primary_qid] = tuple(input_shape)
        else:
            primary_qid = NonWeightQuantizerId(primary_ip.target_node_name, primary_ip.input_port_id)
            self._non_weight_quantizers[primary_qid] = NonWeightQuantizerInfo(quantizer, insertion_points)
            input_shape = target_model_graph.get_input_shape_for_insertion_point(insertion_points[0])
            self._quantizers_input_shapes[primary_qid] = input_shape

        if not (is_weights(primary_ip) and len(insertion_points) == 1):
            assert external_quantizer_storage_key not in target_model.nncf.get_compression_modules_by_type(
                ExtraCompressionModuleType.EXTERNAL_QUANTIZER
            )

            target_model.nncf.add_compression_module(
                external_quantizer_storage_key, quantizer, ExtraCompressionModuleType.EXTERNAL_QUANTIZER
            )

        insertion_commands = []
        for curr_insertion_point in insertion_points:
            if curr_insertion_point in self._processed_insertion_points:
                msg = f"Insertion point {str(curr_insertion_point)} already quantized!"
                raise nncf.InternalError(msg)
            self._processed_insertion_points.add(curr_insertion_point)

            if is_weights(curr_insertion_point):
                if len(insertion_points) == 1:
                    # For backward compatibility, if only one weight is quantized by a single quantizer,
                    # insert UpdateWeight ops with a genuine quantizer module
                    callable_obj = quantizer
                else:
                    # Otherwise use external quantizer module storage since the quantization points will have to
                    # share the single module and this would be impossible for multiple weight quantizer sharing if
                    # the corresponding UpdateWeights operations contained real modules (these would simply get copied
                    # by PyTorch internals)
                    callable_obj = ExternalQuantizerCallHook(external_quantizer_storage_key, self._debug_interface)
            else:
                # Hooks will be identical for each affected op_address in the linked scenario
                # - will call one and the same quantizer
                callable_obj = ExternalQuantizerCallHook(external_quantizer_storage_key, self._debug_interface)

            nncf_logger.debug_once(
                f"Performing "
                f"{'signed' if quantizer.signed else 'unsigned'} "
                f"{'logarithm_scale' if quantizer.is_using_log_scale_storage else ''} "
                f"{'weight' if is_weights(curr_insertion_point) else 'activation'} "
                f"quantization for: {str(curr_insertion_point)}",
            )

            insertion_commands.append(
                PTInsertionCommand(curr_insertion_point, callable_obj, TransformationPriority.QUANTIZATION_PRIORITY)
            )
        return primary_qid, insertion_commands

    def _are_frozen_layers_allowed(self) -> tuple[bool, str]:
        message_template = Template("Frozen layers are$denial allowed for $algo_prefix quantization")
        bits = set()
        bits.update({wq.quantizer_module_ref.num_bits for wq in self._weight_quantizers.values()})
        bits.update({nwq.quantizer_module_ref.num_bits for nwq in self._non_weight_quantizers.values()})

        if self._precision_init_params or len(bits) > 1:
            return False, message_template.substitute(denial=" not", algo_prefix="mixed precision")

        if len(bits) == 1:
            bitwidth = bits.pop()
            algo_prefix = f"INT{bitwidth}"
            if bitwidth == 8:
                return True, message_template.substitute(denial="", algo_prefix=algo_prefix)
            return False, message_template.substitute(denial=" not", algo_prefix=algo_prefix)
        return True, message_template.substitute(denial="", algo_prefix="empty")

    def _get_compression_lr_multiplier(self) -> Optional[float]:
        return self.config.get_redefinable_global_param_value_for_algo("compression_lr_multiplier", self.name)

    def initialize(self, model: NNCFNetwork) -> None:
        if is_main_process() and self.should_init:
            bn_adapt_params = self._parse_bn_adapt_params()
            if bn_adapt_params is not None:
                bn_adaptation = BatchnormAdaptationAlgorithm(
                    **extract_bn_adaptation_init_params(self.config, "quantization")
                )
                bn_adaptation.run(model)


@api()
class QuantizationController(QuantizationControllerBase):
    """
    Controller for the quantization algorithm in PT.
    """

    def __init__(
        self,
        target_model: NNCFNetwork,
        config: NNCFConfig,
        debug_interface: "QuantizationDebugInterface",
        weight_quantizers: dict[WeightQuantizerId, WeightQuantizerInfo],
        non_weight_quantizers: dict[NonWeightQuantizerId, NonWeightQuantizerInfo],
        groups_of_adjacent_quantizers: GroupsOfAdjacentQuantizers,
        quantizers_input_shapes: dict[QuantizerId, tuple[int]],
        build_time_metric_info: QuantizationShareBuildTimeInfo = None,
        build_time_range_init_params: PTRangeInitParams = None,
    ):
        super().__init__(target_model)
        self._loss = ZeroCompressionLoss(get_model_device(target_model))
        self._scheduler = BaseCompressionScheduler()
        self.debug_interface = debug_interface
        self.config = config
        algo_config = self._get_algo_config()
        self._build_time_range_init_params = build_time_range_init_params

        self.weight_quantizers: dict[WeightQuantizerId, WeightQuantizerInfo] = weight_quantizers
        self.non_weight_quantizers: dict[NonWeightQuantizerId, NonWeightQuantizerInfo] = non_weight_quantizers
        self.all_quantizations: dict[QuantizerId, BaseQuantizer] = OrderedDict()
        self.all_quantizations.update({k: v.quantizer_module_ref for k, v in self.weight_quantizers.items()})
        self.all_quantizations.update({k: v.quantizer_module_ref for k, v in self.non_weight_quantizers.items()})
        self._quantizers_input_shapes = quantizers_input_shapes
        self._distributed = False
        self._groups_of_adjacent_quantizers = groups_of_adjacent_quantizers
        self._bn_adaptation = None
        self._build_time_metric_info = build_time_metric_info
        self._target_device = self.config.get("target_device", "ANY")

        should_export_to_onnx_qdq = algo_config.get(
            "export_to_onnx_standard_ops", QUANTIZATION_EXPORT_TO_ONNX_STANDARD_OPS
        )
        if should_export_to_onnx_qdq:
            warning_deprecated(
                "The config option `export_to_onnx_standard_ops` is deprecated and will be removed "
                "in a future version. Please use the `nncf.strip(quantized_model)` method before export to ONNX "
                "to get model with QuantizeLinear-DequantizeLinear node pairs."
            )
            export_mode = QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS
        else:
            export_mode = QuantizerExportMode.FAKE_QUANTIZE

        for quantizer in self.all_quantizations.values():
            quantizer.set_export_mode(export_mode)

        params = algo_config.get("params", None)
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
                nncf_logger.debug(f"Disabled quantization on export to ONNX: {quantizer_id}")

    def distributed(self):
        self._distributed = True
        self._broadcast_initialized_params_for_each_quantizer()

    def _get_algo_config(self) -> dict:
        return extract_algo_specific_config(self.config, "quantization")

    def _broadcast_initialized_params_for_each_quantizer(self):
        # NOTE: Order of quantization modules must be the same on GPUs to correctly broadcast num_bits
        sorted_quantizers = OrderedDict(sorted(self.all_quantizations.items(), key=lambda x: str(x[0])))
        for quantizer in sorted_quantizers.values():
            quantizer.broadcast_initialized_params()

    def _do_runtime_range_init(self, range_init_params: PTRangeInitParams):
        modules_to_init = OrderedDict()
        for wq_id, wq_info in self.weight_quantizers.items():
            group = QuantizerGroup.WEIGHTS
            init_config = range_init_params.get_init_config_for_scope_and_group(wq_id, group)
            is_weights = True
            modules_to_init[str(wq_id)] = (
                wq_info.quantizer_module_ref,
                init_config,
                is_weights,
                self._quantizers_input_shapes[wq_id],
            )

        for aq_id, aq_info in self.non_weight_quantizers.items():
            group = QuantizerGroup.ACTIVATIONS
            init_config = range_init_params.get_init_config_for_scope_and_group(aq_id, group)
            is_weights = False
            modules_to_init[str(aq_id)] = (
                aq_info.quantizer_module_ref,
                init_config,
                is_weights,
                self._quantizers_input_shapes[aq_id],
            )

        # NOTE: Order of modules must be the same to correctly broadcast parameters (e.g. input_low
        # and input_range)
        modules_to_init = OrderedDict(sorted(modules_to_init.items()))
        self.modules_to_range_init = modules_to_init
        runner = DataLoaderRangeInitializeRunner(self._model, modules_to_init, range_init_params.device)

        quantizers = [module for module, config, is_weights, input_shape in modules_to_init.values()]
        quantizers_switcher = QuantizersSwitcher(quantizers)
        # bypass quantization to collect statistics from floating point model
        quantizers_switcher.disable_quantizers()
        with training_mode_switcher(self._model, is_training=False):
            # Statistics should be collected in eval mode because the model in train mode may behave differently
            runner.run(range_init_params.init_range_data_loader, range_init_params.get_max_num_init_steps())
        quantizers_switcher.enable_quantizers()

        self._model.nncf.rebuild_graph()

    def compression_stage(self) -> CompressionStage:
        if self.is_staged_scheduler:
            return self.scheduler.compression_stage()
        return CompressionStage.FULLY_COMPRESSED

    def init_precision(
        self,
        precision_init_type: str,
        precision_init_params: BasePrecisionInitParams,
        precision_constraints: HardwareQuantizationConstraints,
    ) -> SingleConfigQuantizerSetup:
        """
        Precision initialization happens based on an measure of layer sensitivity to perturbations. The measure is
        calculated by average Hessian trace estimation for each layer using Hutchinson algorithm.
        """
        init_impl = PrecisionInitializerFactory.create(precision_init_type)
        initializer = init_impl(self, precision_init_params, precision_constraints)
        nncf_logger.info("Initializing quantizer precisions...")
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
                nncf_logger.error(
                    "Requested a quantization controller to do range initialization without "
                    "`range_init_params` function parameter supplied, but the build time range "
                    "initialization was not supplied with params as well. "
                    "Range initialization will not be done."
                )
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
        if not quickly_collected_only and is_debug():
            stats = MemoryConsumptionStatisticsCollector(
                self.model, self.weight_quantizers, self.non_weight_quantizers
            ).collect()
            nncf_logger.debug(stats.to_str())

            stats = ShareEdgesQuantizedDataPathStatisticsCollector(self.model, self, self._target_device).collect()
            nncf_logger.debug(stats.to_str())

        collector = PTQuantizationStatisticsCollector(
            self.weight_quantizers, self.non_weight_quantizers, self._build_time_metric_info
        )
        stats = collector.collect()

        nncf_stats = NNCFStatistics()
        nncf_stats.register("quantization", stats)
        return nncf_stats

    def strip_model(
        self, model: NNCFNetwork, do_copy: bool = False, strip_format: StripFormat = StripFormat.NATIVE
    ) -> NNCFNetwork:
        if do_copy:
            model = copy_model(model)
        model = strip_quantized_model(model, strip_format)
        return model


class ExperimentalQuantizationBuilder(QuantizationBuilder):
    def __init__(
        self,
        quantizer_setup: MultiConfigQuantizerSetup,
        initial_quantizer_setup: SingleConfigQuantizerSetup,
        tensor_stats_for_all_setup_variations: dict[PTTargetPoint, dict[ReductionAxes, TensorStatistic]],
        hw_config: HWConfig = None,
    ):
        should_init = bool(tensor_stats_for_all_setup_variations)
        super().__init__(NNCFConfig(), should_init=should_init)
        self._initial_quantizer_setup = initial_quantizer_setup
        self._quantizer_setup = quantizer_setup
        self._tensor_stats = tensor_stats_for_all_setup_variations
        self._should_setup_adjust_pad_ops = False
        self.hw_config = hw_config

    def _handle_frozen_layers(self, target_model: NNCFNetwork):
        pass

    def _get_single_config_quantizer_setup(self, target_model) -> SingleConfigQuantizerSetup:
        return self._initial_quantizer_setup

    def _get_statistics_for_final_range_init(
        self, target_model: NNCFNetwork, quantizer_setup: QuantizerSetupBase, range_init_params: PTRangeInitParams
    ) -> dict[PTTargetPoint, dict[ReductionAxes, TensorStatistic]]:
        return self._tensor_stats

    def _build_controller(self, model: NNCFNetwork) -> "ExperimentalQuantizationController":
        groups_of_adjacent_quantizers = GroupsOfAdjacentQuantizers()
        all_quantizations: dict[QuantizerId, BaseQuantizer] = {}
        all_quantizations.update({k: v.quantizer_module_ref for k, v in self._weight_quantizers.items()})
        all_quantizations.update({k: v.quantizer_module_ref for k, v in self._non_weight_quantizers.items()})

        groups_of_adjacent_quantizers.parse_from_quantizer_setup(
            all_quantizations, self._pt_quantizer_setup, self._setup_to_module_id_translation_dict
        )

        build_time_metric_infos = QuantizationShareBuildTimeInfo(
            len(self._non_weight_quantizers), len(self._weight_quantizers)
        )

        return ExperimentalQuantizationController(
            model,
            self._weight_quantizers,
            self._non_weight_quantizers,
            groups_of_adjacent_quantizers,
            self._quantizers_input_shapes,
            self._quantizer_setup,
            self._initial_quantizer_setup,
            self._setup_to_module_id_translation_dict,
            self._tensor_stats,
            build_time_metric_infos,
            self._should_setup_adjust_pad_ops,
            self.hw_config,
        )

    def initialize(self, model: NNCFNetwork) -> None:
        pass

    def _get_algo_specific_config_section(self) -> dict:
        return {}

    def _parse_range_init_params(self) -> Optional[PTRangeInitParams]:
        return None

    def _get_compression_lr_multiplier(self) -> Optional[float]:
        return None


class ExperimentalQuantizationController(QuantizationController):
    def __init__(
        self,
        target_model: NNCFNetwork,
        weight_quantizers: dict[WeightQuantizerId, WeightQuantizerInfo],
        non_weight_quantizers: dict[NonWeightQuantizerId, NonWeightQuantizerInfo],
        groups_of_adjacent_quantizers: GroupsOfAdjacentQuantizers,
        quantizers_input_shapes: dict[QuantizerId, tuple[int]],
        quantizer_setup: MultiConfigQuantizerSetup,
        initial_quantizer_setup: SingleConfigQuantizerSetup,
        setup_to_module_id_translation_dict: dict[QuantizationPointId, QuantizerId],
        tensor_stats: dict[PTTargetPoint, dict[ReductionAxes, TensorStatistic]],
        build_time_metric_info: QuantizationShareBuildTimeInfo,
        should_setup_adjust_pad_ops=False,
        hw_config: HWConfig = None,
    ):
        super().__init__(
            target_model,
            NNCFConfig(),
            debug_interface=None,
            weight_quantizers=weight_quantizers,
            non_weight_quantizers=non_weight_quantizers,
            groups_of_adjacent_quantizers=groups_of_adjacent_quantizers,
            quantizers_input_shapes=quantizers_input_shapes,
            build_time_metric_info=build_time_metric_info,
        )
        self._target_model_ref = target_model
        self._should_setup_adjust_pad_ops = should_setup_adjust_pad_ops
        self._quantizer_setup = quantizer_setup
        self._initial_quantizer_setup = initial_quantizer_setup
        self._tensor_stats = tensor_stats
        self.setup_to_module_id_translation_dict = setup_to_module_id_translation_dict
        self.module_id_to_qp_id_translation_dict: dict[QuantizerId, set[QuantizationPointId]] = {}
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
            msg = "The new setup is inconsistent with the original parameter space!"
            raise ValueError(msg)
        for qp_id, qp in quantizer_setup.quantization_points.items():
            current_qconfig = current_setup.quantization_points[qp_id].qconfig
            new_qconfig = quantizer_setup.quantization_points[qp_id].qconfig
            new_padding_adjust_applicable = CalculatePaddingAdjustment.is_config_applicable(new_qconfig)
            current_padding_adjust_applicable = CalculatePaddingAdjustment.is_config_applicable(current_qconfig)
            need_padding_regeneration = (
                self._should_setup_adjust_pad_ops
                and qp.is_activation_quantization_point()
                and new_padding_adjust_applicable != current_padding_adjust_applicable
            )
            if (
                current_qconfig.per_channel != new_qconfig.per_channel
                or (
                    new_qconfig.signedness_to_force is not None
                    and current_qconfig.signedness_to_force != new_qconfig.signedness_to_force
                )
                or current_qconfig.mode != new_qconfig.mode
                or need_padding_regeneration
            ):
                return True
        return False

    def apply_new_quantizer_setup(
        self, quantizer_setup: SingleConfigQuantizerSetup
    ) -> tuple["ExperimentalQuantizationController", NNCFNetwork]:
        if not self.is_new_setup_requires_regeneration(quantizer_setup):
            for qp_id, qp in quantizer_setup.quantization_points.items():
                quant_module_id = self.setup_to_module_id_translation_dict[qp_id]
                quant_module = self.all_quantizations[quant_module_id]
                quant_module.num_bits = qp.qconfig.num_bits
            return self, self._target_model_ref
        new_model = self._target_model_ref.nncf.get_clean_shallow_copy()
        new_builder = ExperimentalQuantizationBuilder(
            self._quantizer_setup,
            initial_quantizer_setup=quantizer_setup,
            tensor_stats_for_all_setup_variations=self._tensor_stats,
            hw_config=self.hw_config,
        )
        new_builder.apply_to(new_model)
        new_ctrl = new_builder.build_controller(new_model)
        return new_ctrl, new_model

    def _get_algo_config(self) -> dict:
        return {}
