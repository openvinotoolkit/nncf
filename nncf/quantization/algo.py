"""
 Copyright (c) 2019-2020 Intel Corporation
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

# pylint:disable=too-many-lines
import functools
from collections import OrderedDict, namedtuple
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable

import networkx as nx
import numpy as np
import operator
import shutil
import torch
from nncf.layers import NNCFEmbedding, NNCFEmbeddingBag
from torch import nn

from nncf.algo_selector import COMPRESSION_ALGORITHMS
from nncf.compression_method_api import CompressionAlgorithmBuilder, CompressionAlgorithmController, CompressionLevel
from nncf.debug import is_debug, DebugInterface, CallCountTracker
from nncf.dynamic_graph.context import OperatorInput, TracingContext, Scope
from nncf.dynamic_graph.function_input_quantization import FUNCTIONS_TO_QUANTIZE
from nncf.dynamic_graph.graph import NNCFNode, InputAgnosticOperationExecutionContext
from nncf.dynamic_graph.graph import NNCFNodeExpression as N, NNCFGraph
from nncf.dynamic_graph.patch_pytorch import get_arg_positions_to_quantize
from nncf.dynamic_graph.input_wrapping import MODEL_INPUT_OP_NAME
from nncf.dynamic_graph.transform_graph import is_nncf_module
from nncf.hw_config import HWConfig
from nncf.initialization import DataLoaderRangeInitializeRunner
from nncf.module_operations import UpdateWeight, UpdateInputs
from nncf.nncf_logger import logger as nncf_logger
from nncf.nncf_network import NNCFNetwork, CompressionModuleType, InsertionCommand, OperationPriority, \
    InsertionPoint, InsertionType, InsertionPointGraph, InsertionPointGraphNodeType
from nncf.quantization.hw_precision_constraints import HWPrecisionConstraints
from nncf.quantization.init_precision import PrecisionInitializerFactory
from nncf.quantization.layers import QUANTIZATION_MODULES, QuantizationMode, QuantizerConfig, BaseQuantizer, \
    QuantizerExportMode, QuantizersSwitcher
from nncf.quantization.metrics import NetworkQuantizationShareMetric, MemoryCostMetric, ShareEdgesQuantizedDataPath
from nncf.quantization.quantizer_id import WeightQuantizerId, NonWeightQuantizerId, InputQuantizerId, \
    FunctionQuantizerId
from nncf.quantization.quantizer_propagation import QuantizerPropagationSolver, QuantizerPropagationStateGraph, \
    QuantizersBetweenQuantizableLayers, QuantizerInsertionInfo
from nncf.quantization.schedulers import QUANTIZATION_SCHEDULERS
from nncf.structures import QuantizationPrecisionInitArgs, QuantizationRangeInitArgs, AutoQPrecisionInitArgs
from nncf.utils import get_all_modules_by_type, in_scope_list, is_main_process, should_consider_scope
from nncf.utils import get_state_dict_names_with_modules


class QuantizerSetupType(Enum):
    PATTERN_BASED = "pattern_based"
    PROPAGATION_BASED = "propagation_based"

    @staticmethod
    def from_str(quantizer_setup_type: str) -> 'QuantizerSetupType':
        if quantizer_setup_type == QuantizerSetupType.PATTERN_BASED.value:
            return QuantizerSetupType.PATTERN_BASED
        if quantizer_setup_type == QuantizerSetupType.PROPAGATION_BASED.value:
            return QuantizerSetupType.PROPAGATION_BASED
        raise RuntimeError("Unknown quantizer setup type. Please select 'pattern_based' or 'propagation_based'.")


class QuantizationConstraints:
    REF_QCONF_OBJ = QuantizerConfig()

    def __init__(self, **kwargs):
        """Use attribute names of QuantizerConfig as arguments
        to set up constraints.
        E.g. QuantizationConstraint(bits=8, per_channel=True) will set up
        a constraint that corresponds to all 8-bit per-channel quantizers, either
        symmetric or asymmetric, either signed or unsigned."""

        for attr_name in kwargs:
            if not hasattr(QuantizationConstraints.REF_QCONF_OBJ, attr_name):
                raise RuntimeError("Invalid constraint - QuantizerConfig has no attribute '{}'".format(attr_name))
        self.qconf_attr_vs_constraint_dict = kwargs

    def apply_constraints_to(self, qconfig: QuantizerConfig) -> QuantizerConfig:
        for attr_name, constraint in self.qconf_attr_vs_constraint_dict.items():
            if constraint is not None:
                setattr(qconfig, attr_name, constraint)
        return qconfig

    def is_config_compatible(self, qconfig: QuantizerConfig) -> bool:
        is_compatible = True
        for attr_name, constraint in self.qconf_attr_vs_constraint_dict.items():
            if attr_name == 'logarithm_scale':
                continue   # Scale storage type is internal and should not affect HW config matching
            if constraint is not None:
                qconf_attr_value = getattr(qconfig, attr_name)
                if qconf_attr_value != constraint:
                    is_compatible = False
        return is_compatible


class QuantizerGroup(Enum):
    ACTIVATIONS = "activations"
    WEIGHTS = "weights"


PotentialQuantizedModule = namedtuple('PotentialQuantizedModule', 'module module_scope qconfig_list')


class NonWeightQuantizerInfo:
    def __init__(self, quantizer_module_ref: BaseQuantizer,
                 affected_ia_op_exec_contexts: List[InputAgnosticOperationExecutionContext],
                 quantizers_between_quantizable_layers: QuantizersBetweenQuantizableLayers = None):
        self.quantizer_module_ref = quantizer_module_ref
        self.affected_ia_op_exec_contexts = affected_ia_op_exec_contexts
        self.quantizers_between_quantizable_layers = quantizers_between_quantizable_layers


@COMPRESSION_ALGORITHMS.register('quantization')
class QuantizationBuilder(CompressionAlgorithmBuilder):
    DEFAULT_QUANTIZER_CONFIG = QuantizerConfig(bits=8,
                                               mode=QuantizationMode.SYMMETRIC,
                                               signedness_to_force=None,
                                               per_channel=False)

    def __init__(self, config, should_init: bool = True):
        super().__init__(config, should_init)

        self.quantize_inputs = self.config.get('quantize_inputs', True)
        self.quantize_outputs = self.config.get('quantize_outputs', False)
        self.disable_function_quantization_hooks = self.config.get('disable_function_quantization_hooks', False)
        self._debug_interface = QuantizationDebugInterface() if is_debug() else None

        self._quantized_weight_modules_registry = OrderedDict()
        self._quantized_inputs_modules_registry = OrderedDict()
        self._weight_quantizers = OrderedDict()  # Quantizers applied via UpdateWeights
        self._non_weight_quantizers = OrderedDict()  # All the other quantizers
        self._processed_function_quantizers = set()
        self._processed_input_agnostic_op_exec_contexts = set()

        self.global_quantizer_constraints = {}  # type: Dict[QuantizerGroup, QuantizationConstraints]
        self._ignored_scopes_per_group = {}  # type: Dict[QuantizerGroup, List[str]]
        self._target_scopes_per_group = {}  # type: Dict[QuantizerGroup, List[str]]

        for quantizer_group in QuantizerGroup:
            self._parse_group_params(self.config, quantizer_group)

        self.quantizer_setup_type = self.config.get('quantizer_setup_type')
        self.quantizable_subgraph_patterns = self.config.get('quantizable_subgraph_patterns', None)
        self.hw_config = None
        hw_config_type = self.config.get("hw_config_type")
        is_hw_config_enabled = hw_config_type is not None
        self._hw_precision_constraints = HWPrecisionConstraints(is_hw_config_enabled)
        if is_hw_config_enabled:
            hw_config_path = HWConfig.get_path_to_hw_config(hw_config_type)
            self.hw_config = HWConfig.from_json(hw_config_path)
        self.eval_ops_exec_ctx = []

    def _parse_group_params(self, quant_config: 'NNCFConfig', quantizer_group: QuantizerGroup):
        group_name = quantizer_group.value
        params_dict = quant_config.get(group_name, {})
        self.global_quantizer_constraints[quantizer_group] = QuantizationConstraints(
            bits=params_dict.get('bits'),
            mode=params_dict.get('mode'),
            signedness_to_force=params_dict.get('signed'),
            per_channel=params_dict.get('per_channel'),
            logarithm_scale=params_dict.get('logarithm_scale'),
        )
        self._ignored_scopes_per_group[quantizer_group] = params_dict.get('ignored_scopes')
        self._target_scopes_per_group[quantizer_group] = params_dict.get('target_scopes')

    def apply_to(self, target_model: NNCFNetwork) -> NNCFNetwork:
        insertion_commands = self._quantize_weights(target_model) + self._quantize_activations(target_model)
        if self.quantize_inputs and self.quantizer_setup_type is not QuantizerSetupType.PROPAGATION_BASED:
            insertion_commands += self._quantize_inputs(target_model, insertion_commands)

        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]

        # At this point the NNCF module quantization modules are not in the target_model yet,
        # therefore it is extended with the corresponding registries tracked during weights/inputs quantizations
        self._all_quantizations = get_state_dict_names_with_modules(target_model, quantization_types)
        self._all_quantizations.update(self._quantized_weight_modules_registry)
        self._all_quantizations.update(self._quantized_inputs_modules_registry)

        for command in insertion_commands:
            target_model.register_insertion_command(command)

        target_model.register_algorithm(self)

        if self._debug_interface is not None:
            target_model.debug_interface.add_interface(self._debug_interface)
        return target_model

    def build_controller(self, target_model: NNCFNetwork) -> CompressionAlgorithmController:
        return QuantizationController(target_model,
                                      self.config,
                                      self.should_init,
                                      self._debug_interface,
                                      self._quantized_weight_modules_registry,
                                      self._quantized_inputs_modules_registry,
                                      self._weight_quantizers,
                                      self._non_weight_quantizers,
                                      self._hw_precision_constraints)

    def __get_default_qconfig(self, constraints: QuantizationConstraints = None):
        qconfig = deepcopy(self.DEFAULT_QUANTIZER_CONFIG)
        if constraints is not None:
            qconfig = constraints.apply_constraints_to(qconfig)
        return qconfig

    def __get_scoped_quantizer_config(self, target_model: NNCFNetwork,
                                      parent_module_scope_str: str, is_weights=False,
                                      input_shape=None) -> QuantizerConfig:
        group = QuantizerGroup.WEIGHTS if is_weights else QuantizerGroup.ACTIVATIONS
        qconfig = self.__get_default_qconfig(constraints=self.global_quantizer_constraints[group])
        qconfig.is_weights = is_weights
        scope_overrides = self.config.get("scope_overrides", {})
        for overridden_scope in scope_overrides.keys():
            if in_scope_list(parent_module_scope_str, overridden_scope):
                config_overrides = scope_overrides[overridden_scope]
                if config_overrides.get("bits") is not None:
                    qconfig.bits = config_overrides["bits"]
                if config_overrides.get("mode") is not None:
                    qconfig.mode = config_overrides["mode"]
                if config_overrides.get("per_channel") is not None:
                    qconfig.per_channel = config_overrides["per_channel"]
                if config_overrides.get("signed") is not None:
                    qconfig.signedness_to_force = config_overrides["signed"]
                if config_overrides.get("logarithm_scale") is not None:
                    qconfig.logarithm_scale = config_overrides["logarithm_scale"]
        if qconfig.per_channel:
            if is_weights:
                module = target_model.get_module_by_scope(Scope.from_str(parent_module_scope_str))
                qconfig.input_shape = module.weight.shape
            elif input_shape is not None:
                qconfig.input_shape = input_shape
            else:
                raise RuntimeError("Unable to use per channel quantization for module {} activations -"
                                   " input shape is unknown".format(parent_module_scope_str))
        return qconfig

    def __create_quantize_module(self, qconfig: QuantizerConfig):
        quantizer_cls = QUANTIZATION_MODULES.get(qconfig.mode)
        return quantizer_cls(qconfig)

    def _make_quantizable_subgraph_pattern(self):
        full_pattern = self._make_default_quantizable_subgraph_pattern()
        if self.quantizable_subgraph_patterns is not None:
            for pattern in self.quantizable_subgraph_patterns:
                if not isinstance(pattern, str):
                    custom_pattern = functools.reduce(operator.add,
                                                      [N(node) for node in pattern])
                else:
                    custom_pattern = N(pattern)
                full_pattern = full_pattern | custom_pattern
        return full_pattern

    def _check_if_ip_graph_nodes_point_to_single_module(self, ip_graph_node_list: List[dict]):
        """Does not access actual modules - only uses the InputAgnosticOperationExecutionContext info."""
        ia_op_exec_contexts_list = []  # type: List[InputAgnosticOperationExecutionContext]
        for ip_graph_op_node in ip_graph_node_list:
            nncf_node = ip_graph_op_node[InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR]
            ia_op_exec_context = nncf_node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].input_agnostic
            ia_op_exec_contexts_list.append(ia_op_exec_context)

        contexts_correspond_to_single_module = True
        first_op_context = ia_op_exec_contexts_list[0]
        for other_op_context in ia_op_exec_contexts_list:
            if other_op_context.scope_in_model != first_op_context.scope_in_model or \
                    other_op_context.operator_name != first_op_context.operator_name:
                contexts_correspond_to_single_module = False
                break

        if not contexts_correspond_to_single_module:
            raise RuntimeError("NNCF module has more than 1 associated graph operation node corresponding"
                               "to different module hierarchy locations - cannot make sure that weight "
                               "quantization will be correct")

    def get_potential_quantized_modules(self, target_model: NNCFNetwork) -> List[PotentialQuantizedModule]:
        modules = target_model.get_nncf_modules()
        insertion_point_graph = target_model.get_insertion_point_graph()
        quantized_modules_with_potential_qconfig = []
        default_qconfig_list = [self.__get_default_qconfig(
            constraints=self.global_quantizer_constraints[QuantizerGroup.WEIGHTS])]
        if self.hw_config is not None:
            meta_vs_qconfig_map = self.hw_config.get_metatype_vs_quantizer_configs_map(for_weights=True)
        for module_scope, module in modules.items():
            if not self._should_consider_scope_for_group(str(module_scope), QuantizerGroup.WEIGHTS):
                nncf_logger.info("Ignored adding Weight quantizer in scope: {}".format(module_scope))
                continue
            if self.hw_config is None:
                qconfig = self.__get_scoped_quantizer_config(target_model, str(module_scope), is_weights=True)
                qconfig_list = [qconfig]
            else:
                associated_ops = insertion_point_graph.get_op_nodes_in_scope(module_scope)
                if not associated_ops:
                    raise RuntimeError(
                        "Could not find a patched operation corresponding to NNCF module scope {}".format(
                            str(module_scope)))

                if len(associated_ops) > 1:
                    self._check_if_ip_graph_nodes_point_to_single_module(associated_ops)
                graph_operation = associated_ops[0]
                metatype = graph_operation[InsertionPointGraph.OPERATOR_METATYPE_NODE_ATTR]
                qconfig_list = meta_vs_qconfig_map[metatype]
                if HWConfig.is_wildcard_quantization(qconfig_list):  # Empty list = wildcard quantization
                    qconfig_list = default_qconfig_list

            if qconfig_list is not None:
                quantized_modules_with_potential_qconfig.append(PotentialQuantizedModule(module, module_scope,
                                                                                         qconfig_list))
        return quantized_modules_with_potential_qconfig

    def _quantize_weights(self, target_model: NNCFNetwork) -> List[InsertionCommand]:
        device = next(target_model.parameters()).device
        insertion_commands = []
        quantized_modules_with_potential_qconfig = \
            self.get_potential_quantized_modules(target_model)
        for module, module_scope, qconfig_list in quantized_modules_with_potential_qconfig:
            self._quantized_weight_modules_registry[str(module_scope)] = module
            nncf_logger.info("Adding signed Weight quantizer in scope: {}".format(module_scope))

            if self.hw_config is not None:
                try:
                    qconfig_overrides = None
                    if in_scope_list(str(module_scope), self.config.get("scope_overrides", {})):
                        qconfig_overrides = self.__get_scoped_quantizer_config(target_model,
                                                                               str(module_scope),
                                                                               is_weights=True)
                    qconfig = self._select_final_qconfig(qconfig_list,
                                                         self.global_quantizer_constraints[QuantizerGroup.WEIGHTS],
                                                         qconfig_overrides)
                except RuntimeError:
                    err_msg = "Quantization parameter constraints specified in NNCF config are incompatible with HW "
                    err_msg += "capabilities as specified in HW config type '{}'. ".format(self.hw_config.target_device)
                    err_msg += "First conflicting quantizer location: {}".format(str(module_scope))
                    raise RuntimeError(err_msg)
            else:
                assert len(
                    qconfig_list) == 1, "Non-HW config scenarios should produce single quantizer configs for each " \
                                        "weight module!"
                qconfig = qconfig_list[0]

            quantizer_id = WeightQuantizerId(module_scope)
            self._hw_precision_constraints.add(quantizer_id, qconfig_list)
            qconfig.input_shape = module.weight.shape
            quantizer = self.__create_quantize_module(qconfig)
            op = UpdateWeight(quantizer).to(device)
            # TODO: separate insertion point semantic for weights and activations
            insertion_commands.append(InsertionCommand(InsertionPoint(
                InputAgnosticOperationExecutionContext('', module_scope, 0),
                InsertionType.NNCF_MODULE_PRE_OP), op, OperationPriority.QUANTIZATION_PRIORITY))
            self._weight_quantizers[quantizer_id] = quantizer
        return insertion_commands

    class ActivationQuantizationHook:
        """Cannot simply register the quantizer module as a callable hook, since we need to call
        a thread-local version of the quantizer module during base module execution."""

        def __init__(self, context: TracingContext, quantizer_storage_key: str,
                     debug_interface: 'QuantizationDebugInterface' = None):
            self.compressed_context = context
            self.quantizer_storage_key = quantizer_storage_key
            self.debug_interface = debug_interface

        def __call__(self, *args, **kwargs):
            if self.debug_interface is not None:
                self.debug_interface.register_activation_quantize_call(str(self.quantizer_storage_key))
            replica = self.compressed_context.base_module_thread_local_replica
            return replica.activation_quantizers[self.quantizer_storage_key](*args, **kwargs)

    def _quantize_activations(self, target_model: NNCFNetwork) -> List[InsertionCommand]:
        target_model.register_compression_module_type(CompressionModuleType.ACTIVATION_QUANTIZER)

        if self.quantizer_setup_type == QuantizerSetupType.PATTERN_BASED:
            insertion_commands = self._quantize_post_pattern_activations(target_model)
        elif self.quantizer_setup_type == QuantizerSetupType.PROPAGATION_BASED:
            insertion_point_graph = target_model.get_insertion_point_graph()
            if self._debug_interface:
                self._debug_interface.visualize_insertion_point_graph(insertion_point_graph)
            prop_graph_solver = QuantizerPropagationSolver(ignored_scopes=self.ignored_scopes,
                                                           debug_interface=self._debug_interface,
                                                           hw_config=self.hw_config,
                                                           default_qconfig_list=[self.__get_default_qconfig(
                                                               constraints=self.global_quantizer_constraints[
                                                                   QuantizerGroup.ACTIVATIONS])],
                                                           input_infos=target_model.get_input_infos())
            merged_ip_graph = insertion_point_graph.get_ip_graph_with_merged_hw_optimized_operations(self.hw_config)
            insertion_data = prop_graph_solver.run_on_ip_graph(merged_ip_graph)
            insertion_commands = []

            original_nncf_graph = target_model.get_original_graph()
            for insertion_info, quantizer_config_list in insertion_data.items():
                ia_op_exec_context = insertion_info.op_exec_context.input_agnostic
                operator_scope_str = str(ia_op_exec_context)
                if not self.quantize_inputs and ia_op_exec_context.operator_name == MODEL_INPUT_OP_NAME:
                    continue

                # Tailored for post-hook quantization and first output quantization only
                quantizer_input_shape = original_nncf_graph.get_output_shapes_for_ia_op_exec_context(
                    ia_op_exec_context)[0]
                try:
                    qconfig_overrides = None
                    if in_scope_list(operator_scope_str, self.config.get("scope_overrides", {})):
                        qconfig_overrides = self.__get_scoped_quantizer_config(target_model,
                                                                               operator_scope_str,
                                                                               is_weights=True)
                    quantizer_config = self._select_final_qconfig(quantizer_config_list,
                                                                  self.global_quantizer_constraints[
                                                                      QuantizerGroup.ACTIVATIONS],
                                                                  qconfig_overrides)
                except RuntimeError:
                    err_msg = "Quantization parameter constraints specified in NNCF config are incompatible with HW "
                    err_msg += "capabilities as specified in HW config type '{}'. ".format(self.hw_config.target_device)
                    err_msg += "First conflicting quantizer location: "
                    err_msg += str(ia_op_exec_context)
                    raise RuntimeError(err_msg)

                quantizer_config.input_shape = quantizer_input_shape
                quantizer_id = NonWeightQuantizerId(ia_op_exec_context)
                self._hw_precision_constraints.add(quantizer_id, quantizer_config_list)
                insertion_commands += self._add_single_activation_quantizer(target_model, insertion_info,
                                                                            quantizer_config)
        else:
            raise RuntimeError("Invalid quantizer setup type!")

        if not self.disable_function_quantization_hooks:
            insertion_commands += self._quantize_free_function_inputs(target_model)
        return insertion_commands

    def _select_final_qconfig(self, quantizer_config_list: List[QuantizerConfig],
                              constraints: QuantizationConstraints, qconfig_overrides=None) -> QuantizerConfig:
        assert quantizer_config_list is not None

        if self.hw_config is None and qconfig_overrides is not None:
            return qconfig_overrides

        if HWConfig.is_wildcard_quantization(quantizer_config_list):
            # Set a default, most basic quantization config in case wildcard propagating quantizer did
            # not merge to align with other tools
            return self.__get_default_qconfig()

        constrained_quantizer_config_list = list(filter(
            constraints.is_config_compatible,
            quantizer_config_list
        ))

        if qconfig_overrides is not None:
            constrained_quantizer_config_list =\
                 [qconfig for qconfig in constrained_quantizer_config_list if qconfig_overrides == qconfig]

        # TODO: Make the logic more flexible when the flag "warning as error" is implemented.
        # It means that the qconfig from overrides must be selected as final config
        # even if it is not valid in hw-config.
        if not constrained_quantizer_config_list:
            raise RuntimeError()

        # Quantizer config list entries should arrive in the same order as they are listed
        # in the HW config, where they are sorted by descending order of priority
        return constraints.apply_constraints_to(constrained_quantizer_config_list[0])

    def _quantize_post_pattern_activations(self, target_model: NNCFNetwork) -> List[InsertionCommand]:
        pattern = self._make_quantizable_subgraph_pattern()
        target_insertion_infos = target_model.get_post_pattern_insertion_points(pattern)
        insertion_commands = []

        act_config = self.config.get('activations', {})
        if 'linked_quantizer_scopes' in act_config:
            linked_scopes_groups_list = act_config['linked_quantizer_scopes']
            target_insertion_infos = self.coalesce_insertion_infos(target_insertion_infos, linked_scopes_groups_list)

        for insertion_info in target_insertion_infos:
            ia_op_exec_context = insertion_info.op_exec_context.input_agnostic
            operator_scope_str = str(ia_op_exec_context)

            if not self.quantize_outputs and insertion_info.is_output:
                nncf_logger.info("Ignored adding Activation Quantize "
                                 "in scope (output scope, quantize_outputs=False): {}".format(operator_scope_str))
                continue
            if not self._should_consider_scope_for_group(operator_scope_str, QuantizerGroup.ACTIVATIONS):
                nncf_logger.info("Ignored adding Activation quantizer in scope: {}".format(operator_scope_str))
                continue

            qconfig = self.__get_scoped_quantizer_config(target_model, operator_scope_str,
                                                         is_weights=False,
                                                         input_shape=insertion_info.shape_to_operate_on)
            quantizer_insertion_info = QuantizerInsertionInfo.from_insertion_info(insertion_info)
            insertion_commands += self._add_single_activation_quantizer(target_model, quantizer_insertion_info, qconfig)

        # NOTE: Order of activations must be the same to correctly broadcast parameters (e.g. scales) in distributed
        # mode (see call of `_dist_broadcast_coalesced` in torch/nn/parallel/distributed.py for more details)
        # pylint: disable=protected-access
        target_model.sort_compression_modules(CompressionModuleType.ACTIVATION_QUANTIZER)
        return insertion_commands

    def _add_single_activation_quantizer(self, target_model: NNCFNetwork,
                                         insertion_info: QuantizerInsertionInfo,
                                         quantizer_config: QuantizerConfig) -> List[InsertionCommand]:
        """Will return one or more insertion commands - depending on whether insertion_info specifies
        a single input agnostic operation execution context or there are linked contexts along with it."""
        ia_op_exec_context = insertion_info.op_exec_context.input_agnostic
        operator_scope_str = str(ia_op_exec_context)
        device = next(target_model.parameters()).device
        quantizer = self.__create_quantize_module(quantizer_config).to(device)
        affected_ia_op_exec_contexts = [ia_op_exec_context] + [x.input_agnostic for x in
                                                               insertion_info.linked_op_exec_contexts]

        # linked_op_exec_contexts will determine quantization points in graph that have to share
        # a quantization module, e.g. for scales unification
        serialized_context_list = [str(x) for x in affected_ia_op_exec_contexts]
        quantizer_storage_key = ";".join([str(x) for x in serialized_context_list])

        assert quantizer_storage_key not in target_model.get_compression_modules_by_type(
            CompressionModuleType.ACTIVATION_QUANTIZER)

        target_model.add_compression_module(quantizer_storage_key, quantizer,
                                            CompressionModuleType.ACTIVATION_QUANTIZER)

        quantizer_id = NonWeightQuantizerId(ia_op_exec_context)

        if len(affected_ia_op_exec_contexts) > 1:
            nncf_logger.info(
                "Processing linked activation quantizer group:\n {}\n".format("\n".join(serialized_context_list)))

        self._non_weight_quantizers[quantizer_id] = NonWeightQuantizerInfo(
            quantizer, affected_ia_op_exec_contexts, insertion_info.quantizers_between_quantizable_layers)

        insertion_commands = []
        for curr_ia_op_exec_context in affected_ia_op_exec_contexts:
            if curr_ia_op_exec_context in self._processed_input_agnostic_op_exec_contexts:
                raise RuntimeError(
                    "Ambiguous call to {fn} with call order {co} in current scope. "
                    "Cannot insert quantization hooks "
                    "automatically!".format(fn=ia_op_exec_context.operator_name, co=ia_op_exec_context.call_order)
                )
            self._processed_input_agnostic_op_exec_contexts.add(curr_ia_op_exec_context)

            nncf_logger.info("Adding {}{} Activation Quantize in scope: {}".format(
                "signed" if quantizer.signed else "unsigned",
                " logarithm_scale" if quantizer.is_using_log_scale_storage else "",
                operator_scope_str
            ))

            # Hooks will be identical for each affected ia_op_exec_context - will call one and the
            # same quantizer
            hook = self.ActivationQuantizationHook(target_model.get_tracing_context(),
                                                   quantizer_storage_key,
                                                   self._debug_interface)

            insertion_commands.append(InsertionCommand(InsertionPoint(curr_ia_op_exec_context,
                                                                      InsertionType.OPERATOR_POST_HOOK),
                                                       hook,
                                                       OperationPriority.QUANTIZATION_PRIORITY))
        return insertion_commands

    def _quantize_inputs(self, target_model: NNCFNetwork,
                         prev_weight_and_activation_quantizer_insertion_commands: List[InsertionCommand]) -> \
        List[InsertionCommand]:
        device = next(target_model.parameters()).device
        graph_roots = target_model.get_original_graph().get_input_nodes()

        # Have to handle the situation when the input node of the network is an NNCF module -
        # to quantize inputs in this case we will have to use UpdateInputs module pre-op,

        # Traverse down starting from graph roots to search for the first node which belongs to a NNCF module
        # and has no UpdateInputs pre-op

        def traverse_function(node: NNCFNode, output) -> Tuple[bool, List[NNCFNode]]:
            module = target_model.get_module_by_scope(node.op_exec_context.scope_in_model)
            if is_nncf_module(module):
                if isinstance(module, (NNCFEmbedding, NNCFEmbeddingBag)):
                    # Embeddings have integer input and their quantization is rather controlled
                    # by their weights.
                    return True, output
                current_node_scope = node.op_exec_context.scope_in_model
                module_op_insertion_commands = []
                for comm in prev_weight_and_activation_quantizer_insertion_commands:
                    if current_node_scope in comm.insertion_point.ia_op_exec_context.scope_in_model:
                        module_op_insertion_commands.append(comm)
                pre_op_insertion_commands = filter(
                    lambda comm: comm.insertion_point.insertion_type == InsertionType.NNCF_MODULE_PRE_OP,
                    module_op_insertion_commands)
                update_inputs_count = sum(1 for comm in pre_op_insertion_commands if isinstance(comm.fn, UpdateInputs))
                if update_inputs_count == 0:
                    output.append(node)
                    return True, output
            else:
                current_node_ia_op_exec_context = node.op_exec_context.input_agnostic
                op_hook_insertion_commands = []
                for comm in prev_weight_and_activation_quantizer_insertion_commands:
                    if current_node_ia_op_exec_context == comm.insertion_point.ia_op_exec_context:
                        op_hook_insertion_commands.append(comm)
                if op_hook_insertion_commands:
                    return True, output

            return False, output

        nncf_module_input_nodes = set()
        for node in graph_roots:
            scope_str = str(node.op_exec_context.scope_in_model)
            if self._should_consider_scope_for_group(scope_str, QuantizerGroup.ACTIVATIONS):
                nncf_module_input_nodes.update(
                    target_model.get_original_graph().traverse_graph(node, traverse_function))

        insertion_commands = []
        nncf_scope_module_dict = target_model.get_nncf_modules()
        for module_input_node in nncf_module_input_nodes:
            op_scope = module_input_node.op_exec_context.input_agnostic.scope_in_model
            module = None
            scope = None
            for nncf_scope, nncf_module in nncf_scope_module_dict.items():
                if op_scope in nncf_scope:
                    module = nncf_module
                    scope = nncf_scope
                    break

            self._quantized_inputs_modules_registry[str(scope)] = module

            # Only use the shape of the 0-th input info specified in config. TODO: fix this
            input_shape = target_model.input_infos[0].shape if target_model.input_infos is not None else None
            qconfig = self.__get_scoped_quantizer_config(target_model, str(scope), is_weights=False,
                                                         input_shape=input_shape)
            quantizer = self.__create_quantize_module(qconfig)

            nncf_logger.info("Adding {}{} NNCF module input quantizer in scope: {}".format(
                "signed" if quantizer.signed else "unsigned",
                " logarithm_scale" if quantizer.is_using_log_scale_storage else "",
                str(scope)
            ))

            # TODO: separate insertion point semantic for weights and activations
            insertion_commands.append(
                InsertionCommand(InsertionPoint(InputAgnosticOperationExecutionContext("", scope, 0),
                                                InsertionType.NNCF_MODULE_PRE_OP),
                                 UpdateInputs(quantizer).to(device),
                                 OperationPriority.QUANTIZATION_PRIORITY))
            ia_op_exec_context = module_input_node.op_exec_context.input_agnostic
            quantizer_id = InputQuantizerId(ia_op_exec_context)
            self._hw_precision_constraints.add(quantizer_id, [qconfig])
            self._non_weight_quantizers[quantizer_id] = NonWeightQuantizerInfo(
                quantizer, [ia_op_exec_context], QuantizersBetweenQuantizableLayers())

        return insertion_commands

    def _should_consider_scope_for_group(self, scope_str: str, group: QuantizerGroup) -> bool:
        if self.target_scopes is not None or self._target_scopes_per_group[group] is not None:
            if in_scope_list(scope_str, self.target_scopes):
                return True
            if in_scope_list(scope_str, self._target_scopes_per_group[group]):
                return True

            return False

        if in_scope_list(scope_str, self.ignored_scopes):
            return False
        if in_scope_list(scope_str, self._ignored_scopes_per_group[group]):
            return False

        return True

    class FunctionQuantizationPreHook:
        """Cannot simply register the quantizer module as a callable hook, since we need to call
        a thread-local version of the quantizer module during base module execution."""

        def __init__(self, context: TracingContext, func_in_quant_info: FunctionQuantizerId,
                     debug_interface: 'QuantizationDebugInterface' = None):
            self.compressed_context = context
            self.func_in_quant_info = func_in_quant_info
            self.debug_interface = debug_interface

        def __call__(self, op_inputs: OperatorInput):
            quantizer_dict_key = str(self.func_in_quant_info)
            if self.debug_interface is not None:
                self.debug_interface.register_function_quantizer_call(quantizer_dict_key)
            replica = self.compressed_context.base_module_thread_local_replica
            idx = self.func_in_quant_info.input_arg_idx
            op_inputs.op_args[idx] = replica.function_quantizers[quantizer_dict_key](op_inputs.op_args[idx])
            return op_inputs

    def _quantize_free_function_inputs(self, target_model: NNCFNetwork) -> List[InsertionCommand]:
        device = next(target_model.parameters()).device

        if not FUNCTIONS_TO_QUANTIZE:
            return []
        pattern = N(FUNCTIONS_TO_QUANTIZE[0].name)
        for i in range(1, len(FUNCTIONS_TO_QUANTIZE)):
            pattern |= N(FUNCTIONS_TO_QUANTIZE[i].name)

        target_insertion_infos = target_model.get_post_pattern_insertion_points(pattern,
                                                                                omit_nodes_in_nncf_modules=True)
        insertion_commands = []

        target_model.register_compression_module_type(CompressionModuleType.FUNCTION_QUANTIZER)
        for insertion_info in target_insertion_infos:
            ia_op_exec_context = insertion_info.op_exec_context.input_agnostic
            scope_str = str(ia_op_exec_context.scope_in_model)

            if not self._should_consider_scope_for_group(scope_str, QuantizerGroup.ACTIVATIONS):
                nncf_logger.info("Ignored adding function input quantizer in scope: {}".format(scope_str))
                continue

            function_arg_positions_to_quantize = get_arg_positions_to_quantize(ia_op_exec_context.operator_name)
            assert function_arg_positions_to_quantize is not None, "Function with inputs to be quantized has " \
                                                                   "no info struct registered in " \
                                                                   "QUANTIZED_INPUT_FUNCTIONS!"

            for input_arg_idx in function_arg_positions_to_quantize:
                ip_arg_quant_key = FunctionQuantizerId(ia_op_exec_context, input_arg_idx)

                if ip_arg_quant_key in self._processed_function_quantizers:
                    raise RuntimeError(
                        "Ambiguous call to {fn} with call order {co} and argname {arg} in current scope. "
                        "Cannot insert quantization hooks "
                        "automatically!".format(fn=ia_op_exec_context.operator_name,
                                                co=ia_op_exec_context.call_order,
                                                arg=input_arg_idx)
                    )

                self._processed_function_quantizers.add(ip_arg_quant_key)

                ip_arg_quant_name = str(ip_arg_quant_key)
                assert ip_arg_quant_name not in target_model.get_compression_modules_by_type(
                    CompressionModuleType.FUNCTION_QUANTIZER)
                input_shape = insertion_info.op_exec_context.tensor_metas[0].shape

                qconfig = self.__get_scoped_quantizer_config(target_model, scope_str,
                                                             is_weights=False,
                                                             input_shape=input_shape)
                quantizer_module = self.__create_quantize_module(qconfig).to(device)
                target_model.add_compression_module(ip_arg_quant_name, quantizer_module,
                                                    CompressionModuleType.FUNCTION_QUANTIZER)

                nncf_logger.info("Adding {} Function Quantize: {}".format(
                    "signed" if quantizer_module.signed else
                    "unsigned", ip_arg_quant_name))

                hook = self.FunctionQuantizationPreHook(target_model.get_tracing_context(),
                                                        ip_arg_quant_key,
                                                        self._debug_interface)
                insertion_commands.append(InsertionCommand(InsertionPoint(ia_op_exec_context,
                                                                          InsertionType.OPERATOR_PRE_HOOK),
                                                           hook,
                                                           OperationPriority.QUANTIZATION_PRIORITY))
                self._hw_precision_constraints.add(ip_arg_quant_key, [qconfig])
                self._non_weight_quantizers[ip_arg_quant_key] = NonWeightQuantizerInfo(quantizer_module,
                                                                                       [ia_op_exec_context])
        # NOTE: Order of input quantizers must be the same to correctly broadcast parameters (e.g. scales) in
        # distributed mode (see call of `_dist_broadcast_coalesced` in torch/nn/parallel/distributed.py for more
        # details) pylint: disable=protected-access
        target_model.sort_compression_modules(CompressionModuleType.FUNCTION_QUANTIZER)
        return insertion_commands

    @staticmethod
    def _make_default_quantizable_subgraph_pattern():
        import nncf.dynamic_graph.patterns as p
        pattern = p.LINEAR_OPS | p.ARITHMETIC | p.ANY_BN_ACT_COMBO | \
                  p.LINEAR_OPS + p.ANY_BN_ACT_COMBO | p.ARITHMETIC + p.ANY_BN_ACT_COMBO | p.SINGLE_OPS | p.MATMUL
        return pattern

    @staticmethod
    def coalesce_insertion_infos(target_insertion_infos: List[QuantizerInsertionInfo],
                                 linked_scopes_groups_list: List[List[str]]) -> List[QuantizerInsertionInfo]:
        """Accepts a list of InsertionInfos that each correspond only to one InputAgnosticOperationExecutionContext,
        and merges these according to linked_scope_groups_list so that some or all of the resulting InsertionInfo
        objects have non-empty linked_op_exec_contexts lists.
        Each entry in linked_scope_groups_list must be a valid string representation of a single
        InputAgnosticOperationExecutionContext object."""
        ia_op_exec_context_list = [x.op_exec_context.input_agnostic for x in target_insertion_infos]
        retval = []
        insertion_info_indices_vs_group_id = OrderedDict()

        for group_idx, group_list in enumerate(linked_scopes_groups_list):
            for group_member_scope_str in group_list:
                ia_op_exec_context = InputAgnosticOperationExecutionContext.from_str(group_member_scope_str)
                matching_indices = list(
                    filter(lambda x: ia_op_exec_context_list[x] == ia_op_exec_context,
                           range(len(ia_op_exec_context_list))))
                if len(matching_indices) > 1:
                    raise RuntimeError(
                        "Linked activation quantizer entry {} specifies more than 1 activation quantizer:\n {}".format(
                            group_member_scope_str,
                            "\n".join([str(ia_op_exec_context_list[i]) for i in matching_indices])))
                if len(matching_indices) == 0:
                    raise RuntimeError("No match for linked quantizer entry {} among activation quantizers!".format(
                        group_member_scope_str))

                target_idx = matching_indices[0]
                if target_idx in insertion_info_indices_vs_group_id:
                    raise RuntimeError(
                        "Linked activation quantizer groups {} and {} "
                        "overlap!".format(group_idx,
                                          insertion_info_indices_vs_group_id[target_idx])
                    )
                insertion_info_indices_vs_group_id[target_idx] = group_idx

        for i in range(len(ia_op_exec_context_list)):
            if i not in insertion_info_indices_vs_group_id:
                insertion_info_indices_vs_group_id[i] = None

        group_indices_list = [[] for _ in linked_scopes_groups_list]  # type: List[List[int]]
        for insertion_info_idx, group_idx in insertion_info_indices_vs_group_id.items():
            if group_idx is not None:
                group_indices_list[group_idx].append(insertion_info_idx)
            else:
                retval.append(target_insertion_infos[insertion_info_idx])

        for intra_group_indices in group_indices_list:
            main_info_idx = intra_group_indices[0]
            main_info = target_insertion_infos[main_info_idx]
            new_info = QuantizerInsertionInfo(main_info.op_exec_context,
                                              main_info.is_input,
                                              main_info.is_output,
                                              shape_to_operate_on=main_info.shape_to_operate_on)
            for linked_info_idx in intra_group_indices[1:]:
                new_info.linked_op_exec_contexts.append(target_insertion_infos[linked_info_idx].op_exec_context)
            retval.append(new_info)

        return retval


class QuantizationControllerBase(CompressionAlgorithmController):
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
                 should_init: bool,
                 debug_interface: 'QuantizationDebugInterface',
                 quantized_weight_modules_registry: Dict[Scope, torch.nn.Module],
                 quantized_inputs_modules_registry: Dict[Scope, torch.nn.Module],
                 weight_quantizers: Dict[WeightQuantizerId, torch.nn.Module],
                 non_weight_quantizers: Dict[NonWeightQuantizerId, NonWeightQuantizerInfo],
                 hw_precision_constraints: HWPrecisionConstraints,
                 collect_compression_metrics: bool = True):
        super().__init__(target_model)
        self.debug_interface = debug_interface
        self.quantization_config = quantization_config
        self._hw_precision_constraints = hw_precision_constraints
        self._collect_compression_metrics = collect_compression_metrics

        self.quantized_weight_modules_registry = quantized_weight_modules_registry
        self.quantized_inputs_modules_registry = quantized_inputs_modules_registry
        self.weight_quantizers = weight_quantizers
        self.non_weight_quantizers = non_weight_quantizers
        self.all_quantizations = OrderedDict()
        self.all_quantizations.update(self.weight_quantizers)
        self.all_quantizations.update({k: v.quantizer_module_ref for k, v in self.non_weight_quantizers.items()})
        self._distributed = False

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
            quantizer_setup_type = self.quantization_config.get('quantizer_setup_type')
            # These metrics are collected here and are updated when the method .statistics() is called
            self.non_stable_metric_collectors = [NetworkQuantizationShareMetric(target_model, self.weight_quantizers, \
                                                                                self.non_weight_quantizers,
                                                                                quantizer_setup_type),
                                                 MemoryCostMetric(target_model, self.weight_quantizers,
                                                                  self.non_weight_quantizers)]
            # These metrics are collected once here and are not updated when the method .statistics() is called
            self.stable_metric_collectors = [ShareEdgesQuantizedDataPath(target_model)]
            self.update_metric_store(True)

        params = quantization_config.get('params', None)
        self.is_staged_scheduler = bool(params)

        if is_main_process() and should_init:
            self.initialize_quantizer_params()

        # Staged scheduler must be created after initialized to prevent extra logic with disabled quantizations
        if self.is_staged_scheduler:
            scheduler_cls = QUANTIZATION_SCHEDULERS.get("staged")
            self._scheduler = scheduler_cls(self, params)

    def prepare_for_export(self):
        for quantizer_id, quantizer in self.all_quantizations.items():
            if not quantizer.is_enabled_quantization():
                nncf_logger.warning('Disabled quantization on export to ONNX: {}'.format(quantizer_id))

    def update_metric_store(self, do_all: bool = False):
        for collector in self.non_stable_metric_collectors:
            collector.collect()
            self.metric_store[collector.NAME_STR] = collector.get_metric_table()
        if do_all:
            for collector in self.stable_metric_collectors:
                collector.collect()
                self.metric_store[collector.NAME_STR] = collector.get_metric_table()

    def distributed(self):
        self._distributed = True
        self._broadcast_initialized_params_for_each_quantizer()

    def _broadcast_initialized_params_for_each_quantizer(self):
        # NOTE: Order of quantization modules must be the same on GPUs to correctly broadcast num_bits
        sorted_quantizers = OrderedDict(sorted(self.all_quantizations.items(), key=lambda x: str(x[0])))
        for quantizer in sorted_quantizers.values():  # type: BaseQuantizer
            quantizer.broadcast_initialized_params()

    def _do_range_init(self, data_loader,
                       num_init_steps: int,
                       global_init_range_config: dict,
                       device: str):
        modules_to_init = OrderedDict()
        scope_overrides = self.quantization_config.get("scope_overrides", {})

        for class_type in QUANTIZATION_MODULES.registry_dict.values():
            quantization_type = class_type.__name__
            module_dict = get_all_modules_by_type(self._model, quantization_type)
            for scope, module in module_dict.items():
                quantizer_group = "weights" if module.is_weights else "activations"
                module_init_range_config = self._get_local_init_range_config(scope, scope_overrides,
                                                                             global_init_range_config, quantizer_group)
                modules_to_init[str(scope)] = (module, module_init_range_config)

        # NOTE: Order of modules must be the same to correctly broadcast parameters (e.g. input_low
        # and input_range)
        modules_to_init = OrderedDict(sorted(modules_to_init.items()))
        self.modules_to_range_init = modules_to_init
        runner = DataLoaderRangeInitializeRunner(self._model, modules_to_init, device)

        quantizers = [module for module, config in modules_to_init.values()]
        quantizers_switcher = QuantizersSwitcher(quantizers)
        # bypass quantization to collect statistics from floating point model
        quantizers_switcher.disable_quantizers()
        runner.run(data_loader, num_init_steps)
        quantizers_switcher.enable_quantizers()

        self._model.rebuild_graph()

    def compression_level(self) -> CompressionLevel:
        if self.is_staged_scheduler:
            return self.scheduler.compression_level()
        return CompressionLevel.FULL

    def initialize_quantizer_params(self):
        """ For the quantization there are 2 types of initializations: range and precision
            BatchNorm statistics are updated for the quantized model as a final initialization step.
        """
        self.init_range()
        self.init_precision()
        self.run_batchnorm_adaptation(self.quantization_config)

    def init_precision(self):
        """
        Precision initialization happens based on an measure of layer sensitivity to perturbations. The measure is
        calculated by average Hessian trace estimation for each layer using Hutchinson algorithm.
        Parameters for the quantization algorithm:
            'data_loader' - provides an iterable over the given dataset, instance of 'torch.utils.data.DataLoader'
            'criterion' - loss function, instance of `torch.nn.modules.loss._Loss`,
        """
        init_config = self.quantization_config.get('initializer', {})
        init_precision_config = init_config.get('precision', None)
        if init_precision_config:
            precision_init_type = init_precision_config.get('type', 'manual')
            precision_init_args = None
            if precision_init_type == 'hawq':
                try:
                    precision_init_args = self.quantization_config.get_extra_struct(QuantizationPrecisionInitArgs)
                except KeyError:
                    raise ValueError(
                        'Specified non-manual precision initialization in the NNCF config, '
                        'but the initializing data loader and loss criterion are not provided as an extra struct. '
                        'Refer to `NNCFConfig.register_extra_structs` and the `QuantizationPrecisionInitArgs` class')
            elif precision_init_type == 'autoq':
                try:
                    precision_init_args = self.quantization_config.get_extra_struct(AutoQPrecisionInitArgs)
                except KeyError:
                    raise ValueError(
                        'Specified Automated precision initialization in the NNCF config, '
                        'but the initializing data loader and loss criterion are not provided as an extra struct. '
                        'Refer to `NNCFConfig.register_extra_structs` and the `AutoQPrecisionInitArgs` class')

            init_impl = PrecisionInitializerFactory.create(precision_init_type)
            initializer = init_impl(self, init_precision_config, precision_init_args)
            nncf_logger.info("Initialization of quantization precisions")
            initializer.apply_init()

    def init_range(self):
        """
        Calculates per-layer activation statistics on training dataset in order to choose proper output range for
        quantization.
        Parameters for quantization algorithm:
            'data_loader' - provides an iterable over the given dataset, instance of 'torch.utils.data.DataLoader'
        """
        init_config = self.quantization_config.get('initializer', {})
        init_range_config = init_config.get('range', {})
        if not init_range_config:
            try:
                self.quantization_config.get_extra_struct(QuantizationRangeInitArgs)
                has_range_init_args = True
            except KeyError:
                has_range_init_args = False

            if has_range_init_args:
                nncf_logger.warning("Enabling quantization range initialization with default parameters.")
                num_init_samples = 256
            else:
                nncf_logger.warning("Initializer section not specified for quantization algorithm in NNCF config and "
                                    "quantization init args not supplied - quantizer range initialization will not be "
                                    "done")
                num_init_samples = 0

            init_range_config = {'num_init_samples': num_init_samples}
        if isinstance(init_range_config, dict):
            global_init_range_config = self.update_range_config_by_default(init_range_config)
            max_num_init_samples = global_init_range_config['num_init_samples']
        else:
            max_num_init_samples = 0
            global_init_range_config = []
            for sub_init_range_config in init_range_config:
                global_init_range_config.append(self.update_range_config_by_default(sub_init_range_config))
                max_num_init_samples = max(sub_init_range_config['num_init_samples'], max_num_init_samples)

        if max_num_init_samples > 0:
            try:
                range_init_args = self.quantization_config.get_extra_struct(QuantizationRangeInitArgs)
            except KeyError:
                raise ValueError(
                    'Should run range initialization as specified via config,'
                    'but the initializing data loader is not provided as an extra struct. '
                    'Refer to `NNCFConfig.register_extra_structs` and the `QuantizationRangeInitArgs` class')
            data_loader = range_init_args.data_loader
            batch_size = data_loader.batch_size
            max_num_init_steps = np.ceil(max_num_init_samples / batch_size)

            self._do_range_init(data_loader, max_num_init_steps, global_init_range_config,
                                range_init_args.device)

        if self._distributed:
            self._broadcast_initialized_params_for_each_quantizer()

    def update_range_config_by_default(self, init_range_config: Dict):
        global_init_range_config = dict()
        global_init_range_config.update(init_range_config)
        if global_init_range_config.get("type") is None:
            global_init_range_config["type"] = "mean_min_max"

        if global_init_range_config.get("num_init_samples") is None:
            global_init_range_config["num_init_samples"] = 256

        num_init_samples = global_init_range_config.get('num_init_samples', 256)
        if num_init_samples < 0:
            raise AttributeError('Number of initialization samples must be >= 0')
        return global_init_range_config

    def get_weights_activation_quantizers_pairs(self) -> List[Tuple[List[BaseQuantizer], BaseQuantizer]]:
        """
        finds all neighbour weight and input activation quantizers that share the same module (e.g. conv or linear).
        Single activation quantizer can be in pair with multiple neighbour weight quantizers, e.g. like in SqueezeNet,
        when two Convolutions share the same input activation.
        :return: list of pairs - (list of weight quantizers, activation quantizer)
        """
        pairs = []
        qimr = OrderedDict(sorted(self.quantized_inputs_modules_registry.items()))
        for _, quantized_module in qimr.items():
            weight_quantizer = None
            activation_quantizer = None
            for ops in quantized_module.pre_ops.values():
                if isinstance(ops, UpdateWeight):
                    weight_quantizer = ops.op
                if isinstance(ops, UpdateInputs):
                    activation_quantizer = ops.op
            if weight_quantizer:
                pairs.append(([weight_quantizer], activation_quantizer))

        nncf_network = self._model
        nncf_graph = nncf_network.get_original_graph()
        non_weight_quantizers = {key: quantizer_info.quantizer_module_ref for key, quantizer_info \
                                 in self.non_weight_quantizers.items() if not isinstance(key, InputQuantizerId)}

        def traverse_graph(curr_nx_node_key: str, weight_quantizers: List[nn.Module]) -> Optional[List[nn.Module]]:
            nx_node = nncf_graph.get_nx_node_by_key(curr_nx_node_key)
            module_scope = nx_node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].scope_in_model
            module = nncf_network.get_module_by_scope(module_scope)
            if is_nncf_module(module):
                if hasattr(module, 'pre_ops'):
                    for ops in module.pre_ops.values():
                        if isinstance(ops, UpdateWeight):
                            weight_quantizers.append(ops.op)
            else:
                for succ_nx_node_key in nncf_graph.get_successors(curr_nx_node_key):
                    return traverse_graph(succ_nx_node_key, weight_quantizers)
            return weight_quantizers

        # pylint: disable=unnecessary-lambda
        for quantizer_id in sorted(non_weight_quantizers, key=lambda x: str(x)):
            activation_ctx = quantizer_id.ia_op_exec_context
            post_hooked_nx_node_key = nncf_graph.get_node_id_by_iap_context(activation_ctx)
            weight_quantizers = []
            for next_nx_node_key in nncf_graph.get_successors(post_hooked_nx_node_key):
                weight_quantizers = traverse_graph(next_nx_node_key, weight_quantizers)
            if weight_quantizers:
                activation_quantizer = self.non_weight_quantizers[quantizer_id].quantizer_module_ref
                pairs.append((weight_quantizers, activation_quantizer))
        return pairs

    def _set_quantization_status(self, condition_fn: Callable[[BaseQuantizer], bool],
                                 apply_fn: Callable[[BaseQuantizer], None]):
        if self._model is not None:
            for m in self.all_quantizations.values():
                if condition_fn(m):
                    apply_fn(m)

    def enable_activation_quantization(self):
        self._set_quantization_status(lambda x: not x.is_weights, lambda x: x.enable_quantization())

    def enable_weight_quantization(self):
        self._set_quantization_status(lambda x: x.is_weights, lambda x: x.enable_quantization())

    def disable_activation_quantization(self):
        self._set_quantization_status(lambda x: not x.is_weights, lambda x: x.disable_quantization())

    def disable_weight_quantization(self):
        self._set_quantization_status(lambda x: x.is_weights, lambda x: x.disable_quantization())

    def _get_local_init_range_config(self, scope: Scope, scope_overrides: Dict[str, Dict],
                                     global_init_range_config: Dict, quantizer_group: str):
        if isinstance(global_init_range_config, dict):
            module_init_range_config = global_init_range_config
        else:
            module_init_range_config = None
            matched_init_range_config = []
            for range_init_subconfig in global_init_range_config:
                target_scopes = range_init_subconfig.get("target_scopes", None)
                ignored_scopes = range_init_subconfig.get("ignored_scopes", None)
                target_quantizer_group = range_init_subconfig.get("target_quantizer_group", quantizer_group)
                if quantizer_group == target_quantizer_group and\
                     should_consider_scope(str(scope), target_scopes, ignored_scopes):
                    matched_init_range_config.append(range_init_subconfig)

            if len(matched_init_range_config) > 1:
                raise AssertionError("The range initialization configs conflict with each other. "
                                     "Conflicting configs: {} for scope {}.".format(matched_init_range_config,
                                                                                    str(scope)))


            if len(matched_init_range_config) == 1:
                module_init_range_config = matched_init_range_config[0]
            else:
                raise AssertionError("The range initialization configs conflict with each other. "
                                     "Conflicting configs: {} for scope {}.".format(matched_init_range_config,
                                                                                    str(scope)))

        for overridden_scope in scope_overrides.keys():
            if in_scope_list(str(scope), overridden_scope):
                override_config = scope_overrides[overridden_scope].get('initializer', {}).get("range")
                if override_config is not None:
                    module_init_range_config = override_config

        if module_init_range_config is None:
            module_init_range_config = self.update_range_config_by_default({})

        return module_init_range_config

    def statistics(self, quickly_collected_only=False):
        stats = super().statistics()
        num_enabled_quantization = len([1 for q in self.all_quantizations.values() if q.is_enabled_quantization()])
        multiplier = 100 / len(self.all_quantizations)
        stats["ratio_of_enabled_quantizations"] = num_enabled_quantization * multiplier
        if self._collect_compression_metrics and not quickly_collected_only:
            self.update_metric_store()
            for metric in self.metric_store.values():
                for add_info, table in metric.items():
                    stats[add_info] = table
        return stats


class QuantizationDebugInterface(DebugInterface):
    QUANTIZERS_IN_NNCF_MODULES_TRACKER_NAME = 'quantized_modules'
    ACTIVATION_QUANTIZERS_TRACKER_NAME = 'activation_quantizers'
    FUNCTION_QUANTIZERS_TRACKER_NAME = 'function_quantizers'

    def __init__(self):
        self.call_trackers = {
            self.QUANTIZERS_IN_NNCF_MODULES_TRACKER_NAME: CallCountTracker(
                QuantizationDebugInterface.QUANTIZERS_IN_NNCF_MODULES_TRACKER_NAME),
            self.ACTIVATION_QUANTIZERS_TRACKER_NAME: CallCountTracker(
                QuantizationDebugInterface.ACTIVATION_QUANTIZERS_TRACKER_NAME),
            self.FUNCTION_QUANTIZERS_TRACKER_NAME: CallCountTracker(
                self.FUNCTION_QUANTIZERS_TRACKER_NAME)
        }
        self.graph_size = 0

        from nncf.debug import DEBUG_LOG_DIR
        self.dump_dir = Path(DEBUG_LOG_DIR) / Path("debug_dumps")
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.scale_dump_dir = self.dump_dir / Path("scale")
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
            CompressionModuleType.ACTIVATION_QUANTIZER).keys()  # type: List[str]
        function_input_quantizer_id_list = owner_model.get_compression_modules_by_type(
            CompressionModuleType.FUNCTION_QUANTIZER).keys()  # type: List[str]
        self.call_trackers[self.QUANTIZERS_IN_NNCF_MODULES_TRACKER_NAME].init_with_key_list(
            nncf_module_quantizations_id_list)
        self.call_trackers[self.ACTIVATION_QUANTIZERS_TRACKER_NAME].init_with_key_list(
            activation_quantizer_id_list)
        self.call_trackers[self.FUNCTION_QUANTIZERS_TRACKER_NAME].init_with_key_list(
            function_input_quantizer_id_list)
        if self.scale_dump_dir.exists():
            shutil.rmtree(str(self.scale_dump_dir))
        self.scale_dump_dir.mkdir(parents=True, exist_ok=True)
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
            with open(str(self.scale_dump_dir / fname), "ba") as file:
                np.savetxt(file, scale_param.cpu().numpy().flatten())

    def reset_counters(self):
        for tracker in self.call_trackers.values():
            tracker.reset()

    def register_quantizer_module_call(self, key, counts=None):
        self.call_trackers[self.QUANTIZERS_IN_NNCF_MODULES_TRACKER_NAME].register_call(key, counts)

    def register_activation_quantize_call(self, key: str):
        self.call_trackers[self.ACTIVATION_QUANTIZERS_TRACKER_NAME].register_call(key)

    def register_function_quantizer_call(self, key: str):
        self.call_trackers[self.FUNCTION_QUANTIZERS_TRACKER_NAME].register_call(key)

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
                insertion_point_data = node[InsertionPointGraph.INSERTION_POINT_DATA_NODE_ATTR]  # type: InsertionPoint
                label = "IP: {}".format(insertion_point_data.insertion_type)
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
