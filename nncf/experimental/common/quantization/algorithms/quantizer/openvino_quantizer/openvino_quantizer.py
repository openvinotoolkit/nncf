# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from copy import deepcopy
from typing import Dict, List, Optional, Set, TypeVar, Union

import numpy as np

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns.manager import PatternsManager
from nncf.common.hardware.config import get_hw_config_type
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.logging import nncf_logger
from nncf.common.quantization.config_assignment import assign_qconfig_lists_to_modules
from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationRule
from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver
from nncf.common.quantization.quantizer_propagation.structs import IgnoreReason
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.structs import QuantizationConstraints
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.structs import QuantizationScheme
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.parameters import ModelType
from nncf.parameters import QuantizationMode
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import FP8QuantizationParameters
from nncf.quantization.advanced_parameters import FP8Type
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.advanced_parameters import QuantizationParameters
from nncf.experimental.common.quantization.algorithms.quantizer.quantizer import NNCFQuantizer
from nncf.quantization.passes import transform_to_inference_graph
from nncf.scopes import IgnoredScope
from nncf.scopes import get_ignored_node_names_from_ignored_scope

TModel = TypeVar("TModel")

DEFAULT_QCONFIG = QuantizerConfig(
    num_bits=8, mode=QuantizationScheme.SYMMETRIC, signedness_to_force=None, per_channel=False
)


@dataclasses.dataclass
class ModeBasedDefaults:
    """
    Contains default values that should be set in case of abscense.
    """

    overflow_fix: OverflowFix = OverflowFix.FIRST_LAYER
    activations_quantization_params: Union[QuantizationParameters, FP8QuantizationParameters] = dataclasses.field(
        default_factory=QuantizationParameters
    )
    weights_quantization_params: Union[QuantizationParameters, FP8QuantizationParameters] = dataclasses.field(
        default_factory=QuantizationParameters
    )


MODE_BASED_DEFAULTS = {
    None: ModeBasedDefaults(),
    QuantizationMode.FP8_E4M3: ModeBasedDefaults(
        overflow_fix=OverflowFix.DISABLE,
        activations_quantization_params=FP8QuantizationParameters(FP8Type.E4M3),
        weights_quantization_params=FP8QuantizationParameters(FP8Type.E4M3),
    ),
    QuantizationMode.FP8_E5M2: ModeBasedDefaults(
        overflow_fix=OverflowFix.DISABLE,
        activations_quantization_params=FP8QuantizationParameters(FP8Type.E5M2),
        weights_quantization_params=FP8QuantizationParameters(FP8Type.E5M2),
    ),
}


class NNCFOVQuantizer(NNCFQuantizer):
    def __init__(
        self,
        mode: Optional[QuantizationMode] = None,
        preset: Optional[QuantizationPreset] = None,
        target_device: TargetDevice = TargetDevice.ANY,
        model_type: Optional[ModelType] = None,
        ignored_scope: Optional[IgnoredScope] = None,
        overflow_fix: Optional[OverflowFix] = None,
        quantize_outputs: bool = False,
        activations_quantization_params: Union[QuantizationParameters, FP8QuantizationParameters] = None,
        weights_quantization_params: Union[QuantizationParameters, FP8QuantizationParameters] = None,
        quantizer_propagation_rule: Optional[QuantizerPropagationRule] = None,
    ):
        """
        :param mode: Defines optimization mode for the algorithm. None by default.
        :param preset: A preset controls the quantization mode (symmetric and asymmetric).
            It can take the following values:
            - `performance`: Symmetric quantization of weights and activations.
            - `mixed`: Symmetric quantization of weights and asymmetric quantization of activations.
            Default value is None. In this case, `mixed` preset is used for `transformer`
            model type otherwise `performance`.
        :param target_device: A target device the specificity of which will be taken
            into account while compressing in order to obtain the best performance
            for this type of device, defaults to TargetDevice.ANY.
        :param model_type: Model type is needed to specify additional patterns
            in the model. Supported only `transformer` now.
        :param ignored_scope: An ignored scope that defined the list of model control
            flow graph nodes to be ignored during quantization.
        :param overflow_fix: This option controls whether to apply the overflow issue
            fix for the 8-bit quantization.
        :param quantize_outputs: Whether to insert additional quantizers right before
            each of the model outputs.
        :param activations_quantization_params: Quantization parameters for model
            activations.
        :param weights_quantization_params: Quantization parameters for model weights.
        :param quantizer_propagation_rule: The strategy to be used while propagating and merging quantizers.
        """
        self._target_device = target_device
        self._mode = mode
        self._model_type = model_type
        self._overflow_fix = overflow_fix
        self._quantize_outputs = quantize_outputs
        self._activations_quantization_params = activations_quantization_params
        self._weights_quantization_params = weights_quantization_params
        self._preset = preset
        self._ignored_scope = IgnoredScope() if ignored_scope is None else ignored_scope
        self.quantizer_propagation_rule = quantizer_propagation_rule

        # preset definition
        if self._preset is None:
            if model_type == ModelType.TRANSFORMER:
                self._preset = QuantizationPreset.MIXED
            else:
                self._preset = QuantizationPreset.PERFORMANCE

        self._override_device()
        self._set_mode_based_defaults()
        self._review_mode_based_defaults()

        self._quantization_params = {
            QuantizerGroup.WEIGHTS: self._weights_quantization_params,
            QuantizerGroup.ACTIVATIONS: self._activations_quantization_params,
        }

        # Calculates global quantizer constraints
        self._global_quantizer_constraints = {}
        for quantizer_group in QuantizerGroup:
            self._global_quantizer_constraints[quantizer_group] = self._get_quantizer_constraints(
                quantizer_group, self._preset, self._quantization_params[quantizer_group]
            )

        self._algorithm_key = f"MMQ_{hash(self)}"

    def _override_device(self) -> None:
        """
        Overrides NPU device to use CPU quantization scheme.
        """
        if self._target_device == TargetDevice.NPU:
            act_bits, weight_bits = 8, 8
            if self._activations_quantization_params and self._activations_quantization_params.num_bits:
                act_bits = self._activations_quantization_params.num_bits
            if self._weights_quantization_params and self._weights_quantization_params.num_bits:
                weight_bits = self._weights_quantization_params.num_bits

            if act_bits == 8 and weight_bits == 8:
                self._target_device == TargetDevice.CPU
                nncf_logger.debug("Target device NPU was changed to CPU!")

    def _set_mode_based_defaults(self) -> None:
        """
        Sets defaults for the algorithms based on the provided mode.
        """
        mode_based_defaults = MODE_BASED_DEFAULTS[self._mode]
        for field in dataclasses.fields(mode_based_defaults):
            self_name = "_" + field.name
            default_value = getattr(mode_based_defaults, field.name)
            if getattr(self, self_name) is None:
                setattr(self, self_name, default_value)

    def _review_mode_based_defaults(self):
        """
        Reviews default values because mode option doesn't support them.
        """
        if self._mode in (QuantizationMode.FP8_E4M3, QuantizationMode.FP8_E5M2):
            nncf_logger.warning(f"You're using experimental option mode with {self._mode} value.")

            if self._preset != QuantizationPreset.PERFORMANCE:
                raise nncf.ParameterNotSupportedError(
                    f"preset option with {self._preset} value is not supported with the mode option!"
                )

            if self._target_device not in [TargetDevice.CPU, TargetDevice.ANY]:
                raise nncf.ParameterNotSupportedError(
                    f"target_device option with {self._target_device} value is not supported with the mode option!"
                )

            if self._overflow_fix != OverflowFix.DISABLE:
                raise nncf.ParameterNotSupportedError(
                    f"overflow_fix option with {self._overflow_fix} value is not supported with the mode option!"
                )

            if self._quantize_outputs:
                raise nncf.ParameterNotSupportedError("quantize_outputs option is not supported with the mode option!")

            if isinstance(self._weights_quantization_params, QuantizationParameters):
                raise nncf.ParameterNotSupportedError(
                    "quantization_params option for weights with "
                    f"{self._weights_quantization_params} "
                    "value is not supported with the mode option!"
                )

            if isinstance(self._activations_quantization_params, QuantizationParameters):
                raise nncf.ParameterNotSupportedError(
                    "quantization_params option for activations with "
                    f"{self._activations_quantization_params} "
                    "value is not supported with the mode option!"
                )
        elif self._mode is None:
            if isinstance(self._weights_quantization_params, FP8QuantizationParameters):
                raise nncf.ParameterNotSupportedError(
                    "quantization_params option for weights with "
                    f"{self._weights_quantization_params} "
                    "value is not supported with the mode: None option!"
                )

            if isinstance(self._activations_quantization_params, FP8QuantizationParameters):
                raise nncf.ParameterNotSupportedError(
                    "quantization_params option for activations with "
                    f"{self._activations_quantization_params} "
                    "value is not supported with the mode: None option!"
                )

    @property
    def available_backends(self) -> List[BackendType]:
        return [BackendType.ONNX, BackendType.OPENVINO, BackendType.TORCH, BackendType.TORCH_FX]

    def _get_quantizer_constraints(
        self,
        group: QuantizerGroup,
        preset: QuantizationPreset,
        quantization_params: Union[QuantizationParameters, FP8QuantizationParameters],
    ) -> QuantizationConstraints:
        """
        Returns QuantizationConstraints for the provided quantizer group.

        :param group: Quantizer group.
        :param preset: Quantization preset.
        :param quantization_params: Quantization parameters.
        :return: QuantizationConstraints.
        """
        constraints = {"mode": preset.get_params_configured_by_preset(group)["mode"]}
        if quantization_params is None:
            return QuantizationConstraints(**constraints)

        if isinstance(quantization_params, FP8QuantizationParameters):
            if self._mode is None:
                raise nncf.InternalError(
                    f"FP8QuantizationParameters for {group.value} can not be used without QuantizationMode option!"
                )
            return QuantizationConstraints(**constraints)

        if quantization_params.mode is not None:
            constraints["mode"] = quantization_params.mode
        if quantization_params.num_bits is not None:
            constraints["num_bits"] = quantization_params.num_bits
        if quantization_params.per_channel is not None:
            constraints["per_channel"] = quantization_params.per_channel
        if quantization_params.signedness_to_force is not None:
            constraints["signedness_to_force"] = quantization_params.signedness_to_force

        return QuantizationConstraints(**constraints)

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm

        :param model: backend-specific input model
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.TORCH_FX:
            from nncf.experimental.common.quantization.algorithms.quantizer.openvino_quantizer.torch_fx_backend import OpenVINOQuantizerBackend

            self._backend_entity = OpenVINOQuantizerBackend()
        else:
            raise nncf.UnsupportedBackendError(
                "Cannot return backend-specific entity because {} is not supported!".format(model_backend.value)
            )

    def _get_default_qconfig(self, constraints: QuantizationConstraints = None) -> QuantizerConfig:
        """
        Returns default quantizer configuration, based on the provided constraints.

        :param constraints: Quantization constraints.
        :return: Quantizer config.
        """
        qconfig = deepcopy(DEFAULT_QCONFIG)
        if constraints is not None:
            qconfig = constraints.apply_constraints_to(qconfig)
        return qconfig

    def _get_ignored_names(
        self, nncf_graph: NNCFGraph, inference_nncf_graph: NNCFGraph, ignored_patterns: GraphPattern
    ) -> Dict[str, IgnoreReason]:
        """
        Returns all node names that are ignored for quantization:
        Firstly, the ignored names are obtained from user-defined ignored the scope.
        Secondly, the ignored names are updated from model_type parameter.
        Lastly, the ignored names are updated from ignored_patterns.

        :param nncf_graph: NNCFGraph instance.
        :param inference_nncf_graph: Inference graph without constant flows.
        :param ignored_patterns: Ignored patterns.
        :return: Ignored node names and ignore reason for quantization.
        """
        user_ignored_names = get_ignored_node_names_from_ignored_scope(
            self._ignored_scope, nncf_graph, strict=self._ignored_scope.validate
        )
        autogenerated_ignored_names = self._get_ignored_names_by_ignored_patterns(
            inference_nncf_graph, ignored_patterns
        )
        autogenerated_ignored_names |= self._backend_entity.get_ignored_names_by_layer_attributes(inference_nncf_graph)
        autogenerated_ignored_names |= self._get_ignored_names_by_algorithm(inference_nncf_graph)
        ignored_names = {name: IgnoreReason.AUTOGENERATED for name in autogenerated_ignored_names}
        # User ignored scope has higher priority
        ignored_names.update({name: IgnoreReason.USER_REQUESTED for name in user_ignored_names})
        return ignored_names

    def _get_ignored_names_by_ignored_patterns(
        self, inference_nncf_graph: NNCFGraph, ignored_patterns: GraphPattern
    ) -> Set[str]:
        """
        Returns node names matched ignored_patterns.

        :param nncf_graph: Inference graph without constant flows.
        :param ignored_patterns: Ignored patterns.
        :return: IgnoredScope with all node names matched ignored_patterns.
        """
        nncf_node_names = set()
        for subgraph in inference_nncf_graph.find_matching_subgraphs(ignored_patterns, strict=False):
            for nncf_node in subgraph:
                nncf_node_names.add(nncf_node.node_name)
        return nncf_node_names

    def _get_ignored_names_by_algorithm(self, inference_nncf_graph: NNCFGraph) -> Set[str]:
        """
        Returns node names for ignored_algorithms matched `quantization`.

        :param inference_nncf_graph: Inference NNCFGraph instance.
        :return: IgnoredScope with corresponded nodes.
        """
        nncf_node_names = set()
        for nncf_node in inference_nncf_graph.get_all_nodes():
            if "ptq_quantization" in nncf_node.ignored_algorithms:
                nncf_node_names.add(nncf_node.node_name)
        return nncf_node_names

    def _get_scope_overrides(self, inference_nncf_graph: NNCFGraph) -> Dict:
        """
        Returns a dictionary of quantization configuration overrides for inputs to matching operation nodes.

        :param inference_nncf_graph: Inference NNCFGraph instance.
        :return: A dictionary of quantization configuration overrides for inputs to matching operation nodes.
        """
        scaled_dot_product_attention_node_names = [
            node.node_name
            for node in inference_nncf_graph.get_nodes_by_metatypes(
                self._backend_entity.scaled_dot_product_attention_metatypes
            )
        ]

        scope_overrides_activations = {}
        for node_name in scaled_dot_product_attention_node_names:
            scope_overrides_activations[node_name] = {"mode": "symmetric"}
        return {"activations": scope_overrides_activations}

    def _get_quantizer_setup(
        self,
        nncf_graph: NNCFGraph,
        inference_nncf_graph: NNCFGraph,
        hw_patterns: GraphPattern,
        ignored_patterns: GraphPattern,
    ) -> SingleConfigQuantizerSetup:
        """
        Returns SingleConfigQuantizerSetup instance based on the input NNCFGraph.

        :param nncf_graph: NNCFGraph instance.
        :param hw_patterns: Hardware patterns.
        :param ignored_patterns: Ignored patterns.
        :return: SingleConfigQuantizerSetup for the current NNCFGraph entity.
        """
        hw_config_type = get_hw_config_type(self._target_device.value)
        hw_config_path = self._backend_entity.hw_config.get_path_to_hw_config(hw_config_type)
        hw_config = self._backend_entity.hw_config.from_json(hw_config_path)

        ignored_names = self._get_ignored_names(nncf_graph, inference_nncf_graph, ignored_patterns)
        weight_nodes = self._backend_entity.get_weight_nodes(nncf_graph)

        default_weight_qconfig = self._get_default_qconfig(self._global_quantizer_constraints[QuantizerGroup.WEIGHTS])
        weighted_node_and_qconf_lists = assign_qconfig_lists_to_modules(
            nodes_with_weights=weight_nodes,
            default_weight_qconfig=default_weight_qconfig,
            global_weight_constraints=self._global_quantizer_constraints[QuantizerGroup.WEIGHTS],
            scope_overrides_dict=None,
            hw_config=hw_config,
        )
        quantizable_layer_nodes = [
            QuantizableWeightedLayerNode(node, qconf_list) for node, qconf_list in weighted_node_and_qconf_lists.items()
        ]

        scope_overrides = self._get_scope_overrides(inference_nncf_graph)

        ip_graph = InsertionPointGraph(inference_nncf_graph)
        ip_graph = ip_graph.get_ip_graph_with_merged_hw_optimized_operations(hw_patterns)
        post_processing_types = self._backend_entity.post_processing_metatypes
        metatypes_to_ignore = self._backend_entity.get_ignored_metatypes(self._model_type, self._target_device)
        solver = QuantizerPropagationSolver(
            activation_ignored_scopes=ignored_names,
            weight_ignored_scopes=list(ignored_names.keys()),
            hw_config=hw_config,
            default_trait_to_metatype_map=self._backend_entity.quant_trait_op_dict,
            propagation_strategy=self.quantizer_propagation_rule,
            default_qconfig_list=[
                self._get_default_qconfig(self._global_quantizer_constraints[QuantizerGroup.ACTIVATIONS])
            ],
            quantizable_layer_nodes=quantizable_layer_nodes,
            quantize_outputs=self._quantize_outputs,
            global_constraints=self._global_quantizer_constraints,
            post_processing_marker_metatypes=post_processing_types,
            metatypes_to_ignore=metatypes_to_ignore,
            scales_unification_map=self._backend_entity.scales_unification_map,
            scope_overrides=scope_overrides,
        )

        quantization_proposal = solver.run_on_ip_graph(ip_graph, self._backend_entity.elementwise_metatypes)
        multi_config_setup = quantization_proposal.quantizer_setup
        single_config_setup = multi_config_setup.select_first_qconfig_for_each_point()
        finalized_proposal = quantization_proposal.finalize(single_config_setup)
        final_setup = solver.get_final_quantizer_setup(finalized_proposal)
        return final_setup

    def get_quantization_setup(self, model: TModel, nncf_graph: NNCFGraph) -> SingleConfigQuantizerSetup:
        """
        Initializes a cache, finds quantization target points and them puts in the cache.

        :param model: Backend-specific model, for which Quantization Target Points are being seek.
        :param nncf_graph: NNCFGraph instance.
        :return: Mapping of quantization target points with associated quantization configuration,
        along with target points for scale unification.
        """
        self._set_backend_entity(model)
        backend = get_backend(model)
        device = self._target_device
        model_type = self._model_type
        ignored_patterns = PatternsManager.get_full_ignored_pattern_graph(
            backend=backend, device=device, model_type=model_type
        )
        hw_patterns = PatternsManager.get_full_hw_pattern_graph(backend=backend, device=device, model_type=model_type)

        inference_nncf_graph = transform_to_inference_graph(
            deepcopy(nncf_graph),
            self._backend_entity.get_start_nodes_for_activation_path_tracing(nncf_graph),
            self._backend_entity.shapeof_metatypes,
            self._backend_entity.dropout_metatypes,
            self._backend_entity.preserved_metatypes,
        )

        quantizer_setup = self._get_quantizer_setup(nncf_graph, inference_nncf_graph, hw_patterns, ignored_patterns)
        self._apply_model_type_pass(self._model_type, quantizer_setup, nncf_graph)
        self._apply_device_pass(self._target_device, quantizer_setup, inference_nncf_graph)
        return quantizer_setup

    def _apply_model_type_pass(
        self, model_type: Optional[ModelType], quantizer_setup: SingleConfigQuantizerSetup, nncf_graph: NNCFGraph
    ) -> None:
        """
        Applies changes in-place into quantizer setup based on model_type and device parameters.

        :param model_type: Model type parameter.
        :param quantizer_setup: Quantizer setup which considered to update.
        :param nncf_graph: Instance of NNCFGraph.
        :return: None
        """
        if model_type == ModelType.TRANSFORMER:
            for quantization_point in quantizer_setup.quantization_points.values():
                if quantization_point.is_activation_quantization_point():
                    for node_name in quantization_point.directly_quantized_operator_node_names:
                        node = nncf_graph.get_node_by_name(node_name)
                        if node.metatype not in self._backend_entity.mat_mul_metatypes:
                            continue
                        if (
                            quantization_point.qconfig.mode != QuantizationScheme.SYMMETRIC
                            and not self._backend_entity.is_matmul_with_constant(node, nncf_graph)
                        ):
                            quantization_point.qconfig.mode = QuantizationScheme.SYMMETRIC
                            nncf_logger.debug(
                                f"Update quantization mode for the node {node_name}"
                                f" to the symmetric due to ModelType parameter."
                            )

    def _apply_device_pass(
        self, target_device: TargetDevice, quantizer_setup: SingleConfigQuantizerSetup, nncf_graph: NNCFGraph
    ) -> None:
        """
        This method applies model post-processing device passes to SingleConfigQuantizerSetup in-place.

        :param target_device: TargetDevice instance.
        :param quantizer_setup: SingleConfigQuantizerSetup instance to update.
        :param nncf_graph: NNCFGraph.
        :return: None.
        """

        passes_map = {TargetDevice.CPU_SPR: self._apply_spr_pass}

        if target_device not in passes_map:
            return

        passes_map[target_device](quantizer_setup, nncf_graph)

    def _apply_spr_pass(
        self, quantizer_setup: SingleConfigQuantizerSetup, nncf_graph: NNCFGraph
    ) -> SingleConfigQuantizerSetup:
        """
        Applies CPU_SPR-related pass.
        The main action is to remove one of the quantizers before elementwise layer (e.g. Add).
        This action allows to get performance boost on SPR devices.

        :param quantizer_setup: SingleConfigQuantizerSetup instance to update.
        :param nncf_graph: NNCFGraph instance to update.
        :return: Modified SingleConfigQuantizerSetup.
        """

        def _is_node_after_producers(node):
            input_node = node
            while True:
                input_node = nncf_graph.get_previous_nodes(input_node)
                if len(input_node) > 1:
                    return False
                input_node = input_node[0]
                if input_node.metatype in producer_metatypes:
                    return True

        producer_metatypes = (
            self._backend_entity.conv_metatypes
            + self._backend_entity.mat_mul_metatypes
            + self._backend_entity.group_conv_metatypes
        )

        quantizer_setup_map = {
            p.insertion_point.target_node_name: q_key for q_key, p in quantizer_setup.quantization_points.items()
        }

        # Walking through all Add layers.
        for add_node in nncf_graph.get_nodes_by_metatypes(self._backend_entity.add_metatypes):
            add_inputs = nncf_graph.get_previous_nodes(add_node)

            # Filtering Add based on it's input.
            # Need to find Add layer only with two activations as input.
            if len(add_inputs) == 2 and all(n.node_name in quantizer_setup_map for n in add_inputs):
                # Sorting of the inputs based on length of input's consumer in descending order.
                add_inputs.sort(key=lambda n: len(nncf_graph.get_next_nodes(n)), reverse=True)
                fq_1_producer, fq_2_producer = add_inputs
                fq_1_q_key = quantizer_setup_map[fq_1_producer.node_name]
                fq_2_q_key = quantizer_setup_map[fq_2_producer.node_name]

                # In the case of the two quantizers where one of them produces data into branching,
                # it needs to remove the quantizer without branching after it.
                if (
                    len(nncf_graph.get_next_nodes(fq_1_producer)) > 1
                    and len(nncf_graph.get_next_nodes(fq_2_producer)) == 1
                ):
                    quantizer_setup.discard(fq_2_q_key, True)
                    continue

                # In the case of the two quantizers without the branching after them,
                # it needs to check that all quantizers follows after producer nodes.
                if _is_node_after_producers(fq_1_producer) and _is_node_after_producers(fq_2_producer):
                    fq_1_prod_shape = np.prod(nncf_graph.get_output_edges_by_port_id(fq_1_producer, 0)[0].tensor_shape)
                    fq_2_prod_shape = np.prod(nncf_graph.get_output_edges_by_port_id(fq_2_producer, 0)[0].tensor_shape)

                    # Then it needs to remove quantizer with the smallest shape.
                    if fq_1_prod_shape >= fq_2_prod_shape:
                        quantizer_setup.discard(fq_1_q_key, True)
                    else:
                        quantizer_setup.discard(fq_2_q_key, True)

        return quantizer_setup
