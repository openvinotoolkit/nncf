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
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import tensorflow as tf

import nncf
from nncf import NNCFConfig
from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.api.compression import CompressionStage
from nncf.common.compression import BaseCompressionAlgorithmController
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.layer_attributes import ConvertDtypeLayerAttributes
from nncf.common.graph.operator_metatypes import INPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OUTPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.utils import get_first_nodes_of_type
from nncf.common.hardware.config import get_hw_config_type
from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.logging import nncf_logger
from nncf.common.quantization.config_assignment import assign_qconfig_lists_to_modules
from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver
from nncf.common.quantization.quantizer_propagation.structs import IgnoreReason
from nncf.common.quantization.quantizer_setup import ActivationQuantizationInsertionPoint
from nncf.common.quantization.quantizer_setup import QuantizationPointId
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.structs import QuantizationConstraints
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.scopes import check_scopes_in_graph
from nncf.common.stateful_classes_registry import TF_STATEFUL_CLASSES
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.api_marker import api
from nncf.common.utils.backend import copy_model
from nncf.config.extractors import extract_range_init_params
from nncf.config.schemata.defaults import QUANTIZATION_OVERFLOW_FIX
from nncf.config.schemata.defaults import QUANTIZE_INPUTS
from nncf.config.schemata.defaults import QUANTIZE_OUTPUTS
from nncf.config.schemata.defaults import TARGET_DEVICE
from nncf.tensorflow.algorithm_selector import TF_COMPRESSION_ALGORITHMS
from nncf.tensorflow.api.compression import TFCompressionAlgorithmBuilder
from nncf.tensorflow.graph.converter import TFModelConverter
from nncf.tensorflow.graph.converter import TFModelConverterFactory
from nncf.tensorflow.graph.metatypes.common import CAST_METATYPES
from nncf.tensorflow.graph.metatypes.common import ELEMENTWISE_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.common import GENERAL_CONV_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.common import LINEAR_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.keras_layers import TFConcatenateLayerMetatype
from nncf.tensorflow.graph.metatypes.keras_layers import TFLambdaLayerMetatype
from nncf.tensorflow.graph.metatypes.keras_layers import TFLayerWithWeightsMetatype
from nncf.tensorflow.graph.metatypes.tf_ops import TFConcatOpMetatype
from nncf.tensorflow.graph.metatypes.tf_ops import TFIdentityOpMetatype
from nncf.tensorflow.graph.metatypes.tf_ops import TFOpWithWeightsMetatype
from nncf.tensorflow.graph.transformations.commands import TFAfterLayer
from nncf.tensorflow.graph.transformations.commands import TFBeforeLayer
from nncf.tensorflow.graph.transformations.commands import TFInsertionCommand
from nncf.tensorflow.graph.transformations.commands import TFLayerWeight
from nncf.tensorflow.graph.transformations.commands import TFMultiLayerPoint
from nncf.tensorflow.graph.transformations.layout import TFTransformationLayout
from nncf.tensorflow.graph.utils import get_original_name_and_instance_idx
from nncf.tensorflow.hardware.config import TFHWConfig
from nncf.tensorflow.hardware.fused_patterns import TF_HW_FUSED_PATTERNS
from nncf.tensorflow.layers.custom_objects import NNCF_QUANTIZATION_OPERATIONS
from nncf.tensorflow.loss import TFZeroCompressionLoss
from nncf.tensorflow.quantization.collectors import TFQuantizationStatisticsCollector
from nncf.tensorflow.quantization.default_quantization import DEFAULT_TF_QUANT_TRAIT_TO_OP_DICT
from nncf.tensorflow.quantization.init_range import RangeInitializer
from nncf.tensorflow.quantization.init_range import TFRangeInitParams
from nncf.tensorflow.quantization.layers import FakeQuantize
from nncf.tensorflow.quantization.quantizers import Quantizer
from nncf.tensorflow.quantization.quantizers import TFQuantizerSpec
from nncf.tensorflow.quantization.utils import apply_overflow_fix

QUANTIZATION_LAYER_METATYPES = GENERAL_CONV_LAYER_METATYPES + LINEAR_LAYER_METATYPES

UNSUPPORTED_LAYER_METATYPES = [TFLambdaLayerMetatype]


class TFQuantizationPointStateNames:
    QUANTIZER_SPEC = "quantizer_spec"
    TARGET_POINT = "target_point"
    TARGET_POINT_CLASS_NAME = "target_point_class_name"
    OP_NAME = "op_name"


class TFQuantizationPoint:
    """
    Characterizes where and how to insert a single quantization node to the model's graph. Stores TF-specific data.
    """

    _state_names = TFQuantizationPointStateNames

    def __init__(self, op_name: str, quantizer_spec: TFQuantizerSpec, target_point: TargetPoint):
        self.target_point = target_point
        self.op_name = op_name
        self.quantizer_spec = quantizer_spec

    def is_weight_quantization(self) -> bool:
        """
        Determines whether quantization point is for weights
        :return: true, if quantization for weights, false - for activations
        """
        return isinstance(self.target_point, TFLayerWeight)

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {
            self._state_names.TARGET_POINT: self.target_point.get_state(),
            self._state_names.TARGET_POINT_CLASS_NAME: self.target_point.__class__.__name__,
            self._state_names.QUANTIZER_SPEC: self.quantizer_spec.get_state(),
            self._state_names.OP_NAME: self.op_name,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "TFQuantizationPoint":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        target_point_cls = TF_STATEFUL_CLASSES.get_registered_class(state[cls._state_names.TARGET_POINT_CLASS_NAME])
        kwargs = {
            cls._state_names.TARGET_POINT: target_point_cls.from_state(state[cls._state_names.TARGET_POINT]),
            cls._state_names.QUANTIZER_SPEC: TFQuantizerSpec.from_state(state[cls._state_names.QUANTIZER_SPEC]),
            cls._state_names.OP_NAME: state[cls._state_names.OP_NAME],
        }
        return cls(**kwargs)


class TFQuantizationSetupStateNames:
    QUANTIZATION_POINTS = "quantization_points"
    UNIFIED_SCALE_GROUPS = "unified_scale_groups"


class TFQuantizationSetup:
    """
    Characterizes where and how to insert all quantization nodes to the model's graph
    """

    _state_names = TFQuantizationSetupStateNames

    def __init__(self):
        super().__init__()
        self._quantization_points: List[TFQuantizationPoint] = []
        self._unified_scale_groups: List[List[QuantizationPointId]] = []

    def add_quantization_point(self, point: TFQuantizationPoint):
        """
        Adds quantization point to the setup

        :param point: quantization point
        """
        self._quantization_points.append(point)

    def __iter__(self):
        return iter(self._quantization_points)

    def register_unified_scale_group(self, point_ids: List[List[QuantizationPointId]]):
        """
        Adds unified scale group to the setup

        :param point_ids: quantization point indexes
        """
        self._unified_scale_groups.append(point_ids)

    def get_quantization_points(self) -> List[TFQuantizationPoint]:
        """
        Returns quantization points

        :return: quantization points
        """
        return self._quantization_points

    def get_unified_scale_groups(self) -> List[List[QuantizationPointId]]:
        """
        Returns unified scale groups

        :return: unified scale groups
        """
        return self._unified_scale_groups

    def get_state(self) -> Dict:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """

        quantization_points_state = [qp.get_state() for qp in self._quantization_points]
        return {
            self._state_names.QUANTIZATION_POINTS: quantization_points_state,
            self._state_names.UNIFIED_SCALE_GROUPS: self._unified_scale_groups,
        }

    @classmethod
    def from_state(cls, state: Dict) -> "TFQuantizationSetup":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        setup = TFQuantizationSetup()
        for quantization_point_state in state[cls._state_names.QUANTIZATION_POINTS]:
            quantization_point = TFQuantizationPoint.from_state(quantization_point_state)
            setup.add_quantization_point(quantization_point)

        if cls._state_names.UNIFIED_SCALE_GROUPS in state:
            for quantization_group in state[cls._state_names.UNIFIED_SCALE_GROUPS]:
                setup.register_unified_scale_group(quantization_group)
        return setup


class QBuilderStateNames:
    QUANTIZER_SETUP = "quantizer_setup"


@TF_COMPRESSION_ALGORITHMS.register("quantization")
class QuantizationBuilder(TFCompressionAlgorithmBuilder):
    _state_names = QBuilderStateNames

    DEFAULT_QCONFIG = QuantizerConfig(
        num_bits=8, mode=QuantizationMode.SYMMETRIC, signedness_to_force=None, per_channel=False
    )

    def __init__(self, config: NNCFConfig, should_init: bool = True):
        super().__init__(config, should_init)

        self.quantize_inputs = self._algo_config.get("quantize_inputs", QUANTIZE_INPUTS)
        self.quantize_outputs = self._algo_config.get("quantize_outputs", QUANTIZE_OUTPUTS)
        self._overflow_fix = self._algo_config.get("overflow_fix", QUANTIZATION_OVERFLOW_FIX)
        self._target_device = config.get("target_device", TARGET_DEVICE)
        algo_config = self._get_algo_specific_config_section()
        if self._target_device == "NPU" and "preset" in algo_config:
            raise nncf.ValidationError("The NPU target device does not support presets.")
        if self._target_device == "CPU_SPR":
            raise nncf.ValidationError("The CPU_SPR target device does not supported.")

        self.global_quantizer_constraints = {}
        self.ignored_scopes_per_group = {}
        self.target_scopes_per_group = {}
        self._op_names = []

        for quantizer_group in QuantizerGroup:
            self._parse_group_params(self._algo_config, quantizer_group)

        if self.should_init:
            self._parse_init_params()

        self._range_initializer = None
        self._bn_adaptation = None
        self._quantizer_setup = None

        self.hw_config = None
        hw_config_type = get_hw_config_type(self._target_device)
        if hw_config_type is not None:
            hw_config_path = TFHWConfig.get_path_to_hw_config(hw_config_type)
            self.hw_config = TFHWConfig.from_json(hw_config_path)

    def _load_state_without_name(self, state_without_name: Dict[str, Any]):
        """
        Initializes object from the state.

        :param state_without_name: Output of `get_state()` method.
        """
        quantizer_setup_state = state_without_name[self._state_names.QUANTIZER_SETUP]
        self._quantizer_setup = TFQuantizationSetup.from_state(quantizer_setup_state)

    def _get_state_without_name(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        quantizer_setup_state = self._quantizer_setup.get_state()
        return {self._state_names.QUANTIZER_SETUP: quantizer_setup_state}

    def _parse_init_params(self):
        self._range_init_params = self._parse_range_init_params()
        self._bn_adapt_params = self._parse_bn_adapt_params()

    def _parse_range_init_params(self) -> TFRangeInitParams:
        range_init_params = extract_range_init_params(self.config)
        return TFRangeInitParams(**range_init_params) if range_init_params is not None else None

    def _parse_group_params(self, quant_config: Dict, quantizer_group: QuantizerGroup) -> None:
        group_name = quantizer_group.value
        params_dict = {}
        params_dict_from_config = quant_config.get(group_name, {})
        preset = quant_config.get("preset")
        if self._target_device in ["ANY", "CPU", "GPU"] or self._target_device == "TRIAL" and preset is not None:
            preset = QuantizationPreset(quant_config.get("preset", "performance"))
            params_dict = preset.get_params_configured_by_preset(quantizer_group)
            overriden_params = params_dict.keys() & params_dict_from_config.keys()
            if overriden_params:
                nncf_logger.info(f"Preset quantizer parameters {overriden_params} explicitly overridden by config.")
        params_dict.update(params_dict_from_config)
        self.global_quantizer_constraints[quantizer_group] = QuantizationConstraints.from_config_dict(params_dict)
        self.ignored_scopes_per_group[quantizer_group] = params_dict_from_config.get("ignored_scopes", [])
        if self.ignored_scopes is not None:
            self.ignored_scopes_per_group[quantizer_group] += self.ignored_scopes
        target_scopes = params_dict_from_config.get("target_scopes")
        if target_scopes is None and self.target_scopes is not None:
            self.target_scopes_per_group[quantizer_group] = self.target_scopes
        else:
            self.target_scopes_per_group[quantizer_group] = target_scopes

    def _get_default_qconfig(self, constraints: QuantizationConstraints = None) -> QuantizerConfig:
        qconfig = deepcopy(self.DEFAULT_QCONFIG)
        if constraints is not None:
            qconfig = constraints.apply_constraints_to(qconfig)
        return qconfig

    def _get_half_range(
        self, qconfig: QuantizerConfig, target_node: NNCFNode, first_conv_nodes: List[NNCFNode]
    ) -> bool:
        if self._target_device in ["CPU", "ANY"] and qconfig.num_bits == 8:
            if self._overflow_fix == "enable":
                return True
            if self._overflow_fix == "first_layer_only" and target_node in first_conv_nodes:
                return True
        return False

    def _create_quantizer(self, name: str, qspec: TFQuantizerSpec) -> Quantizer:
        quantizer_cls = NNCF_QUANTIZATION_OPERATIONS.get(qspec.mode)
        return quantizer_cls(name, qspec)

    def _build_insertion_commands_for_quantizer_setup(
        self, quantizer_setup: TFQuantizationSetup
    ) -> List[TFInsertionCommand]:
        insertion_commands = []
        quantization_points = quantizer_setup.get_quantization_points()
        non_unified_scales_quantization_point_ids = set(range(len(quantization_points)))

        for unified_scales_group in quantizer_setup.get_unified_scale_groups():
            if not unified_scales_group:
                continue
            us_qp_id = unified_scales_group[0]
            qp = quantization_points[us_qp_id]
            quantizer_spec = qp.quantizer_spec
            op_name = qp.op_name + "/unified_scale_group"
            quantizer = FakeQuantize(quantizer_spec, name=op_name)
            self._op_names.append(quantizer.op_name)
            target_points = []
            for us_qp_id in unified_scales_group:
                non_unified_scales_quantization_point_ids.discard(us_qp_id)
                qp = quantization_points[us_qp_id]
                assert quantizer_spec.get_state() == qp.quantizer_spec.get_state()
                target_points.append(qp.target_point)

            command = TFInsertionCommand(
                target_point=TFMultiLayerPoint(target_points),
                callable_object=quantizer,
                priority=TransformationPriority.QUANTIZATION_PRIORITY,
            )

            insertion_commands.append(command)

        for qp_id in non_unified_scales_quantization_point_ids:
            quantization_point = quantization_points[qp_id]
            op_name = quantization_point.op_name
            quantizer_spec = quantization_point.quantizer_spec
            target_point = quantization_point.target_point
            if quantization_point.is_weight_quantization():
                quantizer = self._create_quantizer(op_name, quantizer_spec)
                self._op_names.append(op_name)
            else:
                quantizer = FakeQuantize(quantizer_spec, name=op_name)
                self._op_names.append(quantizer.op_name)
            command = TFInsertionCommand(
                target_point=target_point,
                callable_object=quantizer,
                priority=TransformationPriority.QUANTIZATION_PRIORITY,
            )
            insertion_commands.append(command)
        return insertion_commands

    def get_transformation_layout(self, model: tf.keras.Model) -> TFTransformationLayout:
        transformations = TFTransformationLayout()
        if self._quantizer_setup is None:
            self._quantizer_setup = self._get_quantizer_setup(model)
        insertion_commands = self._build_insertion_commands_for_quantizer_setup(self._quantizer_setup)
        for command in insertion_commands:
            transformations.register(command)
        return transformations

    def _get_custom_layer_node_names(self, nncf_graph: NNCFGraph, converter: TFModelConverter) -> List[NNCFNodeName]:
        retval = []
        for node in nncf_graph.get_all_nodes():
            metatype = node.metatype
            if metatype in OUTPUT_NOOP_METATYPES:
                continue
            is_custom, _ = converter.get_layer_info_for_node(node.node_name)
            if is_custom:
                retval.append(node.node_name)
        return retval

    def _build_controller(self, model: tf.keras.Model) -> "QuantizationController":
        return QuantizationController(model, self.config, self._op_names)

    def initialize(self, model: tf.keras.Model) -> None:
        if self._range_init_params is not None:
            self._run_range_initialization(model)
        if self._bn_adapt_params is not None:
            self._run_batchnorm_adaptation(model)

    def _run_range_initialization(self, model: tf.keras.Model) -> None:
        if self._range_initializer is None:
            self._range_initializer = RangeInitializer(self._range_init_params)
        self._range_initializer.run(model)

    def _run_batchnorm_adaptation(self, model: tf.keras.Model) -> None:
        if self._bn_adaptation is None:
            self._bn_adaptation = BatchnormAdaptationAlgorithm(
                self._bn_adapt_params["data_loader"],
                self._bn_adapt_params["num_bn_adaptation_samples"],
                self._bn_adapt_params["device"],
            )
        self._bn_adaptation.run(model)

    def _get_quantizer_setup(self, model: tf.keras.Model) -> TFQuantizationSetup:
        converter = TFModelConverterFactory.create(model)
        nncf_graph = converter.convert()
        nncf_graph = QuantizationBuilder._preprocess_cast_nodes(nncf_graph, CAST_METATYPES)

        check_scopes_in_graph(nncf_graph, self.ignored_scopes, self.target_scopes, self.validate_scopes)

        self._raise_not_supported_warning(nncf_graph)

        quantizable_weighted_layer_nodes = self._get_quantizable_weighted_layer_nodes(nncf_graph)
        custom_layer_nodes = self._get_custom_layer_node_names(nncf_graph, converter)

        quantizer_setup = self._get_quantizer_propagation_solution(
            nncf_graph, quantizable_weighted_layer_nodes, custom_layer_nodes, model
        )
        setup = TFQuantizationSetup()

        quantized_layer_names_vs_qconfigs: Dict[str, QuantizerConfig] = {}
        qp_id_to_index: Dict[QuantizationPointId, int] = {}
        tf_setup_qp_index = 0
        applied_overflow_fix = False
        first_conv_nodes = get_first_nodes_of_type(nncf_graph, ["Conv2D", "Conv3D"])
        for qp_id, qp in quantizer_setup.quantization_points.items():
            if qp.is_weight_quantization_point():
                target_node = nncf_graph.get_node_by_name(qp.insertion_point.target_node_name)
                is_custom, layer_info = converter.get_layer_info_for_node(target_node.node_name)
                if is_custom:
                    raise nncf.InternalError("Quantizing custom layer weights is currently unsupported!")
                layer_name = layer_info.layer_name
                qconfig = qp.qconfig
                if layer_name in quantized_layer_names_vs_qconfigs:
                    assigned_qconfig = quantized_layer_names_vs_qconfigs[layer_name]
                    if qconfig != assigned_qconfig:
                        raise nncf.InternalError(
                            f"Inconsistent quantizer configurations selected by solver for one and the "
                            f"same quantizable layer! Tried to assign {qconfig} to {layer_name} as "
                            f"specified by QP {qp_id}, but the layer already has quantizer "
                            f"config {assigned_qconfig} assigned to it!"
                        )
                    continue  # The layer has already been quantized
                quantized_layer_names_vs_qconfigs[layer_name] = qconfig
                metatype = target_node.metatype
                assert issubclass(metatype, TFLayerWithWeightsMetatype)
                for weight_def in metatype.weight_definitions:
                    op_name = self._get_quantizer_operation_name(target_node.node_name, weight_def.weight_attr_name)
                    self._op_names.append(op_name)

                    half_range = self._get_half_range(qconfig, target_node, first_conv_nodes)
                    applied_overflow_fix = applied_overflow_fix or half_range
                    quantizer_spec = TFQuantizerSpec.from_config(
                        qconfig, narrow_range=not half_range, half_range=half_range
                    )
                    target_point = TFLayerWeight(layer_info.layer_name, weight_def.weight_attr_name)
                    qpoint = TFQuantizationPoint(op_name, quantizer_spec, target_point)
            else:
                assert qp.is_activation_quantization_point()
                ip = qp.insertion_point
                assert isinstance(ip, ActivationQuantizationInsertionPoint)
                target_node_name = ip.target_node_name
                input_port_id = ip.input_port_id
                fake_quantize_name = self._get_fake_quantize_name(target_node_name, input_port_id)
                quantizer_spec = TFQuantizerSpec.from_config(qp.qconfig, narrow_range=False, half_range=False)
                fake_quantize_layer = FakeQuantize(quantizer_spec, name=fake_quantize_name)
                self._op_names.append(fake_quantize_layer.op_name)

                is_custom, layer_info = converter.get_layer_info_for_node(target_node_name)
                if is_custom:
                    raise nncf.InternalError("Quantizing custom layer activations is currently unsupported!")
                if input_port_id is not None:
                    target_point = TFBeforeLayer(
                        layer_info.layer_name, instance_idx=layer_info.instance_idx, input_port_id=input_port_id
                    )
                else:
                    target_point = TFAfterLayer(
                        layer_info.layer_name, instance_idx=layer_info.instance_idx, output_port_id=0
                    )
                qpoint = TFQuantizationPoint(fake_quantize_name, quantizer_spec, target_point)

            setup.add_quantization_point(qpoint)
            qp_id_to_index[qp_id] = tf_setup_qp_index
            tf_setup_qp_index += 1

        setup = self._generate_unified_scale_groups(model, quantizer_setup, qp_id_to_index, setup)

        self._log_if_overflow_fix_was_applied(applied_overflow_fix)

        return setup

    def _raise_not_supported_warning(self, graph: NNCFGraph) -> None:
        for node in graph.get_all_nodes():
            if node.metatype in UNSUPPORTED_LAYER_METATYPES:
                nncf_logger.warning(
                    f"Layer "
                    f"{get_original_name_and_instance_idx(node.node_name)[0]} "
                    f"is not supported by the quantization algorithm"
                )

    def _log_if_overflow_fix_was_applied(self, applied_overflow_fix: bool):
        if applied_overflow_fix:
            if self._overflow_fix == "enable":
                quantizers_with_overflow_fix_str = "all weight quantizers"
            elif self._overflow_fix == "first_layer_only":
                quantizers_with_overflow_fix_str = "first convolution weight quantizers"
            elif self._overflow_fix != "disable":
                raise nncf.InternalError(f"Unknown overflow fix type: {self._overflow_fix}")
            nncf_logger.info(f"Overflow issue fix was applied to {quantizers_with_overflow_fix_str}.")

    def _generate_unified_scale_groups(
        self,
        model: tf.keras.Model,
        quantizer_setup: SingleConfigQuantizerSetup,
        qp_id_to_index: Dict[QuantizationPointId, int],
        setup: TFQuantizationSetup,
    ) -> TFQuantizationSetup:
        # To properly set the instance indices for FQ need to save layers order like in the model config
        layer_names = [layer.name for layer in model.layers]
        for unified_group in quantizer_setup.unified_scale_groups.values():
            sorted_unified_group = []
            for qp_id in unified_group:
                qp = quantizer_setup.quantization_points[qp_id]
                qp_layer_name = qp.insertion_point.target_node_name
                original_name, _ = get_original_name_and_instance_idx(qp_layer_name)
                layer_idx = layer_names.index(original_name)
                tf_setup_index = qp_id_to_index[qp_id]
                sorted_unified_group.append((tf_setup_index, layer_idx))

            sorted_unified_group = sorted(sorted_unified_group, key=lambda x: x[1])
            setup.register_unified_scale_group([setup_index for setup_index, _ in sorted_unified_group])
        return setup

    def _get_quantizable_weighted_layer_nodes(self, nncf_graph: NNCFGraph) -> List[QuantizableWeightedLayerNode]:
        nodes_with_weights = []
        for node in nncf_graph.get_all_nodes():
            metatype = node.metatype
            if metatype in OUTPUT_NOOP_METATYPES:
                continue

            if metatype not in QUANTIZATION_LAYER_METATYPES:
                continue

            assert issubclass(metatype, TFLayerWithWeightsMetatype) or issubclass(metatype, TFOpWithWeightsMetatype)
            nodes_with_weights.append(node)
        scope_overrides_dict = self._get_algo_specific_config_section().get("scope_overrides", {})
        weighted_node_and_qconf_lists = assign_qconfig_lists_to_modules(
            nodes_with_weights,
            self.DEFAULT_QCONFIG,
            self.global_quantizer_constraints[QuantizerGroup.WEIGHTS],
            scope_overrides_dict,
            hw_config=self.hw_config,
        )
        return [
            QuantizableWeightedLayerNode(node, qconf_list) for node, qconf_list in weighted_node_and_qconf_lists.items()
        ]

    def _get_quantizer_propagation_solution(
        self,
        nncf_graph: NNCFGraph,
        quantizable_weighted_layer_nodes: List[QuantizableWeightedLayerNode],
        custom_layer_node_names: List[NNCFNodeName],
        model: tf.keras.Model,
    ) -> SingleConfigQuantizerSetup:
        ip_graph = InsertionPointGraph(nncf_graph)

        pattern = TF_HW_FUSED_PATTERNS.get_full_pattern_graph()
        ip_graph = ip_graph.get_ip_graph_with_merged_hw_optimized_operations(pattern)

        input_preprocessing_nodes = self._get_input_preprocessing_nodes(nncf_graph, model)
        input_preprocessing_node_names = [n.node_name for n in input_preprocessing_nodes]
        if custom_layer_node_names:
            custom_layer_node_names_str = ", ".join([str(n) for n in custom_layer_node_names])
            nncf_logger.warning(
                f"Following custom layers will be ignored during quantization (custom layer quantization not supported "
                f"by NNCF yet):\n[{custom_layer_node_names_str}]"
            )
        ignored_scopes_for_solver = {
            **{name: IgnoreReason.USER_REQUESTED for name in self.ignored_scopes_per_group[QuantizerGroup.ACTIVATIONS]},
            **{name: IgnoreReason.AUTOGENERATED for name in input_preprocessing_node_names},
            **{name: IgnoreReason.AUTOGENERATED for name in custom_layer_node_names},
        }
        scales_unification_map = {
            TFConcatenateLayerMetatype: GENERAL_CONV_LAYER_METATYPES + LINEAR_LAYER_METATYPES,
            TFConcatOpMetatype: GENERAL_CONV_LAYER_METATYPES + LINEAR_LAYER_METATYPES,
        }
        solver = QuantizerPropagationSolver(
            activation_ignored_scopes=ignored_scopes_for_solver,
            weight_ignored_scopes=self.ignored_scopes_per_group[QuantizerGroup.WEIGHTS],
            activation_target_scopes=self.target_scopes_per_group[QuantizerGroup.ACTIVATIONS],
            weight_target_scopes=self.target_scopes_per_group[QuantizerGroup.WEIGHTS],
            hw_config=self.hw_config,
            default_trait_to_metatype_map=DEFAULT_TF_QUANT_TRAIT_TO_OP_DICT,
            default_qconfig_list=[
                self._get_default_qconfig(self.global_quantizer_constraints[QuantizerGroup.ACTIVATIONS])
            ],
            quantizable_layer_nodes=quantizable_weighted_layer_nodes,
            global_constraints=self.global_quantizer_constraints,
            quantize_outputs=self.quantize_outputs,
            scales_unification_map=scales_unification_map,
        )

        quantization_proposal = solver.run_on_ip_graph(ip_graph, ELEMENTWISE_LAYER_METATYPES)
        multi_config_setup = quantization_proposal.quantizer_setup
        single_config_setup = multi_config_setup.select_first_qconfig_for_each_point()
        finalized_proposal = quantization_proposal.finalize(single_config_setup)
        final_setup = solver.get_final_quantizer_setup(finalized_proposal)
        final_setup = self._handle_quantize_inputs_option(final_setup, nncf_graph)

        return final_setup

    def _handle_quantize_inputs_option(
        self, quantizer_setup: SingleConfigQuantizerSetup, nncf_graph: NNCFGraph
    ) -> SingleConfigQuantizerSetup:
        qp_ids_to_discard = []
        for qp_id, qp in quantizer_setup.quantization_points.items():
            if qp.is_activation_quantization_point():
                insertion_point = qp.insertion_point
                target_node = nncf_graph.get_node_by_name(insertion_point.target_node_name)
                if not self.quantize_inputs and target_node.metatype in INPUT_NOOP_METATYPES:
                    qp_ids_to_discard.append(qp_id)
        for qp_id in qp_ids_to_discard:
            quantizer_setup.discard(qp_id, keep_shared_input_qps=True)
        return quantizer_setup

    def _get_input_preprocessing_nodes(self, nncf_graph: NNCFGraph, model: tf.keras.Model) -> List[NNCFNode]:
        retval = []

        def traverse_fn(node: NNCFNode, preprocessing_nodes: List[NNCFNode]) -> Tuple[bool, List[NNCFNode]]:
            is_finished = True
            successors = nncf_graph.get_next_nodes(node)
            if len(successors) == 1:
                successor = next(iter(successors))
                # It is necessary to determine the number of input nodes from the model
                # in order to correctly count the duplicated edges
                original_name, _ = get_original_name_and_instance_idx(successor.node_name)
                layer = model.get_layer(name=original_name)

                num_previous_nodes = len(layer.input) if isinstance(layer.input, list) else 1
                if layer.__class__.__name__ in ["TFOpLambda", "SlicingOpLambda"]:
                    num_previous_nodes = 0
                    for inbound_node in layer.inbound_nodes:
                        num_previous_nodes += len(inbound_node.keras_inputs)

                if successor.metatype in ELEMENTWISE_LAYER_METATYPES and num_previous_nodes == 1:
                    preprocessing_nodes.append(successor)
                    is_finished = False
            return is_finished, preprocessing_nodes

        for nncf_node in nncf_graph.get_input_nodes():
            preprocessing_nodes_for_this_input = nncf_graph.traverse_graph(nncf_node, traverse_fn)
            retval += preprocessing_nodes_for_this_input

        return retval

    @staticmethod
    def _preprocess_cast_nodes(nncf_graph: NNCFGraph, cast_metatypes: List[OperatorMetatype]) -> NNCFGraph:
        cast_nodes = nncf_graph.get_nodes_by_metatypes(cast_metatypes)
        for node in cast_nodes:
            if not isinstance(node.layer_attributes, ConvertDtypeLayerAttributes):
                continue
            if node.layer_attributes.src_dtype == node.layer_attributes.dst_dtype:
                node.attributes[NNCFNode.METATYPE_ATTR] = TFIdentityOpMetatype
        return nncf_graph

    def _get_fake_quantize_name(self, node_name: NNCFNodeName, input_port_id: int = None) -> str:
        original_node_name, instance_idx = get_original_name_and_instance_idx(node_name)
        fq_name = "{}/fake_quantize".format(original_node_name)
        if instance_idx != 0:
            fq_name += f"_{instance_idx}"
        if input_port_id is not None:
            fq_name += f"_I{input_port_id}"
        return fq_name

    def _get_quantizer_operation_name(self, layer_name, weight_attr_name):
        return f"{layer_name}_{weight_attr_name}_quantizer"


@api()
class QuantizationController(BaseCompressionAlgorithmController):
    """
    Controller for the quantization algorithm in TensorFlow.
    """

    def __init__(self, target_model, config: NNCFConfig, op_names: List[str]):
        super().__init__(target_model)
        self._scheduler = BaseCompressionScheduler()
        self._loss = TFZeroCompressionLoss()
        self._op_names = op_names
        self._config = config

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    @property
    def loss(self) -> CompressionLoss:
        """
        Returns the loss that is always zero since the quantization algorithm is driven by the original loss and does
        not require additional losses.
        """
        return self._loss

    def strip_model(self, model: tf.keras.Model, do_copy: bool = False) -> tf.keras.Model:
        if do_copy:
            model = copy_model(model)
        apply_overflow_fix(model, self._op_names)
        return model

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        collector = TFQuantizationStatisticsCollector(self.model, self._op_names)
        stats = collector.collect()

        nncf_stats = NNCFStatistics()
        nncf_stats.register("quantization", stats)
        return nncf_stats

    def compression_stage(self) -> CompressionStage:
        return CompressionStage.FULLY_COMPRESSED
