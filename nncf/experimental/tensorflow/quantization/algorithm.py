"""
 Copyright (c) 2021 Intel Corporation
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

from typing import List, Optional, Dict, Any, Tuple
from copy import deepcopy

from nncf import NNCFConfig
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph import OUTPUT_NOOP_METATYPES
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph import INPUT_NOOP_METATYPES
from nncf.common.hardware.config import HWConfigType
from nncf.common.hardware.config import HW_CONFIG_TYPE_TARGET_DEVICE_MAP
from nncf.common.utils.logger import logger
from nncf.common.utils.helpers import should_consider_scope
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.quantization.structs import QuantizationConstraints
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.config_assignment import assign_qconfig_lists_to_modules
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.quantization.quantizer_setup import QuantizationPointId
from nncf.common.quantization.quantizer_setup import ActivationQuantizationInsertionPoint
from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver
from nncf.common.stateful_classes_registry import TF_STATEFUL_CLASSES
from nncf.tensorflow.api.compression import TFCompressionAlgorithmBuilder
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.config.extractors import extract_range_init_params
from nncf.common.statistics import NNCFStatistics

from nncf.tensorflow.quantization.algorithm import QuantizationController

from nncf.experimental.tensorflow.nncf_network import NNCFNetwork
from nncf.experimental.tensorflow.graph.converter import convert_nncf_network_to_nncf_graph
from nncf.experimental.tensorflow.graph.metatypes.common import ALL_TF_OP_METATYPES_WITH_WEIGHTS
from nncf.experimental.tensorflow.graph.metatypes.tf_ops import TFOpWithWeightsMetatype
from nncf.experimental.tensorflow.graph.transformations.commands import TFTargetPoint
from nncf.experimental.tensorflow.hardware.fused_patterns import TF_HW_FUSED_PATTERNS
from nncf.experimental.tensorflow.hardware.config import TFHWConfig
from nncf.experimental.tensorflow.quantization.default_quantization import DEFAULT_TF_QUANT_TRAIT_TO_OP_DICT
from nncf.experimental.tensorflow.graph.transformations.commands import TFInsertionCommand
from nncf.experimental.tensorflow.quantization.quantizers import TFQuantizerSpec
from nncf.experimental.tensorflow.quantization.quantizers import create_quantizer
from nncf.experimental.tensorflow.graph.model_transformer import TFModelTransformer
from nncf.experimental.tensorflow.graph.transformations.layout import TFTransformationLayout
from nncf.experimental.tensorflow.quantization.initializers.init_range import TFRangeInitParamsV2
from nncf.experimental.tensorflow.quantization.initializers.init_range import RangeInitializerV2
from nncf.config.extractors import extract_bn_adaptation_init_params
from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm

# TODO(andrey-churkin): Fill it out
UNSUPPORTED_TF_OP_METATYPES = [
]


class TFQuantizationPointStateNames:
    QUANTIZER_SPEC = 'quantizer_spec'
    TARGET_POINT = 'target_point'
    TARGET_POINT_CLASS_NAME = 'target_point_class_name'
    OP_NAME = 'op_name'


class TFQuantizationPoint:
    """
    Characterizes where and how to insert a single quantization node to the model's graph. Stores TF-specific data.
    """

    _state_names = TFQuantizationPointStateNames

    def __init__(self,
                 op_name: str,
                 quantizer_spec: TFQuantizerSpec,
                 target_point: TargetPoint,
                 is_weight_quantization: bool,
                 input_shape: Optional[List[int]] = None,
                 channel_axes: Optional[List[int]] = None):
        """
        :param op_name:
        :param quantizer_spec:
        :param target_point:
        :param input_shape:
        :param channel_axes:
        """
        self.target_point = target_point
        self.op_name = op_name
        self.is_weight_quantization = is_weight_quantization
        self.quantizer_spec = quantizer_spec
        self.input_shape = input_shape
        self.channel_axes = channel_axes

    # TODO(andrey-churkin):
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
            self._state_names.OP_NAME: self.op_name
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> 'TFQuantizationPoint':
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        target_point_cls = TF_STATEFUL_CLASSES.get_registered_class(state[cls._state_names.TARGET_POINT_CLASS_NAME])
        kwargs = {
            cls._state_names.TARGET_POINT: target_point_cls.from_state(state[cls._state_names.TARGET_POINT]),
            cls._state_names.QUANTIZER_SPEC: TFQuantizerSpec.from_state(state[cls._state_names.QUANTIZER_SPEC]),
            cls._state_names.OP_NAME: state[cls._state_names.OP_NAME]
        }
        return cls(**kwargs)


class TFQuantizationSetupStateNames:
    QUANTIZATION_POINTS = 'quantization_points'
    UNIFIED_SCALE_GROUPS = 'unified_scale_groups'


class TFQuantizationSetup:
    """
    Characterizes where and how to insert all quantization nodes to the model's graph
    """

    _state_names = TFQuantizationSetupStateNames

    def __init__(self):
        super().__init__()
        self._quantization_points = []  # type: List[TFQuantizationPoint]
        self._unified_scale_groups = []  # type: List[List[QuantizationPointId]]

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
            self._state_names.UNIFIED_SCALE_GROUPS: self._unified_scale_groups
        }

    @classmethod
    def from_state(cls, state: Dict) -> 'TFQuantizationSetup':
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


def _is_half_range(disable_saturation_fix: bool,
                   target_device: str,
                   num_bits: int) -> bool:
    if not disable_saturation_fix:
        if target_device in ['CPU', 'ANY'] and num_bits == 8:
            return True
    return False


def _get_quantizer_op_name(prefix: str, is_wq: bool, port_id: int, target_type) -> str:
    pos = 'pre_hook' if target_type == TargetType.OPERATOR_PRE_HOOK else 'post_hook'
    qtype = 'W' if is_wq else 'A'
    name = '_'.join([pos, qtype, str(port_id)])
    quantizer_op_name = f'{prefix}/{name}'
    return quantizer_op_name


def _get_tensor_specs(node: NNCFNode,
                      nncf_graph: NNCFGraph,
                      port_ids: List[int],
                      is_input_tensors: bool,
                      is_weight_tensors: bool):
    """
    Returns specification of tensors for `node` according to `port_ids`.

    :param node:
    :param nncf_graph:
    :param port_ids:
    :param is_input_tensors:
    :param is_weight_tensors:
    """

    tensor_specs = []

    if is_weight_tensors:
        assert is_input_tensors
        metatype = node.metatype

        assert len(port_ids) == 1
        assert len(metatype.weight_definitions) == 1

        channel_axes = metatype.weight_definitions[0].channel_axes
        weight_shape = node.layer_attributes.get_weight_shape()
        tensor_specs.append((weight_shape, channel_axes))
    else:
        data_format = node.layer_attributes.get_data_format()
        channel_axes = [-1] if data_format == 'channels_last' else [1]

        if is_input_tensors:
            edges = nncf_graph.get_input_edges(node)
            for input_port_id in port_ids:
                tensor_specs.extend(
                    (e.tensor_shape, channel_axes) for e in edges if e.input_port_id == input_port_id
                )
        else:
            edges = nncf_graph.get_output_edges(node)
            for output_port_id in port_ids:
                filtered_edges = [e for e in edges if e.output_port_id == output_port_id]

                shape = filtered_edges[0].tensor_shape
                for e in filtered_edges:
                    assert e.tensor_shape == shape

                tensor_specs.append((shape, channel_axes))

    assert len(tensor_specs) == len(port_ids)

    return tensor_specs


def _create_quantization_setup(nncf_graph: NNCFGraph,
                               qp_solution: SingleConfigQuantizerSetup,
                               disable_saturation_fix: bool,
                               target_device: str) -> TFQuantizationSetup:
    quantization_setup = TFQuantizationSetup()
    node_name_to_qconfig_map = {}  # type: Dict[str, QuantizerConfig]
    qp_id_to_setup_index_map = {}  # type: Dict[QuantizationPointId, int]

    for idx, (qp_id, qp) in enumerate(qp_solution.quantization_points.items()):
        qp_id_to_setup_index_map[qp_id] = idx
        target_node = nncf_graph.get_node_by_name(qp.insertion_point.target_node_name)

        if qp.is_weight_quantization_point():
            # Check correctness.
            if target_node.node_name in node_name_to_qconfig_map:
                assigned_qconfig = node_name_to_qconfig_map[target_node.node_name]
                if qp.qconfig != assigned_qconfig:
                    raise RuntimeError('Inconsistent quantizer configurations selected by solver for one '
                                       f'and the same quantizable op! Tried to assign {qp.qconfig} to '
                                       f'{target_node.node_name} as specified by QP {qp_id}, but the op '
                                       f'already has quantizer config {assigned_qconfig} assigned to it!')
                continue  # The operation has already been quantized.
            node_name_to_qconfig_map[target_node.node_name] = qp.qconfig

            # Parameters.
            half_range = _is_half_range(disable_saturation_fix, target_device, qp.qconfig.num_bits)
            narrow_range = not half_range
            target_type = TargetType.OPERATOR_PRE_HOOK
            if not issubclass(target_node.metatype, TFOpWithWeightsMetatype):
                raise RuntimeError(f'Unexpected type of metatype: {type(target_node.metatype)}')
            port_ids = [weight_def.port_id for weight_def in target_node.metatype.weight_definitions]

        elif qp.is_activation_quantization_point():
            # Check correctness.
            if not isinstance(qp.insertion_point, ActivationQuantizationInsertionPoint):
                raise RuntimeError(f'Unexpected type of insertion point: {type(qp.insertion_point)}')

            # Parameters.
            half_range = False
            narrow_range = False
            if qp.insertion_point.input_port_id is not None:
                port_ids = [qp.insertion_point.input_port_id]  # Input port ids
                target_type = TargetType.OPERATOR_PRE_HOOK
            else:
                port_ids = [0]  # Output port ids
                target_type = TargetType.OPERATOR_POST_HOOK

            # TODO(andrey-churkin): Need refactoring
            # if target_node.node_type == 'Placeholder':
            #     edges = nncf_graph.get_output_edges(target_node)
            #     assert len(edges) == 1
            #     target_node = edges[0].to_node
            #     port_ids = [edges[0].input_port_id]
            #     target_type = TargetType.OPERATOR_PRE_HOOK

        else:
            raise RuntimeError('Unknown type of quantization point.')

        tensor_specs = _get_tensor_specs(
            target_node,
            nncf_graph,
            port_ids,
            target_type == TargetType.OPERATOR_PRE_HOOK,
            qp.is_weight_quantization_point()
        )

        for port_id, (tensor_shape, channel_axes) in zip(port_ids, tensor_specs):
            quantizer_op_name = _get_quantizer_op_name(
                target_node.node_name,
                qp.is_weight_quantization_point(),
                port_id,
                target_type
            )
            quantizer_spec = TFQuantizerSpec.from_config(qp.qconfig, narrow_range, half_range)
            target_point = TFTargetPoint(target_node.node_name, target_node.node_type, port_id, target_type)
            qpoint = TFQuantizationPoint(quantizer_op_name, quantizer_spec, target_point, qp.is_weight_quantization_point(), tensor_shape, channel_axes)

            quantization_setup.add_quantization_point(qpoint)

    # Registration of unified scale groups
    for unified_group in qp_solution.unified_scale_groups.values():
        us_group = [
            qp_id_to_setup_index_map[qp_id] for qp_id in unified_group
        ]
        quantization_setup.register_unified_scale_group(us_group)

    return quantization_setup


def _build_insertion_commands_for_quantization_setup(qsetup: TFQuantizationSetup) -> List[TFInsertionCommand]:
    """
    :param qsetup:
    :return:
    """
    insertion_commands = []
    quantization_points = qsetup.get_quantization_points()
    # quantization point id is her index inside the `quantization_points` list
    was_processed = {qp_id: False for qp_id in range(len(quantization_points))}

    for unified_scales_group in qsetup.get_unified_scale_groups():
        qp = quantization_points[unified_scales_group[0]]
        quantizer = create_quantizer(
            f'{qp.op_name}/unified_scale_group',
            qp.quantizer_spec,
            qp.is_weight_quantization,
            qp.input_shape,
            qp.channel_axes
        )

        for qp_id in unified_scales_group:
            if was_processed[qp_id]:
                raise RuntimeError('Unexpected behavior')
            was_processed[qp_id] = True

            curr_qp = quantization_points[qp_id]
            # Checks
            assert curr_qp.quantizer_spec.get_state() == qp.quantizer_spec.get_state()
            assert curr_qp.input_shape == qp.input_shape
            assert curr_qp.channel_axes == qp.channel_axes

            command = TFInsertionCommand(
                curr_qp.target_point,
                quantizer,
                TransformationPriority.QUANTIZATION_PRIORITY
            )
            insertion_commands.append(command)

    for qp_id, qp in enumerate(quantization_points):
        if was_processed[qp_id]:
            continue

        quantizer = create_quantizer(
            qp.op_name,
            qp.quantizer_spec,
            qp.is_weight_quantization,
            qp.input_shape,
            qp.channel_axes
        )
        command = TFInsertionCommand(
            qp.target_point,
            quantizer,
            TransformationPriority.QUANTIZATION_PRIORITY
        )
        insertion_commands.append(command)

    return insertion_commands


class QBuilderStateNames:
    QUANTIZER_SETUP = 'quantizer_setup'


class QuantizationBuilder(TFCompressionAlgorithmBuilder):
    """
    Builder of quantization algorithm.
    """

    _state_names = QBuilderStateNames

    DEFAULT_QCONFIG = QuantizerConfig(num_bits=8,
                                      mode=QuantizationMode.SYMMETRIC,
                                      signedness_to_force=None,
                                      per_channel=False)

    name = 'quantization'

    def __init__(self, config: NNCFConfig, should_init: bool = True):
        super().__init__(config, should_init)

        self.quantize_inputs = self._algo_config.get('quantize_inputs', True)
        self.quantize_outputs = self._algo_config.get('quantize_outputs', False)
        self._disable_saturation_fix = self._algo_config.get('disable_saturation_fix', False)
        self._target_device = config.get('target_device', 'ANY')

        if self._target_device == 'VPU' and 'preset' in self._algo_config:
            raise RuntimeError("The VPU target device does not support presets.")

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
        self._quantization_setup = None

        self.hw_config = None
        if self._target_device != "TRIAL":
            hw_config_type = HWConfigType.from_str(HW_CONFIG_TYPE_TARGET_DEVICE_MAP[self._target_device])
            hw_config_path = TFHWConfig.get_path_to_hw_config(hw_config_type)
            self.hw_config = TFHWConfig.from_json(hw_config_path)

    def _parse_group_params(self, quant_config: Dict, quantizer_group: QuantizerGroup) -> None:
        group_name = quantizer_group.value
        params_dict = {}
        params_dict_from_config = quant_config.get(group_name, {})
        if self._target_device != 'VPU':
            preset = QuantizationPreset.from_str(quant_config.get('preset', 'performance'))
            params_dict = preset.get_params_configured_by_preset(quantizer_group)
            overrided_params = params_dict.keys() & params_dict_from_config.keys()
            if overrided_params:
                logger.warning('Preset quantizer parameters {} explicitly overrided.'.format(overrided_params))
        params_dict.update(params_dict_from_config)
        self.global_quantizer_constraints[quantizer_group] = QuantizationConstraints.from_config_dict(params_dict)
        self.ignored_scopes_per_group[quantizer_group] = params_dict_from_config.get('ignored_scopes', [])
        if self.ignored_scopes is not None:
            self.ignored_scopes_per_group[quantizer_group] += self.ignored_scopes
        target_scopes = params_dict_from_config.get('target_scopes')
        if target_scopes is None and self.target_scopes is not None:
            self.target_scopes_per_group[quantizer_group] = self.target_scopes
        else:
            self.target_scopes_per_group[quantizer_group] = target_scopes

    def get_quantization_setup(self, nncf_network: NNCFNetwork) -> TFQuantizationSetup:
        nncf_graph = convert_nncf_network_to_nncf_graph(nncf_network)

        # Find out which metatypes unsupported by the quantization algorithm
        for node in nncf_graph.get_all_nodes():
            if node.metatype in UNSUPPORTED_TF_OP_METATYPES:
                logger.warning(
                    'The operation {} is unsupported by the quantization algorithm.'.format(node.node_name)
                )

        qp_solution = self._get_quantizer_propagation_solution(nncf_graph)

        quantization_setup = _create_quantization_setup(
            nncf_graph,
            qp_solution,
            self._disable_saturation_fix,
            self._target_device
        )

        return quantization_setup

    def _get_quantizer_propagation_solution(self, nncf_graph: NNCFGraph) -> SingleConfigQuantizerSetup:
        # type: List[QuantizableWeightedLayerNode]
        possible_qconfigs_for_nodes_with_weight = self._get_possible_qconfigs_for_nodes_with_weight(nncf_graph)

        ip_graph = InsertionPointGraph(
            nncf_graph,
            [qn.node.node_name for qn in possible_qconfigs_for_nodes_with_weight]
        )
        pattern = TF_HW_FUSED_PATTERNS.get_full_pattern_graph()
        ip_graph = ip_graph.get_ip_graph_with_merged_hw_optimized_operations(pattern)

        ignored_scopes = self.ignored_scopes_per_group[QuantizerGroup.ACTIVATIONS]

        solver = QuantizerPropagationSolver(
            ignored_scopes=ignored_scopes,
            target_scopes=self.target_scopes_per_group[QuantizerGroup.ACTIVATIONS],
            hw_config=self.hw_config,
            default_trait_to_metatype_map=DEFAULT_TF_QUANT_TRAIT_TO_OP_DICT,
            default_qconfig_list=[
                self._get_default_qconfig(
                    self.global_quantizer_constraints[QuantizerGroup.ACTIVATIONS]
                )
            ],
            quantizable_layer_nodes=possible_qconfigs_for_nodes_with_weight,
            global_constraints=self.global_quantizer_constraints,
            quantize_outputs=self.quantize_outputs
        )

        quantization_proposal = solver.run_on_ip_graph(ip_graph)
        multi_config_setup = quantization_proposal.quantizer_setup
        single_config_setup = multi_config_setup.select_first_qconfig_for_each_point()
        finalized_proposal = quantization_proposal.finalize(single_config_setup)
        final_setup = solver.get_final_quantizer_setup(finalized_proposal)
        final_setup = self._handle_quantize_inputs_option(final_setup, nncf_graph)

        return final_setup

    def _get_default_qconfig(self,
                             constraints: Optional[QuantizationConstraints] = None) -> QuantizerConfig:
        qconfig = deepcopy(self.DEFAULT_QCONFIG)
        if constraints is not None:
            qconfig = constraints.apply_constraints_to(qconfig)
        return qconfig

    def _handle_quantize_inputs_option(self,
                                       quantizer_setup: SingleConfigQuantizerSetup,
                                       nncf_graph: NNCFGraph) -> SingleConfigQuantizerSetup:
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

    def _get_possible_qconfigs_for_nodes_with_weight(self, nncf_graph: NNCFGraph) -> List[QuantizableWeightedLayerNode]:
        """
        Returns possible configurations of quantizer for nodes with weights.

        :param nncf_graph: The NNCF graph.
        :return: Possible configurations of quantizer for nodes with weights.
        """

        # NOTE: the `QuantizableWeightedLayerNode` name confuses.
        # Actually, it is the pair (NNCFNode, Possible configurations of quantizer).

        nodes_with_weights = []
        for node in nncf_graph.get_all_nodes():
            if node.metatype in OUTPUT_NOOP_METATYPES:
                continue

            # `ALL_TF_OP_METATYPES_WITH_WEIGHTS` is a metatypes with weight
            if not (node.metatype in ALL_TF_OP_METATYPES_WITH_WEIGHTS
                    and should_consider_scope(node.node_name,
                                              ignored_scopes=self.ignored_scopes_per_group[QuantizerGroup.WEIGHTS],
                                              target_scopes=None)):
                continue

            if not issubclass(node.metatype, TFOpWithWeightsMetatype):
                raise RuntimeError(f'Unexpected type of metatype: {type(node.metatype)}')

            nodes_with_weights.append(node)

        scope_overrides = self._get_algo_specific_config_section().get('scope_overrides', {})

        possible_qconfigs_for_nodes = assign_qconfig_lists_to_modules(
            nodes_with_weights,
            self.DEFAULT_QCONFIG,
            self.global_quantizer_constraints[QuantizerGroup.WEIGHTS],
            scope_overrides,
            self.hw_config
        )

        return [QuantizableWeightedLayerNode(*item) for item in possible_qconfigs_for_nodes.items()]

    def _build_controller(self, model):
        return QuantizationControllerV2(model, self.config, self._op_names)

    def get_transformation_layout(self, model):
        transformation_layout = TFTransformationLayout()

        if self._quantization_setup is None:
            self._quantization_setup = self.get_quantization_setup(model)

        insertion_commands = _build_insertion_commands_for_quantization_setup(self._quantization_setup)
        for command in insertion_commands:
            transformation_layout.register(command)

            for op in command.insertion_objects:
                self._op_names.append(op.name)

        return transformation_layout

    def apply_to(self, model: NNCFNetwork) -> NNCFNetwork:
        """
        Applies algorithm-specific modifications to the model.

        :param model: The original uncompressed model.
        :return: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        """
        transformation_layout = self.get_transformation_layout(model)
        transformer = TFModelTransformer(model)
        transformed_model = transformer.transform(transformation_layout)

        if self.should_init:
            self.initialize(transformed_model)

        return transformed_model

    def initialize(self, model: NNCFNetwork) -> None:
        if self._range_init_params is not None:
            self._run_range_initialization(model)
        self._run_batchnorm_adaptation(model)

    def _run_range_initialization(self, model: NNCFNetwork) -> None:
        if self._range_initializer is None:
            self._range_initializer = RangeInitializerV2(self._range_init_params)
        self._range_initializer.run(model)

    def _run_batchnorm_adaptation(self, model: NNCFNetwork) -> None:
        if self._bn_adaptation is None:
            self._bn_adaptation = BatchnormAdaptationAlgorithm(
                **extract_bn_adaptation_init_params(self.config, self.name))
        self._bn_adaptation.run(model)

    def _parse_init_params(self):
        self._range_init_params = self._parse_range_init_params()

    def _parse_range_init_params(self) -> TFRangeInitParamsV2:
        range_init_params = extract_range_init_params(self.config)
        return TFRangeInitParamsV2(**range_init_params) if range_init_params is not None else None


class QuantizationControllerV2(QuantizationController):
    def strip_model(self, model: NNCFNetwork) -> NNCFNetwork:
        return model

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        return NNCFStatistics()
