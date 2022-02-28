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

from typing import List, Optional, Dict, Any

from nncf.common.utils.logger import logger
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.utils import get_first_nodes_of_type
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.quantizer_setup import QuantizationPointId
from nncf.common.quantization.quantizer_setup import ActivationQuantizationInsertionPoint
from nncf.common.stateful_classes_registry import TF_STATEFUL_CLASSES
from nncf.common.statistics import NNCFStatistics
from nncf.config.extractors import extract_range_init_params
from nncf.tensorflow.algorithm_selector import TF_COMPRESSION_ALGORITHMS
from nncf.tensorflow.graph.transformations.commands import TFInsertionCommand
from nncf.tensorflow.graph.metatypes.tf_ops import TFOpWithWeightsMetatype
from nncf.tensorflow.quantization.quantizers import TFQuantizerSpec
from nncf.tensorflow.quantization.algorithm import QuantizationController
from nncf.tensorflow.quantization.algorithm import TFQuantizationPointStateNames
from nncf.tensorflow.quantization.algorithm import TFQuantizationPoint
from nncf.tensorflow.quantization.algorithm import TFQuantizationSetup
from nncf.tensorflow.quantization.algorithm import QuantizationBuilder
from nncf.experimental.tensorflow.nncf_network import NNCFNetwork
from nncf.experimental.tensorflow.graph.converter import SubclassedConverter
from nncf.experimental.tensorflow.graph.model_transformer import TFModelTransformerV2
from nncf.experimental.tensorflow.graph.transformations.layout import TFTransformationLayoutV2
from nncf.experimental.tensorflow.quantization.init_range import TFRangeInitParamsV2
from nncf.experimental.tensorflow.quantization.init_range import RangeInitializerV2
from nncf.experimental.tensorflow.quantization.quantizers import create_quantizer
from nncf.experimental.tensorflow.graph.transformations.commands import TFTargetPoint


UNSUPPORTED_TF_OP_METATYPES = [
]


class TFQuantizationPointV2StateNames(TFQuantizationPointStateNames):
    IS_WEIGHT_QUANTIZATION = 'is_weight_quantization'
    INPUT_SHAPE = 'input_shape'
    CHANNEL_AXES = 'channel_axes'


class TFQuantizationPointV2(TFQuantizationPoint):

    _state_names = TFQuantizationPointV2StateNames

    def __init__(self,
                 op_name: str,
                 quantizer_spec: TFQuantizerSpec,
                 target_point: TargetPoint,
                 is_weight_quantization: bool,
                 input_shape: Optional[List[int]] = None,
                 channel_axes: Optional[List[int]] = None):
        super().__init__(op_name, quantizer_spec, target_point)
        self.is_weight_quantization = is_weight_quantization
        self.input_shape = input_shape
        self.channel_axes = channel_axes

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update(
            {
                self._state_names.IS_WEIGHT_QUANTIZATION: self.is_weight_quantization,
                self._state_names.INPUT_SHAPE: self.input_shape,
                self._state_names.CHANNEL_AXES: self.channel_axes,
            }
        )
        return state

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> 'TFQuantizationPointV2':
        target_point_cls = TF_STATEFUL_CLASSES.get_registered_class(state[cls._state_names.TARGET_POINT_CLASS_NAME])
        kwargs = {
            cls._state_names.OP_NAME: state[cls._state_names.OP_NAME],
            cls._state_names.QUANTIZER_SPEC: TFQuantizerSpec.from_state(state[cls._state_names.QUANTIZER_SPEC]),
            cls._state_names.TARGET_POINT: target_point_cls.from_state(state[cls._state_names.TARGET_POINT]),
            cls._state_names.IS_WEIGHT_QUANTIZATION: state[cls._state_names.IS_WEIGHT_QUANTIZATION],
            cls._state_names.INPUT_SHAPE: state[cls._state_names.INPUT_SHAPE],
            cls._state_names.CHANNEL_AXES: state[cls._state_names.CHANNEL_AXES],
        }
        return cls(**kwargs)


class TFQuantizationSetupV2(TFQuantizationSetup):

    @classmethod
    def from_state(cls, state: Dict) -> 'TFQuantizationSetupV2':
        setup = TFQuantizationSetupV2()
        for quantization_point_state in state[cls._state_names.QUANTIZATION_POINTS]:
            quantization_point = TFQuantizationPointV2.from_state(quantization_point_state)
            setup.add_quantization_point(quantization_point)

        if cls._state_names.UNIFIED_SCALE_GROUPS in state:
            for quantization_group in state[cls._state_names.UNIFIED_SCALE_GROUPS]:
                setup.register_unified_scale_group(quantization_group)
        return setup


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


@TF_COMPRESSION_ALGORITHMS.register('experimental_quantization')
class QuantizationBuilderV2(QuantizationBuilder):

    def _load_state_without_name(self, state_without_name: Dict[str, Any]):
        quantizer_setup_state = state_without_name[self._state_names.QUANTIZER_SETUP]
        self._quantizer_setup = TFQuantizationSetupV2.from_state(quantizer_setup_state)

    def _parse_range_init_params(self) -> TFRangeInitParamsV2:
        range_init_params = extract_range_init_params(self.config, self.name)
        return TFRangeInitParamsV2(**range_init_params) if range_init_params is not None else None

    def _build_insertion_commands_for_quantizer_setup(self, quantizer_setup: TFQuantizationSetupV2) \
             -> List[TFInsertionCommand]:
        insertion_commands = []
        quantization_points = quantizer_setup.get_quantization_points()
        # quantization point id is her index inside the `quantization_points` list
        was_processed = {qp_id: False for qp_id in range(len(quantization_points))}

        for unified_scales_group in quantizer_setup.get_unified_scale_groups():
            qp = quantization_points[unified_scales_group[0]]
            quantizer = create_quantizer(
                f'{qp.op_name}/unified_scale_group',
                qp.quantizer_spec,
                qp.is_weight_quantization,
                qp.input_shape,
                qp.channel_axes
            )

            self._op_names.append(quantizer.name)

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

            self._op_names.append(quantizer.name)

            command = TFInsertionCommand(
                qp.target_point,
                quantizer,
                TransformationPriority.QUANTIZATION_PRIORITY
            )
            insertion_commands.append(command)

        return insertion_commands

    def get_transformation_layout(self, model: NNCFNetwork):
        transformations = TFTransformationLayoutV2()
        if self._quantizer_setup is None:
            self._quantizer_setup = self._get_quantizer_setup(model)
        insertion_commands = self._build_insertion_commands_for_quantizer_setup(self._quantizer_setup)
        for command in insertion_commands:
            transformations.register(command)
        return transformations

    def _build_controller(self, model: NNCFNetwork) -> 'QuantizationControllerV2':
        return QuantizationControllerV2(model, self.config, self._op_names)

    def _run_range_initialization(self, model: NNCFNetwork) -> None:
        if self._range_initializer is None:
            self._range_initializer = RangeInitializerV2(self._range_init_params)
        self._range_initializer.run(model)

    def _get_input_preprocessing_nodes(self, nncf_graph: NNCFGraph, model: NNCFNetwork) -> List[NNCFNode]:
        return []

    def _get_quantizer_setup(self, model: NNCFNetwork) -> TFQuantizationSetupV2:
        converter = SubclassedConverter(model, model.input_signature)
        nncf_graph = converter.convert()
        # Find out which metatypes unsupported by the quantization algorithm
        for node in nncf_graph.get_all_nodes():
            if node.metatype in UNSUPPORTED_TF_OP_METATYPES:
                logger.warning(
                    'The operation {} is unsupported by the quantization algorithm.'.format(node.node_name)
                )

        # Possible configurations of quantizer for nodes with weights.
        possible_qconfigs_for_nodes_with_weight = self._get_quantizable_weighted_layer_nodes(nncf_graph)
        qp_solution = self._get_quantizer_propagation_solution(nncf_graph,
                                                               possible_qconfigs_for_nodes_with_weight,
                                                               [],
                                                               model)

        # Logic of the TFQuantizationSetupV2 creation

        quantization_setup = TFQuantizationSetupV2()
        node_name_to_qconfig_map = {}  # type: Dict[str, QuantizerConfig]
        qp_id_to_setup_index_map = {}  # type: Dict[QuantizationPointId, int]
        first_conv_nodes = get_first_nodes_of_type(nncf_graph, ['Conv2D'])

        for idx, (qp_id, qp) in enumerate(qp_solution.quantization_points.items()):
            qp_id_to_setup_index_map[qp_id] = idx
            target_node = nncf_graph.get_node_by_name(qp.insertion_point.target_node_name)

            if qp.is_weight_quantization_point():
                # Check correctness
                if target_node.node_name in node_name_to_qconfig_map:
                    assigned_qconfig = node_name_to_qconfig_map[target_node.node_name]
                    if qp.qconfig != assigned_qconfig:
                        raise RuntimeError('Inconsistent quantizer configurations selected by solver for one '
                                            f'and the same quantizable op! Tried to assign {qp.qconfig} to '
                                            f'{target_node.node_name} as specified by QP {qp_id}, but the op '
                                            f'already has quantizer config {assigned_qconfig} assigned to it!')
                    continue  # The operation has already been quantized
                node_name_to_qconfig_map[target_node.node_name] = qp.qconfig

                # Parameters
                half_range = self._get_half_range(qp.qconfig, target_node, first_conv_nodes)
                narrow_range = not half_range
                target_type = TargetType.OPERATOR_PRE_HOOK
                if not issubclass(target_node.metatype, TFOpWithWeightsMetatype):
                    raise RuntimeError(f'Unexpected type of metatype: {type(target_node.metatype)}')
                port_ids = [weight_def.port_id for weight_def in target_node.metatype.weight_definitions]

            else:
                assert qp.is_activation_quantization_point()

                # Check correctness
                if not isinstance(qp.insertion_point, ActivationQuantizationInsertionPoint):
                    raise RuntimeError(f'Unexpected type of insertion point: {type(qp.insertion_point)}')

                # Parameters
                half_range = False
                narrow_range = False
                if qp.insertion_point.input_port_id is not None:
                    port_ids = [qp.insertion_point.input_port_id]  # Input port ids
                    target_type = TargetType.OPERATOR_PRE_HOOK
                else:
                    port_ids = [0]  # Output port ids
                    target_type = TargetType.OPERATOR_POST_HOOK

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
                qpoint = TFQuantizationPointV2(quantizer_op_name, quantizer_spec, target_point,
                                               qp.is_weight_quantization_point(), tensor_shape, channel_axes)

                quantization_setup.add_quantization_point(qpoint)

        # Registration of unified scale groups
        for unified_group in qp_solution.unified_scale_groups.values():
            us_group = [
                qp_id_to_setup_index_map[qp_id] for qp_id in unified_group
            ]
            quantization_setup.register_unified_scale_group(us_group)

        return quantization_setup

    def apply_to(self, model: NNCFNetwork) -> NNCFNetwork:
        transformation_layout = self.get_transformation_layout(model)
        transformer = TFModelTransformerV2(model)
        transformed_model = transformer.transform(transformation_layout)

        if self.should_init:
            self.initialize(transformed_model)

        return transformed_model


class QuantizationControllerV2(QuantizationController):
    def strip_model(self, model: NNCFNetwork) -> NNCFNetwork:
        return model

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        return NNCFStatistics()

    def prepare_for_export(self) -> None:
        self._model.compute_output_shape(self._model.input_signature.shape.as_list())
        self._model = self.strip_model(self._model)
