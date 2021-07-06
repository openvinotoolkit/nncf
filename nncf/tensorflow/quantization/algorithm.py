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
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import tensorflow as tf

from nncf import NNCFConfig
from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.common.compression import BaseCompressionAlgorithmController
from nncf.common.graph import INPUT_NOOP_METATYPES
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.common.graph import OUTPUT_NOOP_METATYPES
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver
from nncf.common.quantization.quantizer_setup import ActivationQuantizationInsertionPoint
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.structs import QuantizationConstraints
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.stateful_classes_registry import TF_STATEFUL_CLASSES
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.helpers import should_consider_scope
from nncf.common.utils.logger import logger
from nncf.config.extractors import extract_bn_adaptation_init_params
from nncf.config.extractors import extract_range_init_params
from nncf.tensorflow.algorithm_selector import TF_COMPRESSION_ALGORITHMS
from nncf.tensorflow.api.compression import TFCompressionAlgorithmBuilder
from nncf.tensorflow.graph.converter import TFModelConverter
from nncf.tensorflow.graph.converter import TFModelConverterFactory
from nncf.tensorflow.graph.metatypes.common import ELEMENTWISE_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.common import GENERAL_CONV_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.common import LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION
from nncf.tensorflow.graph.metatypes.common import LINEAR_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.keras_layers import TFLambdaLayerMetatype
from nncf.tensorflow.graph.metatypes.keras_layers import TFLayerWithWeightsMetatype
from nncf.tensorflow.graph.transformations.commands import TFAfterLayer
from nncf.tensorflow.graph.transformations.commands import TFBeforeLayer
from nncf.tensorflow.graph.transformations.commands import TFInsertionCommand
from nncf.tensorflow.graph.transformations.commands import TFLayerWeight
from nncf.tensorflow.graph.transformations.layout import TFTransformationLayout
from nncf.tensorflow.graph.utils import get_original_name_and_instance_idx
from nncf.tensorflow.hardware.fused_patterns import TF_HW_FUSED_PATTERNS
from nncf.tensorflow.layers.custom_objects import NNCF_QUANTIZATION_OPERATONS
from nncf.tensorflow.loss import TFZeroCompressionLoss
from nncf.tensorflow.quantization.default_quantization import DEFAULT_TF_QUANT_TRAIT_TO_OP_DICT
from nncf.tensorflow.quantization.initializers.init_range import RangeInitializer
from nncf.tensorflow.quantization.initializers.init_range import TFRangeInitParams
from nncf.tensorflow.quantization.layers import FakeQuantize
from nncf.tensorflow.quantization.quantizers import Quantizer
from nncf.tensorflow.quantization.quantizers import TFQuantizerSpec
from nncf.tensorflow.quantization.utils import apply_saturation_fix
from nncf.tensorflow.quantization.collectors import TFQuantizationStatisticsCollector

QUANTIZATION_LAYER_METATYPES = GENERAL_CONV_LAYER_METATYPES + LINEAR_LAYER_METATYPES

NOT_SUPPORT_LAYER_METATYPES = [
    TFLambdaLayerMetatype
]


class QuantizationPointStateNames:
    QUANTIZER_SPEC = 'quantizer_spec'
    TARGET_POINT = 'target_point'
    TARGET_POINT_CLASS_NAME = 'target_point_class_name'
    OP_NAME = 'op_name'


class QuantizationPoint:
    """
    Characterizes where and how to insert a single quantization node to the model's graph
    """
    _state_names = QuantizationPointStateNames

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
            self._state_names.OP_NAME: self.op_name
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> 'QuantizationPoint':
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


class QBuilderStateNames:
    QUANTIZER_SETUP = 'quantizer_setup'


@TF_COMPRESSION_ALGORITHMS.register('quantization')
class QuantizationBuilder(TFCompressionAlgorithmBuilder):
    _state_names = QBuilderStateNames

    def __init__(self, config: NNCFConfig, should_init: bool = True):
        super().__init__(config, should_init)

        self.quantize_inputs = self._algo_config.get('quantize_inputs', True)
        self.quantize_outputs = self._algo_config.get('quantize_outputs', False)
        self._disable_saturation_fix = self._algo_config.get('disable_saturation_fix', False)
        self._target_device = config.get('target_device', 'ANY')

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

    def _load_state_without_name(self, state_without_name: Dict[str, Any]):
        """
        Initializes object from the state.

        :param state_without_name: Output of `get_state()` method.
        """
        quantizer_setup_state = state_without_name[self._state_names.QUANTIZER_SETUP]
        self._quantizer_setup = QuantizationSetup.from_state(quantizer_setup_state)

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

    def _parse_range_init_params(self) -> TFRangeInitParams:
        range_init_params = extract_range_init_params(self.config)
        return TFRangeInitParams(**range_init_params) if range_init_params is not None else None

    def _parse_group_params(self, quant_config: Dict, quantizer_group: QuantizerGroup) -> None:
        group_name = quantizer_group.value
        params_dict = quant_config.get(group_name, {})
        self.global_quantizer_constraints[quantizer_group] = QuantizationConstraints.from_config_dict(params_dict)
        self.ignored_scopes_per_group[quantizer_group] = params_dict.get('ignored_scopes', [])
        if self.ignored_scopes is not None:
            self.ignored_scopes_per_group[quantizer_group] += self.ignored_scopes
        target_scopes = params_dict.get('target_scopes')
        if target_scopes is None and self.target_scopes is not None:
            self.target_scopes_per_group[quantizer_group] = self.target_scopes
        else:
            self.target_scopes_per_group[quantizer_group] = target_scopes

    def _get_default_qconfig(self, constraints: QuantizationConstraints = None) -> QuantizerConfig:
        qconfig = QuantizerConfig(num_bits=8,
                                  mode=QuantizationMode.SYMMETRIC,
                                  signedness_to_force=None,
                                  per_channel=False)
        if constraints is not None:
            qconfig = constraints.apply_constraints_to(qconfig)
        return qconfig

    def _get_half_range(self, qconfig: QuantizerConfig) -> bool:
        if not self._disable_saturation_fix:
            if self._target_device in ['CPU', 'ANY'] and qconfig.num_bits == 8:
                logger.warning('A saturation issue fix will be applied. '
                               'Now all weight quantizers will effectively use only 7 bits out of 8 bits. '
                               'This resolves the saturation issue problem on AVX2 and AVX-512 machines. '
                               'Please take a look at the documentation for a detailed information.')
                return True
        return False

    def _create_quantizer(self, name: str, qspec: TFQuantizerSpec) -> Quantizer:
        quantizer_cls = NNCF_QUANTIZATION_OPERATONS.get(qspec.mode)
        return quantizer_cls(name, qspec)

    def get_transformation_layout(self, model: tf.keras.Model) -> TFTransformationLayout:
        converter = TFModelConverterFactory.create(model)
        nncf_graph = converter.convert()
        nodes = nncf_graph.get_all_nodes()
        for node in nodes:
            if node.metatype in NOT_SUPPORT_LAYER_METATYPES:
                logger.warning('The layer {} is not supported by the quantization algorithm'
                               .format(get_original_name_and_instance_idx(node.node_name)[0]))

        transformations = TFTransformationLayout()

        quantizable_weighted_layer_nodes = self._get_quantizable_weighted_layer_nodes(nncf_graph)
        custom_layer_nodes = self._get_custom_layer_node_names(nncf_graph, converter)
        if self._quantizer_setup is None:
            self._quantizer_setup = self._get_quantizer_setup(nncf_graph, quantizable_weighted_layer_nodes,
                                                              custom_layer_nodes)

        quantized_layer_names_vs_qconfigs = {}  # type: Dict[str, QuantizerConfig]
        for qp_id, qp in self._quantizer_setup.quantization_points.items():
            if qp.is_weight_quantization_point():
                target_node = nncf_graph.get_node_by_name(qp.insertion_point.target_node_name)
                is_custom, layer_info = converter.get_layer_info_for_node(target_node.node_name)
                if is_custom:
                    raise RuntimeError("Quantizing custom layer weights is currently unsupported!")
                layer_name = layer_info.layer_name
                qconfig = qp.qconfig
                if layer_name in quantized_layer_names_vs_qconfigs:
                    assigned_qconfig = quantized_layer_names_vs_qconfigs[layer_name]
                    if qconfig != assigned_qconfig:
                        raise RuntimeError(f"Inconsistent quantizer configurations selected by solver for one and the "
                                           f"same quantizable layer! Tried to assign {qconfig} to {layer_name} as "
                                           f"specified by QP {qp_id}, but the layer already has quantizer "
                                           f"config {assigned_qconfig} assigned to it!")
                    continue  # The layer has already been quantized
                quantized_layer_names_vs_qconfigs[layer_name] = qconfig
                metatype = target_node.metatype
                assert issubclass(metatype, TFLayerWithWeightsMetatype)
                for weight_def in metatype.weight_definitions:
                    op_name = self._get_quantizer_operation_name(
                        target_node.node_name,
                        weight_def.weight_attr_name)
                    self._op_names.append(op_name)

                    half_range = self._get_half_range(qconfig)
                    operation = self._create_quantizer(
                        op_name,
                        TFQuantizerSpec.from_config(qconfig,
                                                    narrow_range=not half_range,
                                                    half_range=half_range))

                    transformations.register(
                        TFInsertionCommand(
                            target_point=TFLayerWeight(layer_info.layer_name, weight_def.weight_attr_name),
                            callable_object=operation,
                            priority=TransformationPriority.QUANTIZATION_PRIORITY))
            else:
                assert qp.is_activation_quantization_point()
                ip = qp.insertion_point
                assert isinstance(ip, ActivationQuantizationInsertionPoint)
                target_node_name = ip.target_node_name
                input_port_id = ip.input_port_id
                fake_quantize_name = self._get_fake_quantize_name(target_node_name, input_port_id)
                fake_quantize_layer = FakeQuantize(
                    TFQuantizerSpec.from_config(qp.qconfig, narrow_range=False, half_range=False),
                    name=fake_quantize_name)
                self._op_names.append(fake_quantize_layer.op_name)

                is_custom, layer_info = converter.get_layer_info_for_node(target_node_name)
                if is_custom:
                    raise RuntimeError("Quantizing custom layer activations is currently unsupported!")
                if input_port_id is not None:
                    tp = TFBeforeLayer(layer_info.layer_name,
                                       instance_idx=layer_info.instance_idx,
                                       input_port_id=input_port_id)
                else:
                    tp = TFAfterLayer(layer_info.layer_name,
                                      instance_idx=layer_info.instance_idx,
                                      output_port_id=0)
                transformations.register(TFInsertionCommand(
                        target_point=tp,
                        callable_object=fake_quantize_layer,
                        priority=TransformationPriority.QUANTIZATION_PRIORITY))

        return transformations

    def _get_quantizable_weighted_layer_nodes(self, nncf_graph: NNCFGraph) -> List[QuantizableWeightedLayerNode]:
        retval = []
        w_qconfig = self._get_default_qconfig(self.global_quantizer_constraints[QuantizerGroup.WEIGHTS])
        for node in nncf_graph.get_all_nodes():
            metatype = node.metatype
            if metatype in OUTPUT_NOOP_METATYPES:
                continue

            if not (metatype in QUANTIZATION_LAYER_METATYPES
                    and should_consider_scope(node.node_name,
                                              ignored_scopes=self.ignored_scopes_per_group[QuantizerGroup.WEIGHTS],
                                              target_scopes=None)):
                continue

            assert issubclass(metatype, TFLayerWithWeightsMetatype)
            retval.append(QuantizableWeightedLayerNode(node,
                                                       [w_qconfig]))
        return retval

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

    def _build_controller(self, model: tf.keras.Model) -> 'QuantizationController':
        return QuantizationController(model, self.config, self._op_names)

    def initialize(self, model: tf.keras.Model) -> None:
        if self._range_init_params is not None:
            self._run_range_initialization(model)
        self._run_batchnorm_adaptation(model)

    def _run_range_initialization(self, model: tf.keras.Model) -> None:
        if self._range_initializer is None:
            self._range_initializer = RangeInitializer(self._range_init_params)
        self._range_initializer.run(model)

    def _run_batchnorm_adaptation(self, model: tf.keras.Model) -> None:
        if self._bn_adaptation is None:
            self._bn_adaptation = BatchnormAdaptationAlgorithm(
                **extract_bn_adaptation_init_params(self.config, self.name))
        self._bn_adaptation.run(model)

    def _get_quantizer_setup(self, nncf_graph: NNCFGraph,
                             quantizable_weighted_layer_nodes: List[QuantizableWeightedLayerNode],
                             custom_layer_node_names: List[NNCFNodeName]) \
            -> SingleConfigQuantizerSetup:
        ip_graph = InsertionPointGraph(nncf_graph,
                                       [qn.node.node_name for qn in quantizable_weighted_layer_nodes])

        pattern = TF_HW_FUSED_PATTERNS.get_full_pattern_graph()
        ip_graph = ip_graph.get_ip_graph_with_merged_hw_optimized_operations(pattern)

        input_preprocessing_nodes = self._get_input_preprocessing_nodes(nncf_graph)
        input_preprocessing_node_names = [n.node_name for n in input_preprocessing_nodes]
        ignored_scopes_for_solver = self.ignored_scopes_per_group[QuantizerGroup.ACTIVATIONS] + \
                                    input_preprocessing_node_names + custom_layer_node_names

        solver = QuantizerPropagationSolver(
            ignored_scopes=ignored_scopes_for_solver,
            target_scopes=self.target_scopes_per_group[QuantizerGroup.ACTIVATIONS],
            hw_config=None,
            default_trait_to_metatype_map=DEFAULT_TF_QUANT_TRAIT_TO_OP_DICT,
            default_qconfig_list=[self._get_default_qconfig(
                self.global_quantizer_constraints[QuantizerGroup.ACTIVATIONS])],
            quantizable_layer_nodes=quantizable_weighted_layer_nodes,
            global_constraints=self.global_quantizer_constraints,
            quantize_outputs=self.quantize_outputs)

        quantization_proposal = solver.run_on_ip_graph(ip_graph)
        multi_config_setup = quantization_proposal.quantizer_setup
        single_config_setup = multi_config_setup.select_first_qconfig_for_each_point()
        finalized_proposal = quantization_proposal.finalize(single_config_setup)
        final_setup = solver.get_final_quantizer_setup(finalized_proposal)
        final_setup = self._handle_quantize_inputs_option(final_setup, nncf_graph)

        return final_setup

    def _handle_quantize_inputs_option(self, quantizer_setup: SingleConfigQuantizerSetup,
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

    def _get_input_preprocessing_nodes(self, nncf_graph: NNCFGraph) -> List[NNCFNode]:
        retval = []

        def traverse_fn(node: NNCFNode, preprocessing_nodes: List[NNCFNode]) -> Tuple[bool, List[NNCFNode]]:
            is_finished = True
            successors = nncf_graph.get_next_nodes(node)
            if len(successors) == 1:
                successor = next(iter(successors))
                if successor.metatype in ELEMENTWISE_LAYER_METATYPES and len(
                        nncf_graph.get_previous_nodes(successor)) == 1:
                    preprocessing_nodes.append(successor)
                    is_finished = False
            return is_finished, preprocessing_nodes

        for nncf_node in nncf_graph.get_input_nodes():
            preprocessing_nodes_for_this_input = nncf_graph.traverse_graph(nncf_node, traverse_fn)
            retval += preprocessing_nodes_for_this_input

        return retval

    def _get_quantized_nodes_for_output(self, nncf_graph: NNCFGraph,
                                        insertion_points: List[str],
                                        node_key: str,
                                        quantized_nodes_for_output: List[NNCFNode] = None) -> List[NNCFNode]:
        nncf_node = nncf_graph.get_node_by_key(node_key)
        if quantized_nodes_for_output is None:
            if node_key in insertion_points:
                return [nncf_node]
            quantized_nodes_for_output = []

        for predecessor in nncf_graph.get_previous_nodes(nncf_node):
            pred_node_key = nncf_graph.get_node_key_by_id(predecessor.node_id)
            if len(nncf_graph.get_next_nodes(predecessor)) > 1:
                logger.warning('Removing of FakeQuantize after layer {} '
                               'with multiple outputs is not fully supported'.format(predecessor.node_name))
            if predecessor.metatype in LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION:
                self._get_quantized_nodes_for_output(nncf_graph, insertion_points,
                                                     pred_node_key, quantized_nodes_for_output)
            elif nncf_graph.get_node_key_by_id(predecessor.node_id) in insertion_points:
                quantized_nodes_for_output.append(predecessor)
        return quantized_nodes_for_output

    def _get_fake_quantize_name(self, node_name: NNCFNodeName, input_port_id: int = None) -> str:
        original_node_name, instance_idx = get_original_name_and_instance_idx(node_name)
        fq_name = '{}/fake_quantize'.format(original_node_name)
        if instance_idx != 0:
            fq_name += f"_{instance_idx}"
        if input_port_id is not None:
            fq_name += f"_I{input_port_id}"
        return fq_name

    def _get_quantizer_operation_name(self, layer_name, weight_attr_name):
        return f'{layer_name}_{weight_attr_name}_quantizer'


class QuantizationController(BaseCompressionAlgorithmController):
    def __init__(self, target_model, config, op_names: List[str]):
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
        return self._loss

    def strip_model(self, model: tf.keras.Model) -> tf.keras.Model:
        apply_saturation_fix(model, self._op_names)
        return model

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        collector = TFQuantizationStatisticsCollector(self.model, self._op_names)
        stats = collector.collect()

        nncf_stats = NNCFStatistics()
        nncf_stats.register('quantization', stats)
        return nncf_stats
