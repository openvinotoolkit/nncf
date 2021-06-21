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

from typing import List, Set

import networkx as nx
import tensorflow as tf

from nncf import NNCFConfig
from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.common.graph import Dtype
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.common.quantization.initialization.range import RangeInitParams
from nncf.common.quantization.structs import QuantizationConstraints
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.helpers import should_consider_scope
from nncf.common.utils.logger import logger
from nncf.config.extractors import extract_range_init_params
from nncf.config.extractors import extract_bn_adaptation_init_params
from nncf.tensorflow.algorithm_selector import TF_COMPRESSION_ALGORITHMS
from nncf.tensorflow.api.compression import TFCompressionAlgorithmBuilder
from nncf.tensorflow.api.compression import TFCompressionAlgorithmController
from nncf.tensorflow.graph import patterns as p
from nncf.tensorflow.graph.converter import convert_keras_model_to_nncf_graph
from nncf.tensorflow.graph.metatypes.common import ELEMENTWISE_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.common import GENERAL_CONV_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.common import LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION
from nncf.tensorflow.graph.metatypes.common import LINEAR_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.keras_layers import TFLambdaLayerMetatype
from nncf.tensorflow.graph.pattern_matching import search_all
from nncf.tensorflow.graph.transformations.commands import TFAfterLayer
from nncf.tensorflow.graph.transformations.commands import TFInsertionCommand
from nncf.tensorflow.graph.transformations.commands import TFLayerWeight
from nncf.tensorflow.graph.transformations.layout import TFTransformationLayout
from nncf.tensorflow.graph.utils import get_original_name_and_instance_index
from nncf.tensorflow.layers.custom_objects import NNCF_QUANTIZATION_OPERATONS
from nncf.tensorflow.loss import TFZeroCompressionLoss
from nncf.tensorflow.quantization.initializers.minmax import MinMaxInitializer
from nncf.tensorflow.quantization.layers import FakeQuantize
from nncf.tensorflow.quantization.quantizers import Quantizer
from nncf.tensorflow.quantization.quantizers import TFQuantizerSpec
from nncf.tensorflow.quantization.utils import apply_saturation_fix

QUANTIZATION_LAYER_METATYPES = GENERAL_CONV_LAYER_METATYPES + LINEAR_LAYER_METATYPES

NOT_SUPPORT_LAYER_METATYPES = [
    TFLambdaLayerMetatype
]


@TF_COMPRESSION_ALGORITHMS.register('quantization')
class QuantizationBuilder(TFCompressionAlgorithmBuilder):
    def __init__(self, config: NNCFConfig, should_init: bool = True):
        super().__init__(config, should_init)

        self.quantize_inputs = self.config.get('quantize_inputs', True)
        self.quantize_outputs = self.config.get('quantize_outputs', False)
        self._disable_saturation_fix = self.config.get('disable_saturation_fix', False)
        self._target_device = config.get('target_device')

        self.global_quantizer_constraints = {}
        self.ignored_scopes_per_group = {}
        self.target_scopes_per_group = {}
        self._op_names = []

        for quantizer_group in QuantizerGroup:
            self._parse_group_params(self.config, quantizer_group)

        self._parse_init_params()
        self._range_initializer = None
        self._bn_adaptation = None

    def _parse_init_params(self):
        self._batchnorm_adaptation = 'batchnorm_adaptation' in self.config.get('initializer', {})
        self._range_init_params = self._parse_range_init_params()

    def _parse_range_init_params(self) -> RangeInitParams:
        range_init_params = extract_range_init_params(self.config)
        return RangeInitParams(**range_init_params) if range_init_params is not None else None

    def _parse_group_params(self, config: NNCFConfig, quantizer_group: QuantizerGroup) -> None:
        group_name = quantizer_group.value
        params_dict = config.get(group_name, {})
        self.global_quantizer_constraints[quantizer_group] = QuantizationConstraints.from_config_dict(params_dict)
        self.ignored_scopes_per_group[quantizer_group] = config.get('ignored_scopes', []) \
                                                    + params_dict.get('ignored_scopes', [])
        self.target_scopes_per_group[quantizer_group] = params_dict.get('target_scopes')

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
        nncf_graph = convert_keras_model_to_nncf_graph(model)
        nodes = nncf_graph.get_all_nodes()
        for node in nodes:
            if node.metatype in NOT_SUPPORT_LAYER_METATYPES:
                logger.warning('The layer {} is not supported by the quantization algorithm'
                               .format(get_original_name_and_instance_index(node.node_name)[0]))

        transformations = TFTransformationLayout()
        qconfig = self._get_default_qconfig(self.global_quantizer_constraints[QuantizerGroup.WEIGHTS])
        half_range = self._get_half_range(qconfig)
        processed_shared_layer_names = set()  # type: Set[str]
        for node in nodes:
            if node.is_shared():
                target_layer_name, _ = get_original_name_and_instance_index(node.node_name)
                if target_layer_name in processed_shared_layer_names:
                    continue
                processed_shared_layer_names.add(target_layer_name)
            else:
                target_layer_name = node.node_name

            if not (node.metatype in QUANTIZATION_LAYER_METATYPES \
                    and should_consider_scope(node.node_name,
                                              ignored_scopes=self.ignored_scopes_per_group[QuantizerGroup.WEIGHTS],
                                              target_scopes=None)):
                continue

            for weight_def in node.metatype.weight_definitions:
                op_name = self._get_quantizer_operation_name(
                    node.node_name,
                    weight_def.weight_attr_name)
                self._op_names.append(op_name)

                operation = self._create_quantizer(
                    op_name,
                    TFQuantizerSpec.from_config(qconfig,
                                                narrow_range=not half_range,
                                                half_range=half_range))

                transformations.register(
                    TFInsertionCommand(
                        target_point=TFLayerWeight(target_layer_name, weight_def.weight_attr_name),
                        callable_object=operation,
                        priority=TransformationPriority.QUANTIZATION_PRIORITY))

        insertion_points = self._find_insertion_points(nncf_graph)
        qconfig = self._get_default_qconfig(self.global_quantizer_constraints[QuantizerGroup.ACTIVATIONS])
        for original_node_name, instance_index in insertion_points:
            fake_quantize_name = self._get_fake_quantize_name(original_node_name, instance_index)
            fake_quantize_layer = FakeQuantize(
                TFQuantizerSpec.from_config(qconfig, narrow_range=False, half_range=False),
                name=fake_quantize_name)
            self._op_names.append(fake_quantize_layer.op_name)

            transformations.register(
                TFInsertionCommand(
                    target_point=TFAfterLayer(original_node_name, instance_index),
                    callable_object=fake_quantize_layer,
                    priority=TransformationPriority.QUANTIZATION_PRIORITY))

        return transformations

    def build_controller(self, model: tf.keras.Model) -> 'QuantizationController':
        return QuantizationController(model, self.config, self._op_names)

    def initialize(self, model: tf.keras.Model) -> None:
        self._run_range_initialization(model)

        if self._batchnorm_adaptation:
            self._run_batchnorm_adaptation(model)

    def _run_range_initialization(self, model: tf.keras.Model) -> None:
        if self._range_initializer is None:
            self._range_initializer = MinMaxInitializer(self._range_init_params)
        self._range_initializer.run(model)

    def _run_batchnorm_adaptation(self, model: tf.keras.Model) -> None:
        if self._bn_adaptation is None:
            self._bn_adaptation = BatchnormAdaptationAlgorithm(
                **extract_bn_adaptation_init_params(self.config))
        self._bn_adaptation.run(model)

    def _find_insertion_points(self, nncf_graph: NNCFGraph) -> List[str]:
        def _filter_fn(node: NNCFNode):
            ignored = not should_consider_scope(
                node.node_name,
                ignored_scopes=self.ignored_scopes_per_group[QuantizerGroup.ACTIVATIONS],
                target_scopes=None)
            # Works if the insertion is done as an operation after the corresponding node.
            out_nodes = nncf_graph.get_next_nodes(node)
            is_float_dtype = True
            for out_node in out_nodes:
                out_edge = nncf_graph.get_nx_edge(node, out_node)
                if out_edge[NNCFGraph.DTYPE_EDGE_ATTR] != Dtype.FLOAT:
                    is_float_dtype = False
            return (not ignored) and is_float_dtype

        pattern = p.CONV_LINEAR_OPS | p.ELEMENTWISE | p.ANY_BN_ACT_COMBO | \
                  p.CONV_LINEAR_OPS + p.ANY_AG_BN_ACT_COMBO | p.ELEMENTWISE + p.ANY_AG_BN_ACT_COMBO | p.SINGLE_OPS

        nx_graph = nncf_graph.get_nx_graph_copy()
        matches = search_all(nncf_graph.get_nx_graph_copy(), pattern)

        topological_order = {node: k for k, node in enumerate(nx.topological_sort(nx_graph))}
        insertion_point_node_keys = [max(match, key=topological_order.__getitem__) for match in matches]

        if self.quantize_inputs:
            for nncf_node in nncf_graph.get_input_nodes():
                node_key = nncf_graph.get_node_key_by_id(nncf_node.node_id)
                preprocessing_nodes = self._get_input_preprocessing_nodes(nncf_graph, node_key)
                if preprocessing_nodes:
                    for n in preprocessing_nodes[:-1]:
                        preprocessing_node_key = nncf_graph.get_node_key_by_id(n.node_id)
                        if preprocessing_node_key in insertion_point_node_keys:
                            insertion_point_node_keys.remove(node_key)
                elif _filter_fn(nncf_node):
                    insertion_point_node_keys = [node_key] + insertion_point_node_keys

        if not self.quantize_outputs:
            for nncf_node in nncf_graph.get_output_nodes():
                node_key = nncf_graph.get_node_key_by_id(nncf_node.node_id)
                for quantized_node in self._get_quantized_nodes_for_output(nncf_graph,
                                                                           insertion_point_node_keys, node_key):
                    quantized_node_key = nncf_graph.get_node_key_by_id(quantized_node.node_id)
                    insertion_point_node_keys.remove(quantized_node_key)

        insertion_point_node_keys = [point for point in insertion_point_node_keys
                                     if _filter_fn(nncf_graph.get_node_by_key(point))]
        insertion_point_layer_names = [nncf_graph.get_node_by_key(ip).node_name for ip in insertion_point_node_keys]

        return [get_original_name_and_instance_index(ip) for ip in insertion_point_layer_names]

    def _get_input_preprocessing_nodes(self, nncf_graph: NNCFGraph, node_key: str,
                                       preprocessing_nodes: List[NNCFNode] = None) -> List[NNCFNode]:
        if preprocessing_nodes is None:
            preprocessing_nodes = []

        node = nncf_graph.get_node_by_key(node_key)
        successors = nncf_graph.get_next_nodes(node)
        if len(successors) == 1:
            successor = next(iter(successors))
            successor_key = nncf_graph.get_node_key_by_id(successor.node_id)
            if successor.metatype in ELEMENTWISE_LAYER_METATYPES and len(nncf_graph.get_previous_nodes(successor)) == 1:
                preprocessing_nodes.append(successor)
                return self._get_input_preprocessing_nodes(nncf_graph, successor_key, preprocessing_nodes)
        return preprocessing_nodes

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

    def _get_fake_quantize_name(self, node_name, instance_index):
        if instance_index == 0:
            return '{}/fake_quantize'.format(node_name)
        return '{}/fake_quantize_{}'.format(node_name, instance_index)

    def _get_quantizer_operation_name(self, layer_name, weight_attr_name):
        return f'{layer_name}_{weight_attr_name}_quantizer'


class QuantizationController(TFCompressionAlgorithmController):
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
        return NNCFStatistics()
