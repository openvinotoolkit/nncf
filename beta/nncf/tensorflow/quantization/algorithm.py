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

import networkx as nx

from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.utils.logger import logger
from nncf.common.schedulers import BaseCompressionScheduler
from beta.nncf.tensorflow.algorithm_selector import TF_COMPRESSION_ALGORITHMS
from beta.nncf.tensorflow.api.compression import TFCompressionAlgorithmBuilder
from beta.nncf.tensorflow.api.compression import TFCompressionAlgorithmController
from beta.nncf.tensorflow.loss import TFZeroCompressionLoss
from beta.nncf.tensorflow.graph import patterns as p
from beta.nncf.tensorflow.graph.metatypes.common import ELEMENTWISE_LAYER_METATYPES
from beta.nncf.tensorflow.graph.metatypes.common import GENERAL_CONV_LAYER_METATYPES
from beta.nncf.tensorflow.graph.metatypes.common import LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION
from beta.nncf.tensorflow.graph.metatypes.common import LINEAR_LAYER_METATYPES
from beta.nncf.tensorflow.graph.metatypes.keras_layers import TFLambdaLayerMetatype
from beta.nncf.tensorflow.graph.converter import convert_keras_model_to_nncf_graph
from beta.nncf.tensorflow.graph.pattern_matching import search_all
from beta.nncf.tensorflow.graph.transformations.commands import TFInsertionCommand
from beta.nncf.tensorflow.graph.transformations.commands import TFAfterLayer
from beta.nncf.tensorflow.graph.transformations.commands import TFLayerWeight
from beta.nncf.tensorflow.graph.transformations.layout import TFTransformationLayout
from beta.nncf.tensorflow.graph.utils import get_original_name_and_instance_index
from beta.nncf.tensorflow.layers.custom_objects import NNCF_QUANTIZATION_OPERATONS
from beta.nncf.tensorflow.quantization.initializers.minmax import MinMaxInitializer
from beta.nncf.tensorflow.quantization.layers import FakeQuantize
from beta.nncf.tensorflow.quantization.quantizers import TFQuantizerSpec
from beta.nncf.tensorflow.utils.node import is_ignored
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizationConstraints
from nncf.common.statistics import NNCFStatistics
from nncf.api.compression import CompressionScheduler
from nncf.api.compression import CompressionLoss
from nncf.common.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.config.utils import extract_bn_adaptation_init_params

ACTIVATIONS = "activations"
WEIGHTS = "weights"

QUANTIZER_GROUPS = [
    ACTIVATIONS,
    WEIGHTS
]

QUANTIZATION_LAYER_METATYPES = GENERAL_CONV_LAYER_METATYPES + LINEAR_LAYER_METATYPES

NOT_SUPPORT_LAYER_METATYPES = [
    TFLambdaLayerMetatype
]


@TF_COMPRESSION_ALGORITHMS.register('quantization')
class QuantizationBuilder(TFCompressionAlgorithmBuilder):
    def __init__(self, config, should_init: bool = True):
        super().__init__(config, should_init)

        self.quantize_inputs = self.config.get('quantize_inputs', True)
        self.quantize_outputs = self.config.get('quantize_outputs', False)

        self.global_quantizer_constraints = {}
        self.ignored_scopes_per_group = {}
        self.target_scopes_per_group = {}

        for quantizer_group in QUANTIZER_GROUPS:
            self._parse_group_params(self.config, quantizer_group)

    def build_controller(self, model):
        return QuantizationController(model, self.config)

    def _parse_group_params(self, config, quantizer_group):
        params_dict = config.get(quantizer_group, {})
        self.global_quantizer_constraints[quantizer_group] = QuantizationConstraints(
            num_bits=params_dict.get('bits'),
            mode=params_dict.get('mode'),
            signedness_to_force=params_dict.get('signed'),
            per_channel=params_dict.get('per_channel')
        )
        self.ignored_scopes_per_group[quantizer_group] = config.get('ignored_scopes', []) \
                                                         + params_dict.get('ignored_scopes', [])
        self.target_scopes_per_group[quantizer_group] = params_dict.get('target_scopes')

    def _get_default_qconfig(self, constraints: QuantizationConstraints = None):
        qconfig = QuantizerConfig(num_bits=8,
                                  mode=QuantizationMode.SYMMETRIC,
                                  signedness_to_force=None,
                                  per_channel=False)
        if constraints is not None:
            qconfig = constraints.apply_constraints_to(qconfig)
        return qconfig

    def _create_quantizer(self, name: str, qconfig: TFQuantizerSpec):
        quantizer_cls = NNCF_QUANTIZATION_OPERATONS.get(qconfig.mode)
        return quantizer_cls(name, qconfig)

    def get_transformation_layout(self, model):
        graph = convert_keras_model_to_nncf_graph(model)
        nodes = graph.get_all_nodes()
        for node in nodes:
            if node.metatype in NOT_SUPPORT_LAYER_METATYPES:
                logger.warning('The layer {} is not supported by the quantization algorithm'
                               .format(get_original_name_and_instance_index(node.node_name)[0]))

        transformations = TFTransformationLayout()
        qconfig = self._get_default_qconfig(self.global_quantizer_constraints[WEIGHTS])
        shared_nodes = set()
        for node in nodes:
            original_node_name, _ = get_original_name_and_instance_index(node.node_name)
            if node.metatype not in QUANTIZATION_LAYER_METATYPES \
                    or is_ignored(node.node_name, self.ignored_scopes_per_group[WEIGHTS]) \
                    or original_node_name in shared_nodes:
                continue

            if node.data['is_shared']:
                shared_nodes.add(original_node_name)

            for weight_def in node.metatype.weight_definitions:
                op_name = self._get_quantizer_operation_name(
                    node.node_name,
                    weight_def.weight_attr_name)

                operation = self._create_quantizer(
                    op_name,
                    TFQuantizerSpec.from_config(qconfig, narrow_range=True, half_range=False))

                transformations.register(
                    TFInsertionCommand(
                        target_point=TFLayerWeight(original_node_name, weight_def.weight_attr_name),
                        callable_object=operation,
                        priority=TransformationPriority.QUANTIZATION_PRIORITY))

        insertion_points = self._find_insertion_points(graph)
        qconfig = self._get_default_qconfig(self.global_quantizer_constraints[ACTIVATIONS])
        for original_node_name, instance_index in insertion_points:
            fake_quantize_name = self._get_fake_quantize_name(original_node_name, instance_index)
            fake_quantize_layer = FakeQuantize(
                TFQuantizerSpec.from_config(qconfig, narrow_range=False, half_range=False),
                name=fake_quantize_name)

            transformations.register(
                TFInsertionCommand(
                    target_point=TFAfterLayer(original_node_name, instance_index),
                    callable_object=fake_quantize_layer,
                    priority=TransformationPriority.QUANTIZATION_PRIORITY))

        return transformations

    def _find_insertion_points(self, graph):
        def _filter_fn(node):
            return not is_ignored(node.node_name, self.ignored_scopes_per_group[ACTIVATIONS]) \
                   and 'float' in node.data['dtype'].lower()

        pattern = p.CONV_LINEAR_OPS | p.ELEMENTWISE | p.ANY_BN_ACT_COMBO | \
                  p.CONV_LINEAR_OPS + p.ANY_AG_BN_ACT_COMBO | p.ELEMENTWISE + p.ANY_AG_BN_ACT_COMBO | p.SINGLE_OPS

        matches = search_all(graph.nx_graph, pattern)

        topological_order = {node: k for k, node in enumerate(nx.topological_sort(graph.nx_graph))}
        insertion_points = [max(match, key=topological_order.__getitem__) for match in matches]

        if self.quantize_inputs:
            for node in graph.get_input_nodes():
                preprocessing_nodes = self._get_input_preprocessing_nodes(graph, node)
                if preprocessing_nodes:
                    for n in preprocessing_nodes[:-1]:
                        if n in insertion_points:
                            insertion_points.remove(node.node_name)
                elif _filter_fn(node):
                    insertion_points = [node.node_name] + insertion_points

        if not self.quantize_outputs:
            for node in graph.get_output_nodes():
                for quantized_node in self._get_quantized_nodes_for_output(
                        graph, insertion_points, node):
                    insertion_points.remove(quantized_node.node_name)

        insertion_points = [point for point in insertion_points
                            if _filter_fn(graph.get_node_by_key(point))]

        return [get_original_name_and_instance_index(point) for point in insertion_points]

    def _get_input_preprocessing_nodes(self, graph, node, preprocessing_nodes=None):
        if preprocessing_nodes is None:
            preprocessing_nodes = []

        succ_nodes = graph.get_next_nodes(node)
        if len(succ_nodes) == 1:
            pred_nodes = graph.get_previous_nodes(succ_nodes[0])
            if succ_nodes[0].metatype in ELEMENTWISE_LAYER_METATYPES and len(pred_nodes) == 1:
                preprocessing_nodes.append(succ_nodes[0])
                return self._get_input_preprocessing_nodes(graph, succ_nodes[0], preprocessing_nodes)
        return preprocessing_nodes

    def _get_quantized_nodes_for_output(self, graph, insetrion_points, node, quantized_nodes_for_output=None):
        if quantized_nodes_for_output is None:
            if node.node_name in insetrion_points:
                return [node]
            quantized_nodes_for_output = []

        for pred_node in graph.get_previous_nodes(node):
            if len(graph.get_next_nodes(pred_node)) > 1:
                logger.warning('Removing of FakeQuantize after layer {} '
                               'with multiple outputs is not fully supported'.format(pred_node.node_name))
                continue

            if pred_node.metatype in LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION:
                self._get_quantized_nodes_for_output(graph, insetrion_points,
                                                     pred_node, quantized_nodes_for_output)
            elif pred_node.node_name in insetrion_points:
                quantized_nodes_for_output.append(pred_node)
        return quantized_nodes_for_output

    def _get_fake_quantize_name(self, node_name, instance_index):
        if instance_index == 0:
            return '{}/fake_quantize'.format(node_name)
        return '{}/fake_quantize_{}'.format(node_name, instance_index)

    def _get_quantizer_operation_name(self, layer_name, weight_attr_name):
        return f'{layer_name}_{weight_attr_name}_quantizer'


class QuantizationController(TFCompressionAlgorithmController):
    def __init__(self, target_model, config):
        super().__init__(target_model)
        self._initializer = MinMaxInitializer(config)
        self._scheduler = BaseCompressionScheduler()
        self._loss = TFZeroCompressionLoss()
        self._config = config
        self._bn_adaptation = None

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    def initialize(self, dataset=None, loss=None):
        self._initializer(self._model, dataset, loss)

        init_bn_adapt_config = self._config.get('initializer', {}).get('batchnorm_adaptation', {})
        if init_bn_adapt_config:
            self._run_batchnorm_adaptation()

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        return NNCFStatistics()

    def _run_batchnorm_adaptation(self):
        if self._bn_adaptation is None:
            self._bn_adaptation = BatchnormAdaptationAlgorithm(
                **extract_bn_adaptation_init_params(self._config))
        self._bn_adaptation.run(self.model)
