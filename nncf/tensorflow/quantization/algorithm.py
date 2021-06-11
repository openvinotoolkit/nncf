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
from typing import List
from typing import Set

import networkx as nx

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
from nncf.tensorflow.quantization.quantizers import TFQuantizerSpec
from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.quantization.structs import QuantizationConstraints
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.statistics import NNCFStatistics
from nncf.common.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.config.utils import extract_bn_adaptation_init_params
from nncf.common.utils.helpers import should_consider_scope
from nncf.common.utils.logger import logger

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
        nncf_graph = convert_keras_model_to_nncf_graph(model)
        nodes = nncf_graph.get_all_nodes()
        for node in nodes:
            if node.metatype in NOT_SUPPORT_LAYER_METATYPES:
                logger.warning('The layer {} is not supported by the quantization algorithm'
                               .format(get_original_name_and_instance_index(node.node_name)[0]))

        transformations = TFTransformationLayout()
        qconfig = self._get_default_qconfig(self.global_quantizer_constraints[WEIGHTS])
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
                    and should_consider_scope(node.node_name, ignored_scopes=self.ignored_scopes_per_group[WEIGHTS],
                                              target_scopes=None)):
                continue

            for weight_def in node.metatype.weight_definitions:
                op_name = self._get_quantizer_operation_name(
                    node.node_name,
                    weight_def.weight_attr_name)

                operation = self._create_quantizer(
                    op_name,
                    TFQuantizerSpec.from_config(qconfig, narrow_range=True, half_range=False))

                transformations.register(
                    TFInsertionCommand(
                        target_point=TFLayerWeight(target_layer_name, weight_def.weight_attr_name),
                        callable_object=operation,
                        priority=TransformationPriority.QUANTIZATION_PRIORITY))

        insertion_points = self._find_insertion_points(nncf_graph)
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

    def _find_insertion_points(self, nncf_graph: NNCFGraph) -> List[str]:
        def _filter_fn(node: NNCFNode):
            ignored = not should_consider_scope(node.node_name,
                                                ignored_scopes=self.ignored_scopes_per_group[ACTIVATIONS],
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
