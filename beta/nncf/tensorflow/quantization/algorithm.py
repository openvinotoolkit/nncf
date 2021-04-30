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
from beta.nncf.tensorflow.graph.converter import convert_keras_model_to_nxmodel
from beta.nncf.tensorflow.graph.pattern_matching import search_all
from beta.nncf.tensorflow.graph.transformations.commands import TFInsertionCommand
from beta.nncf.tensorflow.graph.transformations.commands import TFAfterLayer
from beta.nncf.tensorflow.graph.transformations.commands import TFLayerWeight
from beta.nncf.tensorflow.graph.transformations.layout import TFTransformationLayout
from beta.nncf.tensorflow.graph.utils import get_original_name_and_instance_index
from beta.nncf.tensorflow.layers.common import ELEMENTWISE_LAYERS
from beta.nncf.tensorflow.layers.common import LAYERS_AGNOSTIC_TO_DATA_PRECISION
from beta.nncf.tensorflow.layers.common import LAYERS_WITH_WEIGHTS
from beta.nncf.tensorflow.layers.common import WEIGHT_ATTR_NAME
from beta.nncf.tensorflow.layers.custom_objects import NNCF_QUANTIZATION_OPERATONS
from beta.nncf.tensorflow.quantization.initializers.minmax import MinMaxInitializer
from beta.nncf.tensorflow.quantization.layers import FakeQuantize
from beta.nncf.tensorflow.quantization.quantizers import TFQuantizerSpec
from beta.nncf.tensorflow.utils.node import is_ignored
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizationConstraints
from nncf.common.compression import StubStatistics
from nncf.api.compression import CompressionScheduler
from nncf.api.compression import CompressionLoss

ACTIVATIONS = "activations"
WEIGHTS = "weights"

QUANTIZER_GROUPS = [
    ACTIVATIONS,
    WEIGHTS
]

QUANTIZATION_LAYERS = LAYERS_WITH_WEIGHTS

NOT_SUPPORT_LAYERS = [
    'Lambda'
]


@TF_COMPRESSION_ALGORITHMS.register('quantization')
class QuantizationBuilder(TFCompressionAlgorithmBuilder):
    def __init__(self, config):
        super().__init__(config)

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

    def _create_quantizer(self, name: str, qconfig: QuantizerConfig):
        quantizer_cls = NNCF_QUANTIZATION_OPERATONS.get(qconfig.mode)
        return quantizer_cls(name, qconfig)

    def get_transformation_layout(self, model):
        nxmodel = convert_keras_model_to_nxmodel(model)
        for node_name, node in nxmodel.nodes.items():
            if node['type'] in NOT_SUPPORT_LAYERS:
                logger.warning('The layer {} is not supported by the quantization algorithm'
                               .format(get_original_name_and_instance_index(node_name)[0]))

        transformations = TFTransformationLayout()
        qconfig = self._get_default_qconfig(self.global_quantizer_constraints[WEIGHTS])
        shared_nodes = set()
        for node_name, node in nxmodel.nodes.items():
            original_node_name, _ = get_original_name_and_instance_index(node_name)
            if node['type'] not in QUANTIZATION_LAYERS \
                    or is_ignored(node_name, self.ignored_scopes_per_group[WEIGHTS]) \
                    or original_node_name in shared_nodes:
                continue

            if node['is_shared']:
                shared_nodes.add(original_node_name)

            weight_attr_name = QUANTIZATION_LAYERS[node['type']][WEIGHT_ATTR_NAME]
            op_name = self._get_quantizer_operation_name(node_name, weight_attr_name)

            operation = self._create_quantizer(op_name, TFQuantizerSpec.from_config(qconfig,
                                                                           narrow_range=True,
                                                                           half_range=False))

            transformations.register(
                TFInsertionCommand(
                    target_point=TFLayerWeight(original_node_name, weight_attr_name),
                    callable_object=operation,
                    priority=TransformationPriority.QUANTIZATION_PRIORITY
                ))

        insertion_points = self._find_insertion_points(nxmodel)
        qconfig = self._get_default_qconfig(self.global_quantizer_constraints[ACTIVATIONS])
        for original_node_name, instance_index in insertion_points:
            fake_quantize_name = self._get_fake_quantize_name(original_node_name, instance_index)
            fake_quantize_layer = FakeQuantize(TFQuantizerSpec.from_config(qconfig, narrow_range=False,
                                                                           half_range=False),
                                               name=fake_quantize_name)

            transformations.register(
                TFInsertionCommand(
                    target_point=TFAfterLayer(original_node_name, instance_index),
                    callable_object=fake_quantize_layer,
                    priority=TransformationPriority.QUANTIZATION_PRIORITY
                ))

        return transformations

    def _find_insertion_points(self, nxmodel):
        def _filter_fn(node_name, node):
            return not is_ignored(node_name, self.ignored_scopes_per_group[ACTIVATIONS]) \
                   and 'float' in node['dtype'].lower()

        pattern = p.LINEAR_OPS | p.ELEMENTWISE | p.ANY_BN_ACT_COMBO | \
                  p.LINEAR_OPS + p.ANY_AG_BN_ACT_COMBO | p.ELEMENTWISE + p.ANY_AG_BN_ACT_COMBO | p.SINGLE_OPS

        matches = search_all(nxmodel, pattern)

        topological_order = {node: k for k, node in enumerate(nx.topological_sort(nxmodel))}
        insertion_points = [max(match, key=topological_order.__getitem__) for match in matches]

        if self.quantize_inputs:
            for node_name, degree in nxmodel.in_degree:
                if degree > 0:
                    continue

                preprocessing_nodes = self._get_input_preprocessing_nodes(nxmodel, node_name)
                if preprocessing_nodes:
                    for n in preprocessing_nodes[:-1]:
                        if n in insertion_points:
                            insertion_points.remove(node_name)
                elif _filter_fn(node_name, nxmodel.nodes[node_name]):
                    insertion_points = [node_name] + insertion_points

        if not self.quantize_outputs:
            outputs = []
            for node_name in nxmodel.nodes:
                if nxmodel.out_degree(node_name) == 0:
                    outputs.append(node_name)
            for output in outputs:
                for quantized_node in self._get_quantized_nodes_for_output(nxmodel, insertion_points, output):
                    insertion_points.remove(quantized_node)

        insertion_points = [point for point in insertion_points
                            if _filter_fn(point, nxmodel.nodes[point])]

        return [get_original_name_and_instance_index(point) for point in insertion_points]

    def _get_input_preprocessing_nodes(self, nxmodel, node_name, preprocessing_nodes=None):
        if preprocessing_nodes is None:
            preprocessing_nodes = []

        if nxmodel.out_degree(node_name) == 1:
            successor = next(nxmodel.successors(node_name))
            if nxmodel.nodes[successor]['type'] in ELEMENTWISE_LAYERS and nxmodel.in_degree(successor) == 1:
                preprocessing_nodes.append(successor)
                return self._get_input_preprocessing_nodes(nxmodel, successor, preprocessing_nodes)
        return preprocessing_nodes

    def _get_quantized_nodes_for_output(self, nxmodel, insetrion_points, node_name, quantized_nodes_for_output=None):
        if quantized_nodes_for_output is None:
            if node_name in insetrion_points:
                return [node_name]
            quantized_nodes_for_output = []

        for predecessor in nxmodel.predecessors(node_name):
            if nxmodel.out_degree(predecessor) > 1:
                logger.warning('Removing of FakeQuantize after layer {} '
                               'with multiple outputs is not fully supported'.format(predecessor))
            if nxmodel.nodes[predecessor]['type'] in LAYERS_AGNOSTIC_TO_DATA_PRECISION:
                self._get_quantized_nodes_for_output(nxmodel, insetrion_points,
                                                     predecessor, quantized_nodes_for_output)
            elif predecessor in insetrion_points:
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

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    def initialize(self, dataset=None, loss=None):
        self._initializer(self._model, dataset, loss)

    def statistics(self, quickly_collected_only: bool = False) -> StubStatistics:
        return StubStatistics()
