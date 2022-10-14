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

from typing import Set, Dict
from copy import deepcopy

import onnx
from onnx import numpy_helper

# pylint: disable=no-member

from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizationPoint
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.utils.backend import BackendType
from nncf.experimental.onnx.graph.model_transformer import ONNXModelTransformer

from nncf.experimental.post_training.algorithms.quantization.min_max_quantization import MinMaxQuantization
from nncf.experimental.post_training.algorithms.quantization.min_max_quantization import MinMaxQuantizationParameters
from nncf.experimental.post_training.algorithms.quantization.min_max_quantization import RangeType
from nncf.experimental.post_training.algorithms.algorithm import PostTrainingAlgorithms
from nncf.experimental.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.experimental.post_training.factories import NNCFGraphFactory
from nncf.experimental.post_training.statistics.statistic_point import StatisticPoint
from nncf.experimental.post_training.statistics.statistic_point import StatisticPointsContainer
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXTopKMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXNonMaxSuppressionMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import POSSIBLE_WEIGHT_LAYERS_METATYPES
from nncf.experimental.onnx.algorithms.quantization.default_quantization import DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT
from nncf.experimental.onnx.graph.transformations.commands import ONNXQuantizerInsertionCommand
from nncf.experimental.onnx.engine import ONNXEngine
from nncf.experimental.onnx.statistics.collectors import ONNXMinMaxStatisticCollector
from nncf.experimental.onnx.statistics.collectors import ONNXMeanMinMaxStatisticCollector
from nncf.experimental.onnx.graph.onnx_graph import ONNXGraph
from nncf.experimental.onnx.hardware.fused_patterns import ONNX_HW_FUSED_PATTERNS
from nncf.experimental.onnx.algorithms.quantization.utils import calculate_activation_quantizer_parameters
from nncf.experimental.onnx.algorithms.quantization.utils import calculate_weight_quantizer_parameters
from nncf.experimental.onnx.hardware.config import ONNXHWConfig


class ONNXMinMaxQuantization(MinMaxQuantization):

    def __init__(self, parameters: MinMaxQuantizationParameters):
        super().__init__(parameters)
        self.nncf_graph = None  # type: NNCFGraph
        # It prevents the duplicate weight quantizers from being added.
        # It can happen when you have layers that share the identical weight tensor.
        self._quantization_target_points = set()  # type: Set[ONNXTargetPoint]

    @property
    def available_backends(self) -> Dict[str, BackendType]:
        pass

    def generate_stat_collector(self, quantizer_config: QuantizerConfig) -> TensorStatisticCollectorBase:
        is_symmetric = quantizer_config.mode == QuantizationMode.SYMMETRIC
        axes = (0, 2, 3) if quantizer_config.per_channel else None
        if self.range_type == RangeType.MINMAX:
            return ONNXMinMaxStatisticCollector(use_abs_max=is_symmetric, reduction_shape=axes,
                                                num_samples=self.number_samples)
        if self.range_type == RangeType.MEAN_MINMAX:
            return ONNXMeanMinMaxStatisticCollector(use_per_sample_stats=False, use_abs_max=is_symmetric,
                                                    reduction_shape=axes, num_samples=self.number_samples)
        raise RuntimeError('This range type is not supported.')

    def _get_quantizer_setup(self, model: onnx.ModelProto):
        self.nncf_graph = NNCFGraphFactory.create(model) if self.nncf_graph is None else self.nncf_graph
        ip_graph = InsertionPointGraph(self.nncf_graph)
        pattern = ONNX_HW_FUSED_PATTERNS.get_full_pattern_graph()
        ip_graph = ip_graph.get_ip_graph_with_merged_hw_optimized_operations(pattern)

        onnx_graph = ONNXGraph(model)
        possible_weight_nodes = self.nncf_graph.get_nodes_by_metatypes(POSSIBLE_WEIGHT_LAYERS_METATYPES)
        weight_nodes = []
        for possible_weight_node in possible_weight_nodes:
            if onnx_graph.get_node_initializers(possible_weight_node.node_name):
                weight_nodes.append(possible_weight_node)

        quantizable_layer_nodes = [QuantizableWeightedLayerNode(weight_node, [QuantizerConfig()]) for weight_node in
                                   weight_nodes]

        hw_config_type = self.target_device
        hw_config_path = ONNXHWConfig.get_path_to_hw_config(hw_config_type)
        hw_config = ONNXHWConfig.from_json(hw_config_path)

        solver = QuantizerPropagationSolver(ignored_scopes=self.ignored_scopes,
                                            hw_config=hw_config,
                                            default_trait_to_metatype_map=DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT,
                                            default_qconfig_list=[self._get_default_qconfig()],
                                            quantizable_layer_nodes=quantizable_layer_nodes,
                                            quantize_outputs=self.quantize_outputs,
                                            post_processing_marker_metatypes=[ONNXTopKMetatype,
                                                                              ONNXNonMaxSuppressionMetatype])
        quantization_proposal = solver.run_on_ip_graph(ip_graph)
        multi_config_setup = quantization_proposal.quantizer_setup
        single_config_setup = multi_config_setup.select_first_qconfig_for_each_point()
        finalized_proposal = quantization_proposal.finalize(single_config_setup)
        final_setup = solver.get_final_quantizer_setup(finalized_proposal)
        return final_setup

    def _add_weight_quantization_target_point(self, quantization_point: SingleConfigQuantizationPoint) -> None:
        weight_quantization_target_point = ONNXTargetPoint(TargetType.OPERATION_WITH_WEIGHTS,
                                                           quantization_point.insertion_point.target_node_name)
        self._quantization_target_points.add(weight_quantization_target_point)

    def _add_activation_quantization_target_point(self, onnx_graph: ONNXGraph,
                                                  quantization_point: SingleConfigQuantizationPoint) -> None:
        node_name = quantization_point.insertion_point.target_node_name
        # If quantization of Model Input node
        if NNCFGraphNodeType.INPUT_NODE in node_name:
            # There is only onde node - input_node
            activation_quantization_target_point = ONNXTargetPoint(TargetType.POST_LAYER_OPERATION, node_name)
        # If not Model Input node
        # If Quantization of node's input
        elif quantization_point.insertion_point.input_port_id is not None:
            edge_name = onnx_graph.get_node_edges(node_name)['input'][quantization_point.insertion_point.input_port_id]
            activation_quantization_target_point = ONNXTargetPoint(TargetType.PRE_LAYER_OPERATION,
                                                                   node_name,
                                                                   edge_name)
        # If quantization of node's output
        else:
            activation_quantization_target_point = ONNXTargetPoint(TargetType.POST_LAYER_OPERATION,
                                                                   node_name)
        self._quantization_target_points.add(activation_quantization_target_point)

    def get_quantization_target_points(self, model: onnx.ModelProto) -> Set[ONNXTargetPoint]:
        """
        Returns Quantization Target Points.
        In the Compression Pipeline logic NNCF assumes that the compression pipeline works only on the single model.
        So for the optimization purpose if Quantization Target Points were computed before the function returns them,
        otherwise builds NNCFGraph from the 'model',
        finds the quantization setup and processes it to the Set of Quantization Target Points.
        :param model: ONNX model, for which Quantization Target Points are being seek.
        :return: Set of Quantization Target Points.
        """
        if self._quantization_target_points:
            return self._quantization_target_points
        quantizer_setup = self._get_quantizer_setup(model)
        onnx_graph = ONNXGraph(model)
        for quantization_point in quantizer_setup.quantization_points.values():
            if quantization_point.is_weight_quantization_point():
                self._add_weight_quantization_target_point(quantization_point)
            elif quantization_point.is_activation_quantization_point():
                self._add_activation_quantization_target_point(onnx_graph, quantization_point)
            else:
                raise RuntimeError('Incorrect quantization point')
        self._quantization_target_points = sorted(self._quantization_target_points)
        return self._quantization_target_points

    def reset_quantization_target_points(self) -> None:
        self._quantization_target_points = set()

    def _apply(self, model: onnx.ModelProto, engine: ONNXEngine,
               statistic_points: StatisticPointsContainer) -> onnx.ModelProto:
        transformation_layout, transformation_commands = TransformationLayout(), []
        onnx_graph = ONNXGraph(model)
        model_transformer = ONNXModelTransformer(model)

        quantization_target_points = self.get_quantization_target_points(model)
        weight_quantizer_config = self._get_weight_quantizer_config(model)
        weight_initializer_names = set()

        for quantization_target_point in quantization_target_points:
            target_node_name = quantization_target_point.target_node_name
            if quantization_target_point.type == TargetType.OPERATION_WITH_WEIGHTS:
                weight_initializers = onnx_graph.get_node_initializers(target_node_name)
                if not weight_initializers:
                    nncf_logger.exception('There is no initializer in the node {}'.format(target_node_name))
                    continue
                # We assume that the weight tensor should be the first in the list
                weight_initializer = weight_initializers[0]
                weight_initializer_name = weight_initializer.name
                # If the nodes share one weight tensor, we should have only one quantizer on that
                if weight_initializer_name in weight_initializer_names:
                    continue
                weight_initializer_names.add(weight_initializer_name)
                weight_tensor = numpy_helper.to_array(weight_initializer)
                parameters = calculate_weight_quantizer_parameters(weight_tensor, weight_quantizer_config)
                command = ONNXQuantizerInsertionCommand(quantization_target_point, parameters)
                transformation_commands.append(command)
            elif quantization_target_point.type in [TargetType.PRE_LAYER_OPERATION, TargetType.POST_LAYER_OPERATION]:
                def filter_func(point):
                    return PostTrainingAlgorithms.MinMaxQuantization in point.algorithm_to_tensor_collectors and \
                           point.target_point.type == quantization_target_point.type

                for tensor_collector in statistic_points.get_algo_statistics_for_node(
                        target_node_name,
                        filter_func,
                        PostTrainingAlgorithms.MinMaxQuantization):
                    parameters = calculate_activation_quantizer_parameters(tensor_collector.get_statistics(),
                                                                           self.activation_quantizer_config)
                    command = ONNXQuantizerInsertionCommand(quantization_target_point, parameters)
                    transformation_commands.append(command)
            else:
                raise RuntimeError('Inccorrect type of Quantization Target Point')

        for transformation_command in transformation_commands:
            transformation_layout.register(transformation_command)

        quantized_model = model_transformer.transform(transformation_layout)
        return quantized_model

    def get_statistic_points(self, model: onnx.ModelProto) -> StatisticPointsContainer:
        quantization_target_points = self.get_quantization_target_points(model)
        output = StatisticPointsContainer()
        for quantization_target_point in quantization_target_points:
            if quantization_target_point.type in [TargetType.PRE_LAYER_OPERATION, TargetType.POST_LAYER_OPERATION]:
                nncf_logger.debug(
                    'Adding {} Quantization Target Point to the Statistics Points,'
                    ' which outputs will be used for statistics collection'.format(
                        quantization_target_point.target_node_name))
                output.add_statistic_point(StatisticPoint(target_point=quantization_target_point,
                                                          tensor_collector=self.generate_stat_collector(
                                                              self.activation_quantizer_config),
                                                          algorithm=PostTrainingAlgorithms.MinMaxQuantization)
                                           )
            else:
                nncf_logger.debug(
                    'Skipping {} Quantization Target Point, which is used for weights quantization'.format(
                        quantization_target_point))
        return output

    def create_subalgorithms(self, backend: BackendType) -> None:
        return

    def _get_weight_quantizer_config(self, model: onnx.ModelProto) -> QuantizerConfig:
        config = deepcopy(self.weight_quantizer_config)

        if model.opset_import[0].version < 13:
            config.per_channel = False
            nncf_logger.warning(
                f"Model opset version is {model.opset_import[0].version} < 13. "
                "Per-channel quantization is not supported. "
                "Set weight_quantizer_config.per_channel = False"
            )

        return config
