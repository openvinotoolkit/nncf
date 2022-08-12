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

from copy import deepcopy
from typing import Set
from typing import List

import onnx
# pylint: disable=no-member

from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.utils.logger import logger as nncf_logger

from nncf.experimental.post_training.algorithms.quantization.min_max_quantization import MinMaxQuantization
from nncf.experimental.post_training.algorithms.quantization.min_max_quantization import MinMaxQuantizationParameters
from nncf.experimental.post_training.algorithms.quantization.min_max_quantization import RangeType
from nncf.experimental.post_training.algorithms.algorithm import PostTrainingAlgorithms
from nncf.experimental.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.experimental.post_training.statistics.statistic_point import StatisticPoint
from nncf.experimental.post_training.statistics.statistic_point import StatisticPointsContainer
from nncf.experimental.onnx.graph.transformations.layout import ONNXTransformationLayout
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import GENERAL_WEIGHT_LAYER_METATYPES
from nncf.experimental.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.experimental.onnx.algorithms.quantization.default_quantization import DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT
from nncf.experimental.onnx.graph.transformations.commands import ONNXQuantizerInsertionCommand
from nncf.experimental.onnx.engine import ONNXEngine
from nncf.experimental.onnx.statistics.collectors import ONNXMinMaxStatisticCollector
from nncf.experimental.onnx.statistics.collectors import ONNXMeanMinMaxStatisticCollector
from nncf.experimental.onnx.graph.model_transformer import ONNXModelTransformer
from nncf.experimental.onnx.graph.onnx_graph import ONNXGraph
from nncf.experimental.onnx.hardware.fused_patterns import ONNX_HW_FUSED_PATTERNS
from nncf.experimental.onnx.algorithms.quantization.utils import calculate_activation_quantizer_parameters
from nncf.experimental.onnx.algorithms.quantization.utils import calculate_weight_quantizer_parameters
from nncf.experimental.onnx.hardware.config import ONNXHWConfig
from nncf.experimental.post_training.backend import Backend

QUANTIZATION_LAYER_METATYPES = GENERAL_WEIGHT_LAYER_METATYPES


class ONNXMinMaxQuantization(MinMaxQuantization):

    def __init__(self, parameters: MinMaxQuantizationParameters):
        super().__init__(parameters)
        self.nncf_graph = None
        self._quantization_target_points = []  # type: List[ONNXTargetPoint]

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

    def _create_model_transformer(self, model: onnx.ModelProto) -> ONNXModelTransformer:
        return ONNXModelTransformer(model)

    def _get_quantizer_setup(self, model: onnx.ModelProto):
        self.nncf_graph = GraphConverter.create_nncf_graph(model) if self.nncf_graph is None else self.nncf_graph
        ip_graph = InsertionPointGraph(self.nncf_graph)
        pattern = ONNX_HW_FUSED_PATTERNS.get_full_pattern_graph()
        ip_graph = ip_graph.get_ip_graph_with_merged_hw_optimized_operations(pattern)

        weight_nodes = self.nncf_graph.get_nodes_by_metatypes(QUANTIZATION_LAYER_METATYPES)
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
                                            quantize_outputs=self.quantize_outputs)

        quantization_proposal = solver.run_on_ip_graph(ip_graph)
        multi_config_setup = quantization_proposal.quantizer_setup
        single_config_setup = multi_config_setup.select_first_qconfig_for_each_point()
        finalized_proposal = quantization_proposal.finalize(single_config_setup)
        final_setup = solver.get_final_quantizer_setup(finalized_proposal)
        return final_setup

    def get_quantization_target_points(self, model: onnx.ModelProto) -> List[ONNXTargetPoint]:
        if self._quantization_target_points:
            return self._quantization_target_points
        quantizer_setup = self._get_quantizer_setup(model)
        onnx_graph = ONNXGraph(model)
        weight_quantizer_node_names, filled_outputs = set(), set()
        for _, quantization_point in quantizer_setup.quantization_points.items():
            if quantization_point.is_weight_quantization_point():
                # It prevents the duplicate weight quantizers from being added.
                # It can happen when you have layers that share the identical weight tensor.
                if quantization_point.insertion_point.target_node_name in weight_quantizer_node_names:
                    continue
                weight_quantizer_node_names.add(quantization_point.insertion_point.target_node_name)

                weight_quantization_target_point = ONNXTargetPoint(TargetType.OPERATION_WITH_WEIGHTS,
                                                                   quantization_point.insertion_point.target_node_name)
                self._quantization_target_points.append(weight_quantization_target_point)
            else:
                assert quantization_point.is_activation_quantization_point()
                # If not Input node
                node_name = quantization_point.insertion_point.target_node_name
                if NNCFGraphNodeType.INPUT_NODE not in quantization_point.insertion_point.target_node_name:
                    # If quantization of Input
                    if quantization_point.insertion_point.input_port_id is not None:
                        # TODO (kshpv): need to be reconsidered:
                        #  some operators such as Mul and Add could have activation input tensor on 0 or 1 indices
                        # TODO (kshpv): input_port_id can be usefull in terms of quantizing only one edge.
                        #  Some of the models could required this.
                        outputs = onnx_graph.get_node_edges(node_name)['input'][0]
                        activation_quantization_target_point = ONNXTargetPoint(TargetType.PRE_LAYER_OPERATION,
                                                                               node_name)
                    # If quantization of Output
                    else:
                        outputs = onnx_graph.get_node_edges(node_name)['output'][0]
                        activation_quantization_target_point = ONNXTargetPoint(TargetType.POST_LAYER_OPERATION,
                                                                               node_name)
                # If Input node
                else:
                    outputs = \
                        onnx_graph.get_node_edges(quantization_point.directly_quantized_operator_node_names[0])[
                            'input'][0]
                    activation_quantization_target_point = ONNXTargetPoint(TargetType.POST_LAYER_OPERATION, node_name)

                if self._is_valid_activation_quantizer(outputs, filled_outputs, onnx_graph):
                    filled_outputs.add(outputs)
                    self._quantization_target_points.append(activation_quantization_target_point)

        return self._quantization_target_points

    def reset_quantization_target_points(self):
        self._quantization_target_points = []

    def _apply(self, model: onnx.ModelProto, engine: ONNXEngine,
               statistic_points: StatisticPointsContainer) -> onnx.ModelProto:
        model_transformer = self._create_model_transformer(model)
        transformation_layout, transformation_commands = ONNXTransformationLayout(), []
        onnx_graph = ONNXGraph(model)

        quantization_target_points = self.get_quantization_target_points(model)
        weight_quantizer_config = self._get_weight_quantizer_config(model)
        weight_initializer_names = set()

        for quantization_target_point in quantization_target_points:
            target_node_name = quantization_target_point.target_node_name
            if quantization_target_point.type == TargetType.OPERATION_WITH_WEIGHTS:
                try:
                    weight_initializer_name = onnx_graph.get_weight_tensor_with_initializer(target_node_name)
                    # If the nodes share one weight tensor, we should have only one quantizer on that
                    if weight_initializer_name in weight_initializer_names:
                        continue
                    weight_initializer_names.add(weight_initializer_name)
                except RuntimeError as er:
                    nncf_logger.exception(er)
                    continue
                weight_tensor = onnx_graph.get_initializers_value(weight_initializer_name)
                parameters = calculate_weight_quantizer_parameters(weight_tensor, weight_quantizer_config)

                command = ONNXQuantizerInsertionCommand(quantization_target_point, parameters)
                transformation_commands.append(command)
            elif quantization_target_point.type in [TargetType.PRE_LAYER_OPERATION, TargetType.POST_LAYER_OPERATION]:
                def filter_func(point):
                    return PostTrainingAlgorithms.MinMaxQuantization in point.algorithm_to_tensor_collectors and \
                           point.target_point.type == quantization_target_point.type

                for tensor_collector in statistic_points.iter_through_algorithm_tensor_collectors_in_target_node(
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

    def create_subalgorithms(self, backend: Backend) -> None:
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

    def _is_valid_activation_quantizer(
            self, outputs: str, filled_outputs: Set[str], onnx_graph: ONNXGraph) -> bool:

        if outputs in filled_outputs:
            # TODO (kshpv): resolve this problem with inception v3.
            nncf_logger.debug(f"Skipping {outputs} layer because it's duplicated.")
            return False
        if not onnx_graph.is_valid_tensor(outputs):
            nncf_logger.warning(f"Skipping {outputs} activation layer because it's not a valid node output.")
            return False

        return True
