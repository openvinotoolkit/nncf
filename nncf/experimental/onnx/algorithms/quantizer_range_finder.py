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

from typing import Union
from typing import List
from typing import Tuple

from copy import deepcopy
import numpy as np

from nncf.common.hardware.config import HWConfigType
from nncf.common.hardware.config import HW_CONFIG_TYPE_TARGET_DEVICE_MAP
from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.quantization.structs import QuantizationMode

from nncf.experimental.post_training.algorithms.quantizer_range_finder import QuantizerRangeFinderAlgorithm
from nncf.experimental.post_training.compressed_model import CompressedModel
from nncf.experimental.onnx.graph.onnx_graph import ONNXGraph
from nncf.experimental.onnx.graph.metatypes.onnx_ops import GENERAL_WEIGHT_LAYER_METATYPES
from nncf.experimental.onnx.algorithms.quantization.default_quantization import DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT
from nncf.experimental.onnx.graph.transformations.commands import ONNXQuantizerInsertionCommand
from nncf.experimental.onnx.graph.transformations.commands import ONNXInsertionCommand
from nncf.experimental.post_training.algorithms.quantizer_range_finder import QuantizerRangeFinderParameters
from nncf.experimental.post_training.statistics.statistics_collector import MinMaxLayerStatistic
from nncf.experimental.onnx.statistics.functions import ONNXTensorMaxFunc
from nncf.experimental.onnx.statistics.functions import ONNXTensorMinFunc
from nncf.experimental.onnx.statistics.functions import ONNXBatchMaxFunc
from nncf.experimental.onnx.statistics.functions import ONNXBatchMinFunc
from nncf.experimental.onnx.statistics.functions import ONNXBatchMeanFunc
from nncf.experimental.onnx.statistics.functions import ONNXStatisticsMeanFunc
from nncf.experimental.onnx.statistics.functions import ONNXStatisticsABSMAXFunc

from nncf.experimental.onnx.hardware.fused_patterns import ONNX_HW_FUSED_PATTERNS

from nncf.experimental.post_training.statistics.statistics_collector import WEIGHTS_ESTIMATOR_FUNCTION
from nncf.experimental.post_training.statistics.statistics_collector import ACTIVATIONS_ESTIMATOR_FUNCTION
from nncf.experimental.post_training.statistics.statistics_collector import BATCH_AGGREGATION_FUNCTION
from nncf.experimental.post_training.statistics.statistics_collector import STATISTICS_AGGREGATION_FUNCTION

from nncf.experimental.onnx.hardware.config import ONNXHWConfig

QUANTIZATION_LAYER_METATYPES = GENERAL_WEIGHT_LAYER_METATYPES


class ONNXQuantizerRangeFinderAlgorithm(QuantizerRangeFinderAlgorithm):
    """
    The base class for all post-training quantization initialization algorithms.
    """
    DEFAULT_QCONFIG = QuantizerConfig(num_bits=8,
                                      mode=QuantizationMode.SYMMETRIC,
                                      signedness_to_force=None,
                                      per_channel=False)

    def __init__(self, compressed_model: CompressedModel, engine, parameters: QuantizerRangeFinderParameters):
        super().__init__(compressed_model, engine, parameters)
        self._weight_quantizers = []
        self._activation_quantizers = []
        self.global_quantizer_constraints = {}

    def apply(self, quantized_model, statistics=None):
        pass

    def _determine_aggregation_func(self):
        self.weight_statistics_min_func = self._get_statistics_collection_func(
            self.parameters.weight_min_func)
        self.weight_statistics_max_func = self._get_statistics_collection_func(
            self.parameters.weight_max_func)
        self.activation_statistics_min_func = self._get_statistics_collection_func(
            self.parameters.activation_min_func)
        self.activation_statistics_max_func = self._get_statistics_collection_func(
            self.parameters.activation_max_func)
        self.batch_aggregation_min_func = self._get_batch_aggregation_func(
            self.parameters.batch_aggregation_min_func)
        self.batch_aggregation_max_func = self._get_batch_aggregation_func(
            self.parameters.batch_aggregation_max_func)
        self.statistics_aggregator_func = self._get_statistic_aggregation_func(
            self.parameters.statistics_aggregator_func)

    def _get_statistics_collection_func(self,
                                        function: Union[WEIGHTS_ESTIMATOR_FUNCTION, ACTIVATIONS_ESTIMATOR_FUNCTION]):
        if function == WEIGHTS_ESTIMATOR_FUNCTION.MIN or function == ACTIVATIONS_ESTIMATOR_FUNCTION.MIN:
            return ONNXTensorMinFunc
        elif function == WEIGHTS_ESTIMATOR_FUNCTION.MAX or function == ACTIVATIONS_ESTIMATOR_FUNCTION.MAX:
            return ONNXTensorMaxFunc

    def _get_batch_aggregation_func(self, function: BATCH_AGGREGATION_FUNCTION):
        if function == BATCH_AGGREGATION_FUNCTION.MEAN:
            return ONNXBatchMeanFunc
        elif function == BATCH_AGGREGATION_FUNCTION.MIN:
            return ONNXBatchMinFunc
        elif function == BATCH_AGGREGATION_FUNCTION.MAX:
            return ONNXBatchMaxFunc

    def _get_statistic_aggregation_func(self,
                                        function: STATISTICS_AGGREGATION_FUNCTION):
        if function == STATISTICS_AGGREGATION_FUNCTION.MEAN:
            return ONNXStatisticsMeanFunc
        elif function == STATISTICS_AGGREGATION_FUNCTION.ABS_MAX:
            return ONNXStatisticsABSMAXFunc

    def get_layers_for_statistics(self, weight_quantizer_config: QuantizerConfig,
                                  activation_quantizer_config: QuantizerConfig) -> List[MinMaxLayerStatistic]:
        quantizer_setup = self._get_quantizer_setup(self.compressed_model)
        original_model = self.compressed_model.original_model
        onnx_graph = ONNXGraph(original_model)
        filled_outputs = []
        for qp_id, qp in quantizer_setup.quantization_points.items():
            if qp.is_weight_quantization_point():
                weight_initializer_name = onnx_graph.find_weight_input_in_module(qp.insertion_point.target_node_name)
                self._weight_quantizers.append(weight_initializer_name)
            else:
                assert qp.is_activation_quantization_point()
                if 'model_input' not in qp.insertion_point.target_node_name:
                    node_name = qp.insertion_point.target_node_name
                    if qp.insertion_point.input_port_id == 1:
                        # If quantization of Input
                        outputs = onnx_graph.get_node_edges(node_name)['input'][0]
                    else:
                        # If quantization of Output
                        outputs = onnx_graph.get_node_edges(node_name)['output'][0]
                else:
                    node_name = qp.directly_quantized_operator_node_names[0]
                    outputs = onnx_graph.get_node_edges(node_name)['input'][0]
                if outputs in filled_outputs:
                    # TODO: resolve this
                    # Problems with inception v3
                    print(f'Skipping {outputs} layer')
                    continue
                filled_outputs.append(outputs)
                self._activation_quantizers.append(outputs)

        output = []
        for activation_quantizer in self._activation_quantizers:
            axis = 1 if activation_quantizer_config.per_channel else None
            layer_statistics = MinMaxLayerStatistic(activation_quantizer,
                                                    min_value_func=self.activation_statistics_min_func,
                                                    max_value_func=self.activation_statistics_max_func,
                                                    min_batch_aggregator_func=self.batch_aggregation_min_func,
                                                    max_batch_aggregator_func=self.batch_aggregation_max_func,
                                                    statistics_aggregation_func=self.statistics_aggregator_func,
                                                    axis=axis)
            output.append(layer_statistics)

        return output

    def get_transformation_commands(self, layers_statistics) -> List[ONNXInsertionCommand]:
        transformation_commands = []
        original_model = self.compressed_model.original_model
        onnx_graph = ONNXGraph(original_model)

        for weight_quantizer in self._weight_quantizers:
            weight_tensor = onnx_graph.get_initializers_value(weight_quantizer)
            parameters = self._calculate_weight_quantizer_parameters(weight_tensor, self.weight_quantizer_config)
            command = ONNXQuantizerInsertionCommand(weight_quantizer, parameters)
            transformation_commands.append(command)

        for activation_quantizer in self._activation_quantizers:
            for layer_statistics in layers_statistics:
                # TODO: improve perfomance of matching
                if layer_statistics.layer_name == activation_quantizer:
                    parameters = self._calculate_activation_quantizer_parameters(layer_statistics,
                                                                                 self.activation_quantizer_config)
                    command = ONNXQuantizerInsertionCommand(activation_quantizer, parameters)
                    transformation_commands.append(command)

        return transformation_commands

    def _get_default_qconfig(self) -> QuantizerConfig:
        qconfig = deepcopy(self.DEFAULT_QCONFIG)
        return qconfig

    def _get_quantizer_setup(self, compressed_model: CompressedModel):
        nncf_graph = compressed_model.nncf_graph
        nncf_graph.visualize_graph('/home/aleksei/tmp/onnx/onnx_ptq_api/nncf_graph.dot')
        ip_graph = InsertionPointGraph(nncf_graph)
        pattern = ONNX_HW_FUSED_PATTERNS.get_full_pattern_graph()
        ip_graph = ip_graph.get_ip_graph_with_merged_hw_optimized_operations(pattern)

        weight_nodes = nncf_graph.get_nodes_by_metatypes(QUANTIZATION_LAYER_METATYPES)
        quantizable_layer_nodes = [QuantizableWeightedLayerNode(weight_node, [QuantizerConfig()]) for weight_node in
                                   weight_nodes]

        hw_config_type = HWConfigType.from_str(HW_CONFIG_TYPE_TARGET_DEVICE_MAP[self.target_device])
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

    def _calculate_scale_level(self,
                               max_val: Union[float, np.ndarray],
                               min_val: Union[float, np.ndarray],
                               num_bits: int,
                               mode: QuantizationMode):
        # Always full range
        if mode == QuantizationMode.SYMMETRIC:
            input_abs_max = np.maximum(np.abs(max_val), np.abs(min_val))
            return input_abs_max / ((2 ** num_bits - 1) / 2)
        return (max_val - min_val) / 2 ** num_bits

    def _calculate_weight_quantizer_parameters(self, weight_tensor: np.ndarray, quantizer_config: QuantizerConfig) -> \
            Tuple[List[float], List[int], bool]:
        per_channel = quantizer_config.per_channel
        num_bits = quantizer_config.num_bits
        mode = quantizer_config.mode

        if per_channel:
            axes = tuple(range(len(weight_tensor.shape))[1:])
        else:
            axes = None
        input_high = np.amax(weight_tensor, axis=axes)
        input_low = np.amin(weight_tensor, axis=axes)
        scales = self._calculate_scale_level(input_high, input_low, num_bits, mode)
        zero_points = np.zeros_like(scales, dtype=np.int)
        return scales.tolist(), zero_points.tolist(), mode

    def _calculate_activation_quantizer_parameters(self, layer_statistics: MinMaxLayerStatistic,
                                                   quantizer_config: QuantizerConfig) -> \
            Tuple[List[float], List[int], bool]:
        # TODO:PERCHANNEL IS NOT SUPPORTED.
        per_channel = quantizer_config.per_channel
        num_bits = quantizer_config.num_bits
        mode = quantizer_config.mode

        # if per_channel:
        #     axes = tuple(range(len(weight_tensor.shape))[1:])
        # else:
        #     axes = None

        axes = None
        input_high = layer_statistics.get_global_max_value()
        input_low = layer_statistics.get_global_min_value()
        if input_low < 0:
            mode = QuantizationMode.SYMMETRIC
        else:
            mode = QuantizationMode.ASYMMETRIC

        scales = self._calculate_scale_level(input_high, input_low, num_bits, mode)
        zero_points = np.zeros_like(scales, dtype=np.int)

        return scales.tolist(), zero_points.tolist(), mode
