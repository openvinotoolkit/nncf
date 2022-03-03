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

from nncf.common.hardware.config import HWConfigType
from nncf.common.hardware.config import HW_CONFIG_TYPE_TARGET_DEVICE_MAP
from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.insertion_point_graph import InsertionPointGraph

from nncf.experimental.post_training.algorithms.quantizer_range_finder import QuantizerRangeFinderAlgorithm
from nncf.experimental.post_training.compressed_model import CompressedModel
from nncf.experimental.onnx.graph.transformations.layout import ONNXTransformationLayout
from nncf.experimental.onnx.graph.metatypes.onnx_ops import GENERAL_WEIGHT_LAYER_METATYPES
from nncf.experimental.onnx.algorithms.quantization.default_quantization import DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT
from nncf.experimental.onnx.graph.transformations.commands import ONNXQuantizerInsertionCommand
from nncf.experimental.onnx.engine import ONNXEngine
from nncf.experimental.post_training.algorithms.quantizer_range_finder import QuantizerRangeFinderParameters
from nncf.experimental.onnx.statistics.collectors import ONNXMinMaxStatisticCollector
from nncf.experimental.onnx.graph.model_transformer import ONNXModelTransformer

from nncf.experimental.onnx.hardware.fused_patterns import ONNX_HW_FUSED_PATTERNS

from nncf.experimental.onnx.algorithms.quantization.helper import calculate_activation_quantizer_parameters
from nncf.experimental.onnx.algorithms.quantization.helper import calculate_weight_quantizer_parameters

from nncf.experimental.onnx.hardware.config import ONNXHWConfig

QUANTIZATION_LAYER_METATYPES = GENERAL_WEIGHT_LAYER_METATYPES


class ONNXQuantizerRangeFinderAlgorithm(QuantizerRangeFinderAlgorithm):

    def __init__(self, statistics_collector,
                 parameters: QuantizerRangeFinderParameters):
        super().__init__(statistics_collector, parameters)
        self._weight_quantizers = []
        self._activation_quantizers = []
        self.global_quantizer_constraints = {}

    def generate_stat_collector(self, quantizer_config: QuantizerConfig):
        if self.range_type == 'min_max':
            is_symmetric = True if quantizer_config.mode == QuantizationMode.SYMMETRIC else False
            if quantizer_config.per_channel:
                axes = (0, 2, 3)
                return ONNXMinMaxStatisticCollector(is_symmetric, axes)
            return ONNXMinMaxStatisticCollector(is_symmetric, None)

    def _create_model_transformer(self, compressed_model):
        self.model_transformer = ONNXModelTransformer(compressed_model)

    def _get_quantizer_setup(self, compressed_model: CompressedModel):
        nncf_graph = compressed_model.nncf_graph
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

    def apply(self, compressed_model: CompressedModel, engine: ONNXEngine):
        self._create_model_transformer(compressed_model)
        transformation_layout = ONNXTransformationLayout()
        transformation_commands = []
        onnx_graph = compressed_model.original_onnx_graph
        layers_statistics = self.statistics_collector.layers_statistics

        for weight_quantizer in self._weight_quantizers:
            weight_tensor = onnx_graph.get_initializers_value(weight_quantizer)
            parameters = calculate_weight_quantizer_parameters(weight_tensor, self.weight_quantizer_config)
            command = ONNXQuantizerInsertionCommand(weight_quantizer, parameters)
            transformation_commands.append(command)
        # We are sure that layer_statistics math self._activation_quantizers
        for layer_statistics in layers_statistics:
            for k, v in layer_statistics.items():
                parameters = calculate_activation_quantizer_parameters(v,
                                                                       self.activation_quantizer_config)
                command = ONNXQuantizerInsertionCommand(k, parameters)
                transformation_commands.append(command)

        for transformation_command in transformation_commands:
            transformation_layout.register(transformation_command)

        quantized_model = self.model_transformer.transform(transformation_layout)
        return quantized_model

    def get_layers_for_statistics(self, compressed_model: CompressedModel):
        quantizer_setup = self._get_quantizer_setup(compressed_model)
        onnx_graph = compressed_model.original_onnx_graph
        filled_outputs = []
        for qp_id, qp in quantizer_setup.quantization_points.items():
            if qp.is_weight_quantization_point():
                weight_initializer_name = onnx_graph.get_weight_input_in_module(qp.insertion_point.target_node_name)
                self._weight_quantizers.append(weight_initializer_name)
            else:
                assert qp.is_activation_quantization_point()
                if 'model_input' not in qp.insertion_point.target_node_name:  # If not input node
                    node_name = qp.insertion_point.target_node_name
                    if qp.insertion_point.input_port_id == 1:
                        # If quantization of Input
                        outputs = onnx_graph.get_node_edges(node_name)['input'][0]
                    else:
                        # If quantization of Output
                        outputs = onnx_graph.get_node_edges(node_name)['output'][0]
                else:  # If input node
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
            layer_statistics = {activation_quantizer: self.generate_stat_collector(self.activation_quantizer_config)}
            output.append(layer_statistics)

        return output
