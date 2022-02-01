from nncf.experimental.post_training.graph.model_analyzer import ModelAnalyzer
from nncf.experimental.post_training.compressed_model import CompressedModel
from nncf.experimental.onnx.graph.onnx_graph import ONNXGraph

from nncf.experimental.onnx.graph.metatypes.onnx_ops import GENERAL_WEIGHT_LAYER_METATYPES
from nncf.experimental.onnx.quantization.default_quantization import DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT

from nncf.experimental.onnx.graph.transformations.commands import ONNXInsertionCommand
from nncf.experimental.onnx.graph.transformations.layout import ONNXTransformationLayout

from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.insertion_point_graph import InsertionPointGraph

from nncf.experimental.onnx.hardware.fused_patterns import ONNX_HW_FUSED_PATTERNS

QUANTIZATION_LAYER_METATYPES = GENERAL_WEIGHT_LAYER_METATYPES


class ONNXModelAnalyzer(ModelAnalyzer):

    def get_quantization_transformations(self, compressed_model: CompressedModel):
        transformation_layout = ONNXTransformationLayout()
        quantizer_setup = self._get_quantizer_setup(compressed_model)
        onnx_graph = ONNXGraph(compressed_model.original_model)
        for qp_id, qp in quantizer_setup.quantization_points.items():
            if qp.is_weight_quantization_point():
                weight_initializer_name = onnx_graph.find_weight_input_in_module(qp.insertion_point.target_node_name)
                weight_tensor = onnx_graph.get_initializers_value(weight_initializer_name)
                transformation_layout.register(ONNXInsertionCommand(weight_initializer_name, weight_tensor, True))
            else:
                assert qp.is_activation_quantization_point()
                if not 'model_input' in qp.insertion_point.target_node_name:
                    node_name = qp.insertion_point.target_node_name
                    outputs = onnx_graph.get_node_edges(node_name)['output']
                else:
                    node_name = qp.directly_quantized_operator_node_names[0]
                    outputs = onnx_graph.get_node_edges(node_name)['input']
                import numpy as np
                transformation_layout.register(ONNXInsertionCommand(outputs[0], np.random.random((10, 10)), False))
        return transformation_layout

    def _get_quantizer_setup(self, compressed_model: CompressedModel):
        nncf_graph = compressed_model.nncf_graph

        ip_graph = InsertionPointGraph(nncf_graph)
        pattern = ONNX_HW_FUSED_PATTERNS.get_full_pattern_graph()
        ip_graph = ip_graph.get_ip_graph_with_merged_hw_optimized_operations(pattern)

        weight_nodes = nncf_graph.get_nodes_by_metatypes(QUANTIZATION_LAYER_METATYPES)
        quantizable_layer_nodes = [QuantizableWeightedLayerNode(weight_node, [QuantizerConfig()]) for weight_node in
                                   weight_nodes]
        solver = QuantizerPropagationSolver(ignored_scopes=None,
                                            default_trait_to_metatype_map=DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT,
                                            quantizable_layer_nodes=quantizable_layer_nodes)

        quantization_proposal = solver.run_on_ip_graph(ip_graph)
        multi_config_setup = quantization_proposal.quantizer_setup
        single_config_setup = multi_config_setup.select_first_qconfig_for_each_point()
        finalized_proposal = quantization_proposal.finalize(single_config_setup)
        final_setup = solver.get_final_quantizer_setup(finalized_proposal)
        return final_setup

    def get_sparsity_transformations(self, compressed_model: CompressedModel):
        pass
