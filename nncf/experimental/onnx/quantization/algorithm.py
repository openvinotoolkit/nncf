

from nncf.experimental.onnx.graph.onnx_graph_helpers import get_initializers_value
from nncf.experimental.onnx.graph.onnx_graph_helpers import find_weight_input_in_module
from nncf.experimental.onnx.graph.onnx_graph_helpers import get_all_node_outputs
from nncf.experimental.onnx.graph.onnx_graph_helpers import get_all_node_inputs
from nncf.experimental.onnx.graph.onnx_graph_helpers import add_quantize_dequantize

from nncf.experimental.onnx.quantization.quantizer_initialization import calculate_statistics_for_activation_quantizer
from nncf.experimental.onnx.quantization.quantizer_initialization import collect_tensor_statistics

from nncf.experimental.onnx.nncf_network import NNCFNetwork
from nncf.experimental.onnx.graph.metatypes.onnx_ops import GENERAL_WIGHT_LAYER_METATYPES
from nncf.experimental.onnx.quantization.default_quantization import DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT

from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.quantization.structs import QuantizerSpec, QuantizationMode

from nncf.experimental.onnx.hardware.fused_patterns import ONNX_HW_FUSED_PATTERNS


QUANTIZATION_LAYER_METATYPES = GENERAL_WIGHT_LAYER_METATYPES


def apply_quantization(nncf_network: NNCFNetwork, data_loader):
    num_iters = 2
    original_graph = nncf_network._original_graph

    ip_graph = InsertionPointGraph(original_graph)
    pattern = ONNX_HW_FUSED_PATTERNS.get_full_pattern_graph()
    ip_graph = ip_graph.get_ip_graph_with_merged_hw_optimized_operations(pattern)

    weight_nodes = original_graph.get_nodes_by_metatypes(QUANTIZATION_LAYER_METATYPES)
    quantizable_layer_nodes = [QuantizableWeightedLayerNode(weight_node, [QuantizerConfig()]) for weight_node in weight_nodes]
    solver = QuantizerPropagationSolver(default_trait_to_metatype_map=DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT,
                                        quantizable_layer_nodes=quantizable_layer_nodes)

    quantization_proposal = solver.run_on_ip_graph(ip_graph)
    multi_config_setup = quantization_proposal.quantizer_setup
    single_config_setup = multi_config_setup.select_first_qconfig_for_each_point()
    finalized_proposal = quantization_proposal.finalize(single_config_setup)
    final_setup = solver.get_final_quantizer_setup(finalized_proposal)

    for qp_id, qp in final_setup.quantization_points.items():
        if qp.is_weight_quantization_point():
            target_node = original_graph.get_node_by_name(qp.insertion_point.target_node_name)

            quantizer_spec = QuantizerSpec(8, QuantizationMode.SYMMETRIC, False, False, False)
            weight_initializer_name = find_weight_input_in_module(qp.insertion_point.target_node_name, nncf_network.onnx_model.graph)
            weight_tensor = get_initializers_value(weight_initializer_name, nncf_network.onnx_model.graph)
            scale, zero_point = collect_tensor_statistics(weight_tensor)
            add_quantize_dequantize(qp_id, nncf_network.onnx_nncf_model, weight_initializer_name, scale, zero_point)
        else:
            assert qp.is_activation_quantization_point()
            if not qp.insertion_point.target_node_name == 'input_node':
                outputs = get_all_node_outputs(qp.insertion_point.target_node_name, nncf_network.onnx_model.graph)
                scale, zero_point = calculate_statistics_for_activation_quantizer(outputs, nncf_network, data_loader, num_iters)
                add_quantize_dequantize(qp_id, nncf_network.onnx_nncf_model, outputs[0], scale, zero_point)
            else:
                node_name = qp.directly_quantized_operator_node_names[0]
                outputs = get_all_node_inputs(node_name, nncf_network.onnx_model.graph)
                scale, zero_point = calculate_statistics_for_activation_quantizer(outputs, nncf_network, data_loader,
                                                                                  num_iters)
                add_quantize_dequantize(qp_id, nncf_network.onnx_nncf_model, outputs[0], scale, zero_point)

    import onnx
    onnx.save(nncf_network.onnx_nncf_model, '/home/aleksei/nncf_work/onnx_quantization/quantized.onnx')


