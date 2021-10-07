import onnx
import torch

from nncf.experimental.onnx.graph.onnx_graph_helpers import get_initializers_value
from nncf.experimental.onnx.graph.onnx_graph_helpers import find_weight_input_in_module
from nncf.experimental.onnx.graph.onnx_graph_helpers import get_all_node_outputs
from nncf.experimental.onnx.graph.onnx_graph_helpers import get_all_node_inputs
from nncf.experimental.onnx.graph.onnx_graph_helpers import add_quantize_dequantize

from nncf.experimental.onnx.quantization.quantizer_initialization import calculate_statistics_for_activation_quantizer
from nncf.experimental.onnx.quantization.quantizer_initialization import calculate_statistics_for_weight_quantizer
from nncf.experimental.onnx.quantization.quantizer_initialization import calculate_scale_level

from nncf.experimental.onnx.nncf_network import NNCFNetwork
from nncf.experimental.onnx.graph.metatypes.onnx_ops import GENERAL_WIGHT_LAYER_METATYPES
from nncf.experimental.onnx.quantization.default_quantization import DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT

from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.insertion_point_graph import InsertionPointGraph

from nncf.experimental.onnx.hardware.fused_patterns import ONNX_HW_FUSED_PATTERNS

QUANTIZATION_LAYER_METATYPES = GENERAL_WIGHT_LAYER_METATYPES


def choose_activation_quantization_configuration(nncf_network: NNCFNetwork, outputs: onnx.TensorProto,
                                                 dataloader: torch.utils.data.DataLoader,
                                                 num_initialization_samples: int) -> [QuantizerConfig, float, int]:
    max_val, min_val = calculate_statistics_for_activation_quantizer(nncf_network.onnx_model,
                                                                     outputs,
                                                                     dataloader,
                                                                     num_initialization_samples)

    signedness_to_force = True if min_val < 0 else False
    num_bits = 8
    quantizer_config = QuantizerConfig(num_bits=num_bits, signedness_to_force=signedness_to_force, per_channel=False)
    scale = calculate_scale_level(max_val, min_val, num_bits)
    return quantizer_config, scale, 0


def choose_weight_quantization_configuration(outputs: onnx.TensorProto) -> [QuantizerConfig, float, int]:
    num_bits = 8
    per_channel = False
    scale = calculate_statistics_for_weight_quantizer(outputs, num_bits, per_channel)
    signedness_to_force = True
    quantizer_config = QuantizerConfig(num_bits=num_bits, signedness_to_force=signedness_to_force,
                                       per_channel=per_channel)
    return quantizer_config, scale, 0


def apply_post_training_quantization(onnx_model: onnx.ModelProto, data_loader: torch.utils.data.DataLoader,
                                     initialization_number: int) -> onnx.ModelProto:
    nncf_network = NNCFNetwork(onnx_model)
    nncf_graph = nncf_network.nncf_graph

    ip_graph = InsertionPointGraph(nncf_graph)
    pattern = ONNX_HW_FUSED_PATTERNS.get_full_pattern_graph()
    ip_graph = ip_graph.get_ip_graph_with_merged_hw_optimized_operations(pattern)

    weight_nodes = nncf_graph.get_nodes_by_metatypes(QUANTIZATION_LAYER_METATYPES)
    quantizable_layer_nodes = [QuantizableWeightedLayerNode(weight_node, [QuantizerConfig()]) for weight_node in
                               weight_nodes]
    solver = QuantizerPropagationSolver(default_trait_to_metatype_map=DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT,
                                        quantizable_layer_nodes=quantizable_layer_nodes)

    quantization_proposal = solver.run_on_ip_graph(ip_graph)
    multi_config_setup = quantization_proposal.quantizer_setup
    single_config_setup = multi_config_setup.select_first_qconfig_for_each_point()
    finalized_proposal = quantization_proposal.finalize(single_config_setup)
    final_setup = solver.get_final_quantizer_setup(finalized_proposal)

    for qp_id, qp in final_setup.quantization_points.items():
        if qp.is_weight_quantization_point():
            weight_initializer_name = find_weight_input_in_module(qp.insertion_point.target_node_name,
                                                                  nncf_network.onnx_model.graph)
            weight_tensor = get_initializers_value(weight_initializer_name, nncf_network.onnx_model.graph)
            quantizer_config, scale, zero_point = choose_weight_quantization_configuration(weight_tensor)
            add_quantize_dequantize(nncf_network, quantizer_config, qp_id, weight_initializer_name, scale,
                                    zero_point)
        else:
            assert qp.is_activation_quantization_point()
            if not qp.insertion_point.target_node_name == 'input_node':
                node_name = qp.insertion_point.target_node_name
                outputs = get_all_node_outputs(node_name, nncf_network.onnx_model.graph)
            else:
                node_name = qp.directly_quantized_operator_node_names[0]
                outputs = get_all_node_inputs(node_name, nncf_network.onnx_model.graph)
            quantizer_config, scale, zero_point = choose_activation_quantization_configuration(nncf_network,
                                                                                               outputs, data_loader,
                                                                                               initialization_number)
            add_quantize_dequantize(nncf_network, quantizer_config, qp_id, outputs[0], scale, zero_point)
    onnx.checker.check_model(nncf_network.onnx_compressed_model)
    return nncf_network.onnx_compressed_model
