from onnx import AttributeProto, TensorProto, GraphProto
from onnx import helper
import onnx
from nncf.experimental.onnx.graph.helpers import find_nodes_by_input

import numpy as np

from nncf.experimental.onnx.nncf_network import NNCFNetwork
from nncf.experimental.onnx.graph.metatypes.onnx_ops import GENERAL_CONV_LAYER_METATYPES
from nncf.experimental.onnx.quantization.default_quantization import DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT

from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.quantization.structs import QuantizerSpec, QuantizationMode

from nncf.experimental.onnx.hardware.fused_patterns import ONNX_HW_FUSED_PATTERNS

from nncf.experimental.onnx.graph.helpers import find_node_by_output

QUANTIZATION_LAYER_METATYPES = GENERAL_CONV_LAYER_METATYPES


def get_all_node_inputs(module_name, onnx_model_graph):
    node_inputs = None
    for node in onnx_model_graph.node:
        if node.name == module_name:
            node_inputs = node.input
    return node_inputs


def get_all_node_outputs(module_name, onnx_model_graph):
    node_inputs = None
    for node in onnx_model_graph.node:
        if node.name == module_name:
            node_outputs = node.output
    return node_outputs


def find_weight_input_in_module(module_name, onnx_model_graph) -> str:
    node_inputs = get_all_node_inputs(module_name, onnx_model_graph)
    #TODO: add search of input weight tensor
    return node_inputs[1]


def get_initializers_value(initializer_name, onnx_model_graph):
    from onnx import numpy_helper
    for init in onnx_model_graph.initializer:
        if init.name == initializer_name:
            tensor = numpy_helper.to_array(init)
    return tensor


def collect_tensor_statistics(tensor):
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    scale = (max_val - min_val) / 256
    zero_point = 0
    return scale, zero_point


def calculate_statistics_for_activation_quantizer(outputs, nncf_network, data_loader, num_iters):
    from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
    import onnxruntime as rt
    onnx_model = nncf_network.onnx_model
    model_with_intermediate_outputs = select_model_inputs_outputs(onnx_model, outputs=outputs[0])
    onnx.save(model_with_intermediate_outputs, '/home/aleksei/nncf_work/onnx_quantization/model_with_intermediate_outputs.onnx')

    sess = rt.InferenceSession('/home/aleksei/nncf_work/onnx_quantization/model_with_intermediate_outputs.onnx')
    input_name = sess.get_inputs()[0].name
    avg_scale, zero_point = 0, 0
    for i, (input_, target) in enumerate(data_loader):
        input_tensor = input_.cpu().detach().numpy()
        img_size = [1, 3, 224, 224]

        output_tensor = sess.run([], {input_name: input_tensor.astype(np.float32)})
        scale, zero_point = collect_tensor_statistics(output_tensor)
        avg_scale += scale
        if i == num_iters:
            break
    return avg_scale / num_iters, zero_point


def add_quantize_dequantize(qp_id, onnx_model, weight_tensor_name, scale, zero_point):


    def find_node_index(node_name, onnx_model):
        for i, node in enumerate(onnx_model.graph.node):
            if node.name == node_name:
                return i

    name = str(qp_id)
    scale = helper.make_tensor('scale_' + name, TensorProto.FLOAT, [], [scale])
    zero_point = helper.make_tensor('zero_point_' + name, TensorProto.INT8, [], [zero_point])

    quantizer = helper.make_node(
        'QuantizeLinear',  # name
        [weight_tensor_name, 'scale_' + name, 'zero_point_' + name],  # inputs
        ['q_output_' + name]  # outputs
    )

    dequantizer = helper.make_node(
        'DequantizeLinear',  # name
        ['q_output_' + name, 'scale_' + name, 'zero_point_' + name],  # inputs
        ['dq_output_' + name]  # outputs
    )
    conv = find_nodes_by_input(weight_tensor_name, onnx_model.graph)
    for i, inp in enumerate(conv[0].input):
        if inp == weight_tensor_name:
            conv[0].input[i] = 'dq_output_' + name
    onnx_model.graph.initializer.extend([scale])
    onnx_model.graph.initializer.extend([zero_point])
    i = find_node_index(conv[0].name, onnx_model)
    onnx_model.graph.node.insert(i, quantizer)
    onnx_model.graph.node.insert(i + 1, dequantizer)


def apply_quantization(nncf_network: NNCFNetwork, data_loader):
    num_iters = 2
    original_graph = nncf_network._original_graph

    ip_graph = InsertionPointGraph(original_graph, ['Conv'])
    pattern = ONNX_HW_FUSED_PATTERNS.get_full_pattern_graph()
    ip_graph = ip_graph.get_ip_graph_with_merged_hw_optimized_operations(pattern)

    conv_nodes = original_graph.get_nodes_by_metatypes(QUANTIZATION_LAYER_METATYPES)
    quantizable_layer_nodes = [QuantizableWeightedLayerNode(conv_node, [QuantizerConfig()]) for conv_node in conv_nodes]
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
            metatype = target_node.metatype

            quantizer_spec = QuantizerSpec(8, QuantizationMode.SYMMETRIC, False, False, False)
            weight_initializer_name = find_weight_input_in_module(qp.insertion_point.target_node_name, nncf_network.onnx_model.graph)
            weight_tensor = get_initializers_value(weight_initializer_name, nncf_network.onnx_model.graph)
            scale, zero_point = collect_tensor_statistics(weight_tensor)

            add_quantize_dequantize(qp_id, nncf_network.onnx_model, weight_initializer_name, scale, zero_point)
        else:
            assert qp.is_activation_quantization_point()
            outputs = get_all_node_outputs(qp.insertion_point.target_node_name, nncf_network.onnx_model.graph)
            scale, zero_point = calculate_statistics_for_activation_quantizer(outputs, nncf_network, data_loader, num_iters)
            add_quantize_dequantize(qp_id, nncf_network.onnx_model, outputs[0], scale, zero_point)


    import onnx
    onnx.save(nncf_network.onnx_model, '/home/aleksei/nncf_work/onnx_quantization/quantized.onnx')


