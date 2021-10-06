import networkx as nx
import onnx
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs

from nncf.common.quantization.structs import QuantizerConfig


def find_node_by_output(output: str, graph: onnx.GraphProto):
    retval = []
    for node in graph.node:
        if output in node.output or output == node.output:
            retval.append(node)
    return retval


def find_nodes_by_input(input: str, graph: onnx.GraphProto):
    retval = []
    for node in graph.node:
        if input in node.input or input == node.input:
            retval.append(node)
    return retval


def add_quantize_dequantize(nncf_network, quantizer_config: QuantizerConfig, qp_id, weight_tensor_name, scale,
                            zero_point):
    def find_node_index(node_name, onnx_model):
        for i, node in enumerate(onnx_model.graph.node):
            if node.name == node_name:
                return i
        return 0

    onnx_model = nncf_network.onnx_compressed_model
    name = str(qp_id)
    if quantizer_config.per_channel:
        onnx_scale = onnx.helper.make_tensor('scale_' + name, onnx.TensorProto.FLOAT, scale.shape, scale)
    else:
        onnx_scale = onnx.helper.make_tensor('scale_' + name, onnx.TensorProto.FLOAT, [], [scale])
    if quantizer_config.signedness_to_force:
        if quantizer_config.per_channel:
            onnx_zero_point = onnx.helper.make_tensor('zero_point_' + name, onnx.TensorProto.INT8, scale.shape,
                                                      [zero_point] * scale.shape[0])
        else:
            onnx_zero_point = onnx.helper.make_tensor('zero_point_' + name, onnx.TensorProto.INT8, [], [zero_point])
    else:
        if quantizer_config.per_channel:
            onnx_zero_point = onnx.helper.make_tensor('zero_point_' + name, onnx.TensorProto.UINT8, scale.shape,
                                                      [zero_point] * scale.shape[0])
        else:
            onnx_zero_point = onnx.helper.make_tensor('zero_point_' + name, onnx.TensorProto.UINT8, [], [zero_point])

    quantizer = onnx.helper.make_node(
        'QuantizeLinear',  # name
        [weight_tensor_name, 'scale_' + name, 'zero_point_' + name],  # inputs
        ['q_output_' + name]  # outputs
    )

    dequantizer = onnx.helper.make_node(
        'DequantizeLinear',  # name
        ['q_output_' + name, 'scale_' + name, 'zero_point_' + name],  # inputs
        ['dq_output_' + name]  # outputs
    )
    input_nodes = find_nodes_by_input(weight_tensor_name, onnx_model.graph)
    for node in input_nodes:
        for i, inp in enumerate(node.input):
            if inp == weight_tensor_name:
                node.input[i] = 'dq_output_' + name
    onnx_model.graph.initializer.extend([onnx_scale])
    onnx_model.graph.initializer.extend([onnx_zero_point])
    i = find_node_index(input_nodes[0].name, onnx_model)
    onnx_model.graph.node.insert(i, quantizer)
    onnx_model.graph.node.insert(i + 1, dequantizer)


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
    # TODO: add search of input weight tensor
    return node_inputs[1]


def get_initializers_value(initializer_name, onnx_model_graph):
    from onnx import numpy_helper
    for init in onnx_model_graph.initializer:
        if init.name == initializer_name:
            tensor = numpy_helper.to_array(init)
    return tensor


def find_nx_graph_node_by_label(label, nx_graph):
    for node, attrs in nx_graph.nodes(data=True):
        if attrs['label'] == label:
            return node
    return None


def dump_graph(nx_graph, path: str):
    nx.drawing.nx_pydot.write_dot(nx_graph, path)


def find_output_shape(output, activation_shapes):
    shape = []
    for tensor in activation_shapes:
        if tensor.name == output:
            for dim in tensor.type.tensor_type.shape.dim:
                shape.append(dim.dim_value)
    return shape


def add_edges_for_nodes(nx_graph, onnx_model):
    inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
    activations_shapes = inferred_model.graph.value_info
    for node, attrs in nx_graph.nodes(data=True):
        outputs = attrs['output']
        for output in outputs:
            nodes = find_nodes_by_input(output, onnx_model.graph)
            shape = find_output_shape(output, activations_shapes)
            for in_node in nodes:
                nx_graph_in_node = find_nx_graph_node_by_label(in_node.name, nx_graph)
                nx_graph.add_edge(node, nx_graph_in_node, shape)


def get_nodes_by_type(onnx_model: onnx.ModelProto, node_type: str):
    retval = []
    for node in onnx_model.graph.node:
        if str(node.op_type) == node_type:
            retval.append(node)
    return retval


def add_output_layers_for_all_convs(onnx_model):
    nodes = get_nodes_by_type(onnx_model, 'Conv')
    outputs = [node.output[0] for node in nodes]
    model_with_intermediate_outputs = select_model_inputs_outputs(onnx_model, outputs=outputs)


def get_input_tensor():
    ...
