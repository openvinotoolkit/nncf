import networkx as nx
import onnx

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
    from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
    nodes = get_nodes_by_type(onnx_model, 'Conv')
    outputs = [node.output[0] for node in nodes]
    model_with_intermediate_outputs = select_model_inputs_outputs(onnx_model, outputs=outputs)