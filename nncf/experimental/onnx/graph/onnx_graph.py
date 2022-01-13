import onnx


class ONNXGraphHelper:
    @staticmethod
    def find_nodes_by_output(output_name: str, graph: onnx.GraphProto):
        output = []
        for node in graph.node:
            if output_name in node.output or output_name == node.output:
                output.append(node)
        return output

    @staticmethod
    def find_nodes_by_input(input_name: str, graph: onnx.GraphProto):
        output = []
        for node in graph.node:
            if input_name in node.input or input_name == node.input:
                output.append(node)
        return output

    @staticmethod
    def get_all_node_inputs(node_name: str, graph: onnx.GraphProto):
        node_inputs = None
        for node in graph.node:
            if node.name == node_name:
                node_inputs = node.input
        return node_inputs

    @staticmethod
    def get_all_node_outputs(node_name: str, graph: onnx.GraphProto):
        node_outputs = None
        for node in graph.node:
            if node.name == node_name:
                node_outputs = node.output
        return node_outputs

    @staticmethod
    def get_nodes_by_type(graph: onnx.GraphProto, node_type: str):
        output = []
        for node in graph.node:
            if str(node.op_type) == node_type:
                output.append(node)
        return output

    @staticmethod
    def find_weight_input_in_module(node_name: str, graph: onnx.GraphProto) -> str:
        node_inputs = ONNXGraphHelper.get_all_node_inputs(node_name, graph)
        # TODO: add search of input weight tensor
        return node_inputs[1]

    @staticmethod
    def get_initializers_value(initializer_name: str, graph: onnx.GraphProto):
        tensor = None
        for init in graph.initializer:
            if init.name == initializer_name:
                tensor = onnx.numpy_helper.to_array(init)
        return tensor

    @staticmethod
    def find_node_output_shape_in_activation_shapes(node_output_name: str, activation_shapes):
        shape = []
        for tensor in activation_shapes:
            if tensor.name == node_output_name:
                for dim in tensor.type.tensor_type.shape.dim:
                    shape.append(dim.dim_value)
        return shape
