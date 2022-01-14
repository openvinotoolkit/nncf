import onnx


# pylint: disable=no-member

class ONNXGraph:
    def __init__(self, onnx_model: onnx.ModelProto):
        self.onnx_model = onnx_model
        self.model_with_shapes = onnx.shape_inference.infer_shapes(self.onnx_model)

    def find_nodes_by_output(self, output_name: str):
        output = []
        graph = self.onnx_model.graph
        for node in graph.node:
            if output_name in node.output or output_name == node.output:
                output.append(node)
        return output

    def find_nodes_by_input(self, input_name: str):
        output = []
        graph = self.onnx_model.graph
        for node in graph.node:
            if input_name in node.input or input_name == node.input:
                output.append(node)
        return output

    def get_all_nodes(self):
        return self.onnx_model.graph.node

    def get_all_model_inputs(self):
        return self.onnx_model.graph.input

    def get_all_node_inputs(self, node_name: str):
        node_inputs = None
        graph = self.onnx_model.graph
        for node in graph.node:
            if node.name == node_name:
                node_inputs = node.input
        return node_inputs

    def get_all_node_outputs(self, node_name: str):
        node_outputs = None
        graph = self.onnx_model.graph
        for node in graph.node:
            if node.name == node_name:
                node_outputs = node.output
        return node_outputs

    def get_nodes_by_type(self, node_type: str):
        output = []
        graph = self.onnx_model.graph
        for node in graph.node:
            if str(node.op_type) == node_type:
                output.append(node)
        return output

    def find_weight_input_in_module(self, node_name: str) -> str:
        node_inputs = self.get_all_node_inputs(node_name)
        # TODO: add search of input weight tensor
        return node_inputs[1]

    def get_initializers_value(self, initializer_name: str):
        tensor = None
        graph = self.onnx_model.graph
        for init in graph.initializer:
            if init.name == initializer_name:
                tensor = onnx.numpy_helper.to_array(init)
        return tensor

    def find_node_output_shape(self, node_output_name: str):
        shape = []
        activations_shapes = self.model_with_shapes.graph.value_info
        for tensor in activations_shapes:
            if tensor.name == node_output_name:
                for dim in tensor.type.tensor_type.shape.dim:
                    shape.append(dim.dim_value)
        return shape
