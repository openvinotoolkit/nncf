import numpy as np
import onnx


def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:
    # (TensorProto)
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())

    return initializer_tensor


class ONNXReferenceModel:
    def __init__(self, onnx_model, graph_path):
        self.onnx_model = onnx_model
        self.path_ref_graph = graph_path


class LinearModel(ONNXReferenceModel):
    def __init__(self):
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [1, 3, 32, 32])
        model_output_name = "Y"
        model_output_channels = 10
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [1, model_output_channels, 1, 1])

        # Create a Conv node (NodeProto).
        # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#conv
        conv1_output_node_name = "Conv1_Y"
        # Dummy weights for conv.
        conv1_in_channels = 3
        conv1_out_channels = 32
        conv1_kernel_shape = (3, 3)
        conv1_pads = (1, 1, 1, 1)
        conv1_W = np.ones(shape=(conv1_out_channels, conv1_in_channels,
                                 *conv1_kernel_shape)).astype(np.float32)
        conv1_B = np.ones(shape=(conv1_out_channels)).astype(np.float32)
        # Create the initializer tensor for the weights.
        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = create_initializer_tensor(
            name=conv1_W_initializer_tensor_name,
            tensor_array=conv1_W,
            data_type=onnx.TensorProto.FLOAT)
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = create_initializer_tensor(
            name=conv1_B_initializer_tensor_name,
            tensor_array=conv1_B,
            data_type=onnx.TensorProto.FLOAT)

        conv1_node = onnx.helper.make_node(
            name="Conv1",  # Name is optional.
            op_type="Conv",
            # Must follow the order of input and output definitions.
            # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
            inputs=[
                model_input_name, conv1_W_initializer_tensor_name,
                conv1_B_initializer_tensor_name
            ],
            outputs=[conv1_output_node_name],
            # The following arguments are attributes.
            kernel_shape=conv1_kernel_shape,
            # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
            pads=conv1_pads,
        )

        # Create a BatchNorm node (NodeProto).
        bn1_output_node_name = "BN1_Y"
        # Dummy paramters for batchnorm.
        bn1_scale = np.random.randn(conv1_out_channels).astype(np.float32)
        bn1_bias = np.random.randn(conv1_out_channels).astype(np.float32)
        bn1_mean = np.random.randn(conv1_out_channels).astype(np.float32)
        bn1_var = np.random.rand(conv1_out_channels).astype(np.float32)
        # Create the initializer tensors.
        bn1_scale_initializer_tensor_name = "BN1_Scale"
        bn1_bias_initializer_tensor_name = "BN1_Bias"
        bn1_mean_initializer_tensor_name = "BN1_Mean"
        bn1_var_initializer_tensor_name = "BN1_Var"
        bn1_scale_initializer_tensor = create_initializer_tensor(
            name=bn1_scale_initializer_tensor_name,
            tensor_array=bn1_scale,
            data_type=onnx.TensorProto.FLOAT)
        bn1_bias_initializer_tensor = create_initializer_tensor(
            name=bn1_bias_initializer_tensor_name,
            tensor_array=bn1_bias,
            data_type=onnx.TensorProto.FLOAT)
        bn1_mean_initializer_tensor = create_initializer_tensor(
            name=bn1_mean_initializer_tensor_name,
            tensor_array=bn1_mean,
            data_type=onnx.TensorProto.FLOAT)
        bn1_var_initializer_tensor = create_initializer_tensor(
            name=bn1_var_initializer_tensor_name,
            tensor_array=bn1_var,
            data_type=onnx.TensorProto.FLOAT)

        bn1_node = onnx.helper.make_node(
            name="BN1",  # Name is optional.
            op_type="BatchNormalization",
            inputs=[
                conv1_output_node_name, bn1_scale_initializer_tensor_name,
                bn1_bias_initializer_tensor_name, bn1_mean_initializer_tensor_name,
                bn1_var_initializer_tensor_name
            ],
            outputs=[bn1_output_node_name],
        )

        # Create a ReLU node (NodeProto).
        relu1_output_node_name = "ReLU1_Y"

        relu1_node = onnx.helper.make_node(
            name="ReLU1",  # Name is optional.
            op_type="Relu",
            inputs=[bn1_output_node_name],
            outputs=[relu1_output_node_name],
        )

        # Create a GlobalAveragePool node (NodeProto).
        avg_pool1_output_node_name = "Avg_Pool1_Y"

        avg_pool1_node = onnx.helper.make_node(
            name="Avg_Pool1",  # Name is optional.
            op_type="GlobalAveragePool",
            inputs=[relu1_output_node_name],
            outputs=[avg_pool1_output_node_name],
        )

        # Create a Conv node (NodeProto).
        # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#conv
        # Dummy weights for conv.
        conv2_in_channels = conv1_out_channels
        conv2_out_channels = model_output_channels
        conv2_kernel_shape = (1, 1)
        conv2_pads = (0, 0, 0, 0)
        conv2_W = np.ones(shape=(conv2_out_channels, conv2_in_channels,
                                 *conv2_kernel_shape)).astype(np.float32)
        conv2_B = np.ones(shape=(conv2_out_channels)).astype(np.float32)
        # Create the initializer tensor for the weights.
        conv2_W_initializer_tensor_name = "Conv2_W"
        conv2_W_initializer_tensor = create_initializer_tensor(
            name=conv2_W_initializer_tensor_name,
            tensor_array=conv2_W,
            data_type=onnx.TensorProto.FLOAT)
        conv2_B_initializer_tensor_name = "Conv2_B"
        conv2_B_initializer_tensor = create_initializer_tensor(
            name=conv2_B_initializer_tensor_name,
            tensor_array=conv2_B,
            data_type=onnx.TensorProto.FLOAT)

        conv2_node = onnx.helper.make_node(
            name="Conv2",
            op_type="Conv",
            inputs=[
                avg_pool1_output_node_name, conv2_W_initializer_tensor_name,
                conv2_B_initializer_tensor_name
            ],
            outputs=[model_output_name],
            kernel_shape=conv2_kernel_shape,
            pads=conv2_pads,
        )

        # Create the graph (GraphProto)
        graph_def = onnx.helper.make_graph(
            nodes=[conv1_node, bn1_node, relu1_node, avg_pool1_node, conv2_node],
            name="ConvNet",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
            initializer=[
                conv1_W_initializer_tensor, conv1_B_initializer_tensor,
                bn1_scale_initializer_tensor, bn1_bias_initializer_tensor,
                bn1_mean_initializer_tensor, bn1_var_initializer_tensor,
                conv2_W_initializer_tensor, conv2_B_initializer_tensor
            ],
        )

        # Create the model (ModelProto)
        model_def = onnx.helper.make_model(graph_def, producer_name="onnx-example")
        model_def.opset_import[0].version = 13

        model_def = onnx.shape_inference.infer_shapes(model_def)

        super().__init__(model_def, 'linear_model.dot')


class MultiInputOutputModel(ONNXReferenceModel):
    def __init__(self):
        model_input_name_1 = "X_1"
        X_1 = onnx.helper.make_tensor_value_info(model_input_name_1,
                                                 onnx.TensorProto.FLOAT,
                                                 [1, 6, 3, 3])
        model_input_name_2 = "X_2"
        X_2 = onnx.helper.make_tensor_value_info(model_input_name_2,
                                                 onnx.TensorProto.FLOAT,
                                                 [2, 6, 3, 3])

        model_input_name_3 = "X_3"
        X_3 = onnx.helper.make_tensor_value_info(model_input_name_3,
                                                 onnx.TensorProto.FLOAT,
                                                 [3, 6, 3, 3])

        model_output_name_1 = "Y_1"
        model_output_channels = 10
        Y_1 = onnx.helper.make_tensor_value_info(model_output_name_1,
                                                 onnx.TensorProto.FLOAT,
                                                 [6, 6, 3, 3])

        model_output_name_2 = "Y_2"
        model_output_channels = 10
        Y_2 = onnx.helper.make_tensor_value_info(model_output_name_2,
                                                 onnx.TensorProto.FLOAT,
                                                 [2, 6, 3, 3])

        concat_node = onnx.helper.make_node(
            name="Concat1",  # Name is optional.
            op_type="Concat",
            inputs=[
                model_input_name_1, model_input_name_2, model_input_name_3
            ],
            outputs=[model_output_name_1],
            axis=0
        )

        add_node = onnx.helper.make_node(
            name="Add1",
            op_type="Add",
            inputs=[
                model_input_name_1, model_input_name_2
            ],
            outputs=[model_output_name_2]
        )

        # Create the graph (GraphProto)
        graph_def = onnx.helper.make_graph(
            nodes=[concat_node, add_node],
            name="TestNet",
            inputs=[X_1, X_2, X_3],
            outputs=[Y_1, Y_2],
            initializer=[],
        )

        model_def = onnx.helper.make_model(graph_def, producer_name="onnx-example")
        model_def.opset_import[0].version = 13

        model_def = onnx.shape_inference.infer_shapes(model_def)
        super().__init__(model_def, 'multi_input_output_model.dot')
