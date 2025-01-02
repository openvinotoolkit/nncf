# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import numpy as np
import onnx

from nncf.common.utils.registry import Registry
from tests.onnx.common import get_random_generator

OPSET_VERSION = 13
ALL_SYNTHETIC_MODELS = Registry("ONNX_SYNTHETIC_MODELS")


def create_initializer_tensor(
    name: str, tensor_array: np.ndarray, data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:
    initializer_tensor = onnx.helper.make_tensor(
        name=name, data_type=data_type, dims=tensor_array.shape, vals=tensor_array.flatten().tolist()
    )
    return initializer_tensor


class ONNXReferenceModel:
    def __init__(self, onnx_model, input_shape: List[List[int]], graph_path):
        self.onnx_model = onnx_model
        self.onnx_model.ir_version = 9
        self.input_shape = input_shape
        self.path_ref_graph = graph_path


@ALL_SYNTHETIC_MODELS.register()
class LinearModel(ONNXReferenceModel):
    INPUT_NAME = "X"

    def __init__(self):
        input_shape = [1, 3, 32, 32]
        model_input_name = self.INPUT_NAME
        X = onnx.helper.make_tensor_value_info(model_input_name, onnx.TensorProto.FLOAT, input_shape)
        model_output_name = "Y"
        model_output_channels = 10
        Y = onnx.helper.make_tensor_value_info(
            model_output_name, onnx.TensorProto.FLOAT, [1, model_output_channels, 1, 1]
        )
        rng = get_random_generator()
        conv1_output_node_name = "Conv1_Y"
        conv1_in_channels, conv1_out_channels, conv1_kernel_shape = 3, 32, (3, 3)
        conv1_W = rng.uniform(0, 1, (conv1_out_channels, conv1_in_channels, *conv1_kernel_shape)).astype(np.float32)
        conv1_B = rng.uniform(0, 1, conv1_out_channels).astype(np.float32)

        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = create_initializer_tensor(
            name=conv1_W_initializer_tensor_name, tensor_array=conv1_W, data_type=onnx.TensorProto.FLOAT
        )
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = create_initializer_tensor(
            name=conv1_B_initializer_tensor_name, tensor_array=conv1_B, data_type=onnx.TensorProto.FLOAT
        )

        conv1_node = onnx.helper.make_node(
            name="Conv1",
            op_type="Conv",
            inputs=[model_input_name, conv1_W_initializer_tensor_name, conv1_B_initializer_tensor_name],
            outputs=[conv1_output_node_name],
            kernel_shape=conv1_kernel_shape,
        )

        bn1_output_node_name = "BN1_Y"
        bn1_scale = rng.uniform(0, 1, conv1_out_channels).astype(np.float32)
        bn1_bias = rng.uniform(0, 1, conv1_out_channels).astype(np.float32)
        bn1_mean = rng.uniform(0, 1, conv1_out_channels).astype(np.float32)
        bn1_var = rng.uniform(0, 1, conv1_out_channels).astype(np.float32)
        bn1_scale_initializer_tensor_name = "BN1_Scale"
        bn1_bias_initializer_tensor_name = "BN1_Bias"
        bn1_mean_initializer_tensor_name = "BN1_Mean"
        bn1_var_initializer_tensor_name = "BN1_Var"
        bn1_scale_initializer_tensor = create_initializer_tensor(
            name=bn1_scale_initializer_tensor_name, tensor_array=bn1_scale, data_type=onnx.TensorProto.FLOAT
        )
        bn1_bias_initializer_tensor = create_initializer_tensor(
            name=bn1_bias_initializer_tensor_name, tensor_array=bn1_bias, data_type=onnx.TensorProto.FLOAT
        )
        bn1_mean_initializer_tensor = create_initializer_tensor(
            name=bn1_mean_initializer_tensor_name, tensor_array=bn1_mean, data_type=onnx.TensorProto.FLOAT
        )
        bn1_var_initializer_tensor = create_initializer_tensor(
            name=bn1_var_initializer_tensor_name, tensor_array=bn1_var, data_type=onnx.TensorProto.FLOAT
        )

        bn1_node = onnx.helper.make_node(
            name="BN1",
            op_type="BatchNormalization",
            inputs=[
                conv1_output_node_name,
                bn1_scale_initializer_tensor_name,
                bn1_bias_initializer_tensor_name,
                bn1_mean_initializer_tensor_name,
                bn1_var_initializer_tensor_name,
            ],
            outputs=[bn1_output_node_name],
        )

        relu1_output_node_name = "ReLU1_Y"
        relu1_node = onnx.helper.make_node(
            name="ReLU1",
            op_type="Relu",
            inputs=[bn1_output_node_name],
            outputs=[relu1_output_node_name],
        )

        avg_pool1_output_node_name = "Avg_Pool1_Y"
        avg_pool1_node = onnx.helper.make_node(
            name="Avg_Pool1",
            op_type="GlobalAveragePool",
            inputs=[relu1_output_node_name],
            outputs=[avg_pool1_output_node_name],
        )

        conv2_in_channels, conv2_out_channels, conv2_kernel_shape = conv1_out_channels, model_output_channels, (1, 1)
        conv2_W = rng.uniform(0, 1, (conv2_out_channels, conv2_in_channels, *conv2_kernel_shape)).astype(np.float32)
        conv2_B = rng.uniform(0, 1, conv2_out_channels).astype(np.float32)

        conv2_W_initializer_tensor_name = "Conv2_W"
        conv2_W_initializer_tensor = create_initializer_tensor(
            name=conv2_W_initializer_tensor_name, tensor_array=conv2_W, data_type=onnx.TensorProto.FLOAT
        )
        conv2_B_initializer_tensor_name = "Conv2_B"
        conv2_B_initializer_tensor = create_initializer_tensor(
            name=conv2_B_initializer_tensor_name, tensor_array=conv2_B, data_type=onnx.TensorProto.FLOAT
        )

        conv2_node = onnx.helper.make_node(
            name="Conv2",
            op_type="Conv",
            inputs=[avg_pool1_output_node_name, conv2_W_initializer_tensor_name, conv2_B_initializer_tensor_name],
            outputs=[model_output_name],
            kernel_shape=conv2_kernel_shape,
        )

        graph_def = onnx.helper.make_graph(
            nodes=[conv1_node, bn1_node, relu1_node, avg_pool1_node, conv2_node],
            name="ConvNet",
            inputs=[X],
            outputs=[Y],
            initializer=[
                conv1_W_initializer_tensor,
                conv1_B_initializer_tensor,
                bn1_scale_initializer_tensor,
                bn1_bias_initializer_tensor,
                bn1_mean_initializer_tensor,
                bn1_var_initializer_tensor,
                conv2_W_initializer_tensor,
                conv2_B_initializer_tensor,
            ],
        )
        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], "linear_model.dot")


@ALL_SYNTHETIC_MODELS.register()
class MultiInputOutputModel(ONNXReferenceModel):
    def __init__(self):
        input_shape_1 = [1, 6, 3, 3]
        model_input_name_1 = "X_1"
        X_1 = onnx.helper.make_tensor_value_info(model_input_name_1, onnx.TensorProto.FLOAT, input_shape_1)
        input_shape_2 = [2, 6, 3, 3]
        model_input_name_2 = "X_2"
        X_2 = onnx.helper.make_tensor_value_info(model_input_name_2, onnx.TensorProto.FLOAT, input_shape_2)
        input_shape_3 = [3, 6, 3, 3]
        model_input_name_3 = "X_3"
        X_3 = onnx.helper.make_tensor_value_info(model_input_name_3, onnx.TensorProto.FLOAT, input_shape_3)

        model_output_name_1 = "Y_1"
        Y_1 = onnx.helper.make_tensor_value_info(model_output_name_1, onnx.TensorProto.FLOAT, [6, 6, 3, 3])

        model_output_name_2 = "Y_2"
        Y_2 = onnx.helper.make_tensor_value_info(model_output_name_2, onnx.TensorProto.FLOAT, [2, 6, 3, 3])

        concat_node = onnx.helper.make_node(
            name="Concat1",
            op_type="Concat",
            inputs=[model_input_name_1, model_input_name_2, model_input_name_3],
            outputs=[model_output_name_1],
            axis=0,
        )

        add_node = onnx.helper.make_node(
            name="Add1", op_type="Add", inputs=[model_input_name_1, model_input_name_2], outputs=[model_output_name_2]
        )

        # Create the graph (GraphProto)
        graph_def = onnx.helper.make_graph(
            nodes=[concat_node, add_node],
            name="MultiInputOutputNet",
            inputs=[X_1, X_2, X_3],
            outputs=[Y_1, Y_2],
            initializer=[],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape_1, input_shape_2, input_shape_3], "multi_input_output_model.dot")


@ALL_SYNTHETIC_MODELS.register()
class DoubleInputOutputModel(ONNXReferenceModel):
    def __init__(self):
        input_shape_1 = [1, 6, 3, 3]
        model_input_name_1 = "X_1"
        X_1 = onnx.helper.make_tensor_value_info(model_input_name_1, onnx.TensorProto.FLOAT, input_shape_1)
        input_shape_2 = [2, 6, 3, 3]
        model_input_name_2 = "X_2"
        X_2 = onnx.helper.make_tensor_value_info(model_input_name_2, onnx.TensorProto.FLOAT, input_shape_2)

        model_output_name_1 = "Y_1"
        Y_1 = onnx.helper.make_tensor_value_info(model_output_name_1, onnx.TensorProto.FLOAT, [2, 6, 3, 3])

        model_output_name_2 = "Y_2"
        Y_2 = onnx.helper.make_tensor_value_info(model_output_name_2, onnx.TensorProto.FLOAT, [2, 6, 3, 3])

        concat_node = onnx.helper.make_node(
            name="Add2",
            op_type="Add",
            inputs=[model_input_name_1, model_input_name_2],
            outputs=[model_output_name_1],
        )

        add_node = onnx.helper.make_node(
            name="Add1", op_type="Add", inputs=[model_input_name_2, model_input_name_1], outputs=[model_output_name_2]
        )

        # Create the graph (GraphProto)
        graph_def = onnx.helper.make_graph(
            nodes=[concat_node, add_node],
            name="DoubleInputOutputNet",
            inputs=[X_1, X_2],
            outputs=[Y_1, Y_2],
            initializer=[],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape_1, input_shape_2], "double_input_output_model.dot")


@ALL_SYNTHETIC_MODELS.register()
class ModelWithIntEdges(ONNXReferenceModel):
    def __init__(self):
        model_input_name_1 = "X_1"
        input_shape = [1, 6, 3, 3]
        X_1 = onnx.helper.make_tensor_value_info(model_input_name_1, onnx.TensorProto.FLOAT, input_shape)

        model_output_name_1 = "Y_1"
        Y_1 = onnx.helper.make_tensor_value_info(model_output_name_1, onnx.TensorProto.FLOAT, [1, 6, 3, 3])

        shape_node_output_name = "shape_output"
        # Output is int64
        shape_node = onnx.helper.make_node(
            name="Shape1", op_type="Shape", inputs=[model_input_name_1], outputs=[shape_node_output_name]
        )

        constant_node = onnx.helper.make_node(
            name="Constant1", op_type="ConstantOfShape", inputs=[shape_node_output_name], outputs=[model_output_name_1]
        )

        graph_def = onnx.helper.make_graph(
            nodes=[shape_node, constant_node],
            name="MultiInputOutputNet",
            inputs=[X_1],
            outputs=[Y_1],
            initializer=[],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], "int_edges_model.dot")


@ALL_SYNTHETIC_MODELS.register()
class OneConvolutionalModel(ONNXReferenceModel):
    def __init__(self):
        input_shape = [1, 3, 10, 10]
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name, onnx.TensorProto.FLOAT, input_shape)
        rng = get_random_generator()
        conv1_in_channels, conv1_out_channels, conv1_kernel_shape = 3, 32, (1, 1)
        conv1_W = rng.uniform(0, 1, (conv1_out_channels, conv1_in_channels, *conv1_kernel_shape)).astype(np.float32)
        conv1_B = rng.uniform(0, 1, conv1_out_channels).astype(np.float32)

        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name, onnx.TensorProto.FLOAT, [1, conv1_out_channels, 10, 10]
        )

        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = create_initializer_tensor(
            name=conv1_W_initializer_tensor_name, tensor_array=conv1_W, data_type=onnx.TensorProto.FLOAT
        )
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = create_initializer_tensor(
            name=conv1_B_initializer_tensor_name, tensor_array=conv1_B, data_type=onnx.TensorProto.FLOAT
        )

        conv1_node = onnx.helper.make_node(
            name="Conv1",
            op_type="Conv",
            inputs=[model_input_name, conv1_W_initializer_tensor_name, conv1_B_initializer_tensor_name],
            outputs=[model_output_name],
            kernel_shape=conv1_kernel_shape,
        )

        graph_def = onnx.helper.make_graph(
            nodes=[conv1_node],
            name="ConvNet",
            inputs=[X],
            outputs=[Y],
            initializer=[conv1_W_initializer_tensor, conv1_B_initializer_tensor],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        self.conv_bias = conv1_B
        super().__init__(model, [input_shape], "one_convolutional_model.dot")


@ALL_SYNTHETIC_MODELS.register()
class OneConvolutionalIdentityBiasModel(ONNXReferenceModel):
    def __init__(self):
        input_shape = [1, 3, 10, 10]
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name, onnx.TensorProto.FLOAT, input_shape)
        rng = get_random_generator()
        conv1_in_channels, conv1_out_channels, conv1_kernel_shape = 3, 32, (1, 1)
        conv1_W = rng.uniform(0, 1, (conv1_out_channels, conv1_in_channels, *conv1_kernel_shape)).astype(np.float32)
        conv1_B = rng.uniform(0, 1, conv1_out_channels).astype(np.float32)

        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name, onnx.TensorProto.FLOAT, [1, conv1_out_channels, 10, 10]
        )

        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = create_initializer_tensor(
            name=conv1_W_initializer_tensor_name, tensor_array=conv1_W, data_type=onnx.TensorProto.FLOAT
        )
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = create_initializer_tensor(
            name=conv1_B_initializer_tensor_name, tensor_array=conv1_B, data_type=onnx.TensorProto.FLOAT
        )

        identity_output = "Identity_OUT"
        identity_node = onnx.helper.make_node(
            name="Identity",
            op_type="Identity",
            inputs=[conv1_B_initializer_tensor_name],
            outputs=[identity_output],
        )

        conv1_node = onnx.helper.make_node(
            name="Conv1",
            op_type="Conv",
            inputs=[model_input_name, conv1_W_initializer_tensor_name, identity_output],
            outputs=[model_output_name],
            kernel_shape=conv1_kernel_shape,
        )

        graph_def = onnx.helper.make_graph(
            nodes=[identity_node, conv1_node],
            name="ConvIdentityBiasNet",
            inputs=[X],
            outputs=[Y],
            initializer=[conv1_W_initializer_tensor, conv1_B_initializer_tensor],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        self.conv_bias = conv1_B
        super().__init__(model, [input_shape], "one_convolutional_identity_bias_model.dot")


@ALL_SYNTHETIC_MODELS.register()
class ReshapeWeightModel(ONNXReferenceModel):
    # This graph pattern is in inception-v1-12:
    # https://github.com/onnx/models/tree/main/vision/classification/inception_and_googlenet/inception_v1
    #
    #       X + Z      reshaped_W
    #          \     /
    #           GEMM
    #             |
    #          softmax
    def __init__(self):
        model_input_name = "X"
        model_input_channels = 10
        input_shape = [1, model_input_channels]
        X = onnx.helper.make_tensor_value_info(model_input_name, onnx.TensorProto.FLOAT, input_shape)
        model_output_name = "Y"
        model_output_channels = 5
        Y = onnx.helper.make_tensor_value_info(model_output_name, onnx.TensorProto.FLOAT, [1, model_output_channels])
        rng = get_random_generator()
        shape = [1, 1, model_input_channels, model_output_channels]
        w_tensor = create_initializer_tensor(
            name="W", tensor_array=rng.uniform(0, 1, shape).astype(np.float32), data_type=onnx.TensorProto.FLOAT
        )

        w_shape_tensor = create_initializer_tensor(
            name="w_shape",
            tensor_array=np.array([model_input_channels, model_output_channels]),
            data_type=onnx.TensorProto.INT64,
        )

        z_tensor = create_initializer_tensor(
            name="z_tensor",
            tensor_array=rng.uniform(0, 1, [1, model_input_channels]).astype(np.float32),
            data_type=onnx.TensorProto.FLOAT,
        )

        reshaped_w_node = onnx.helper.make_node(
            name="Reshape",
            op_type="Reshape",
            inputs=["W", "w_shape"],
            outputs=["reshaped_w"],
        )

        added_x_node = onnx.helper.make_node(
            name="Add",
            op_type="Add",
            inputs=["X", "z_tensor"],
            outputs=["added_x"],
        )

        gemm_node = onnx.helper.make_node(
            name="Gemm", op_type="Gemm", inputs=["added_x", "reshaped_w"], outputs=["logit"]
        )

        softmax_node = onnx.helper.make_node(
            name="Softmax",
            op_type="Softmax",
            inputs=["logit"],
            outputs=["Y"],
        )

        graph_def = onnx.helper.make_graph(
            nodes=[reshaped_w_node, added_x_node, gemm_node, softmax_node],
            name="Net",
            inputs=[X],
            outputs=[Y],
            initializer=[w_tensor, w_shape_tensor, z_tensor],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], "reshape_weight_model.dot")


@ALL_SYNTHETIC_MODELS.register()
class WeightSharingModel(ONNXReferenceModel):
    # This graph pattern is in retinanet-9:
    # https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/retinanet
    #
    #             X
    #             |
    #           ReLU
    #          /    \
    # W -> Conv1    Conv2 <- W
    #          \    /
    #           Add
    #            |
    #            Y
    def __init__(self):
        input_shape = [1, 1, 5, 5]
        output_shape = [1, 5, 5, 5]
        W_shape = [5, 1, 3, 3]
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name, onnx.TensorProto.FLOAT, input_shape)
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name, onnx.TensorProto.FLOAT, output_shape)

        rng = get_random_generator()

        w_tensor = create_initializer_tensor(
            name="W", tensor_array=rng.uniform(0, 1, W_shape), data_type=onnx.TensorProto.FLOAT
        )

        relu_x_node = onnx.helper.make_node(
            name="Relu",
            op_type="Relu",
            inputs=["X"],
            outputs=["relu_X"],
        )

        conv1_node = onnx.helper.make_node(
            name="Conv1",
            op_type="Conv",
            inputs=["relu_X", "W"],
            outputs=["conv_1"],
            kernel_shape=[3, 3],
            # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
            pads=[1, 1, 1, 1],
        )

        conv2_node = onnx.helper.make_node(
            name="Conv2",
            op_type="Conv",
            inputs=["relu_X", "W"],
            outputs=["conv_2"],
            kernel_shape=[3, 3],
            # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
            pads=[1, 1, 1, 1],
        )

        add_node = onnx.helper.make_node(
            name="Add",
            op_type="Add",
            inputs=["conv_1", "conv_2"],
            outputs=["Y"],
        )

        graph_def = onnx.helper.make_graph(
            nodes=[relu_x_node, conv1_node, conv2_node, add_node],
            name="Net",
            inputs=[X],
            outputs=[Y],
            initializer=[w_tensor],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], "weight_sharing_model.dot")


@ALL_SYNTHETIC_MODELS.register()
class OneInputPortQuantizableModel(ONNXReferenceModel):
    #             X
    #             |
    #            ReLU
    #             |   \
    #             |   Softmax
    #             |  /
    #             Mul
    #             |
    #             Y
    def __init__(self):
        input_shape = output_shape = [1, 1, 5, 5]

        # IO tensors (ValueInfoProto).
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name, onnx.TensorProto.FLOAT, input_shape)
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name, onnx.TensorProto.FLOAT, output_shape)

        relu_x_node = onnx.helper.make_node(
            name="Relu",
            op_type="Relu",
            inputs=["X"],
            outputs=["relu_X"],
        )

        softmax_node = onnx.helper.make_node(
            name="Softmax", op_type="Softmax", inputs=["relu_X"], outputs=["softmax_1"]
        )

        mul_node = onnx.helper.make_node(
            name="Mul",
            op_type="Mul",
            inputs=["relu_X", "softmax_1"],
            outputs=["Y"],
        )

        graph_def = onnx.helper.make_graph(
            nodes=[relu_x_node, softmax_node, mul_node],
            name="Net",
            inputs=[X],
            outputs=[Y],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], "one_input_port_quantizable_model.dot")


@ALL_SYNTHETIC_MODELS.register()
class ManyInputPortsQuantizableModel(ONNXReferenceModel):
    #             X
    #             |
    #            ReLU
    #           /  |
    #    Identity  |
    #       |    \  |
    #    Softmax  Mul
    #             |
    #             Y
    def __init__(self):
        input_shape = output_shape = [1, 1, 5, 5]

        # IO tensors (ValueInfoProto).
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name, onnx.TensorProto.FLOAT, input_shape)
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name, onnx.TensorProto.FLOAT, output_shape)

        model_output_name1 = "Y1"
        Y1 = onnx.helper.make_tensor_value_info(model_output_name1, onnx.TensorProto.FLOAT, output_shape)

        relu_x_node = onnx.helper.make_node(
            name="Relu",
            op_type="Relu",
            inputs=["X"],
            outputs=["relu_X"],
        )

        identity_node = onnx.helper.make_node(name="Identity", op_type="Identity", inputs=["X"], outputs=["identity_1"])

        softmax_node = onnx.helper.make_node(name="Softmax", op_type="Softmax", inputs=["identity_1"], outputs=["Y1"])

        mul_node = onnx.helper.make_node(
            name="Mul",
            op_type="Mul",
            inputs=["relu_X", "identity_1"],
            outputs=["Y"],
        )

        graph_def = onnx.helper.make_graph(
            nodes=[relu_x_node, identity_node, softmax_node, mul_node],
            name="Net",
            inputs=[X],
            outputs=[Y, Y1],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], "many_input_ports_quantizable_model.dot")


class OneDepthwiseConvolutionalModel(ONNXReferenceModel):
    def __init__(self):
        input_shape = [1, 3, 10, 10]
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name, onnx.TensorProto.FLOAT, input_shape)
        conv_group = 3
        conv1_in_channels, conv1_out_channels, conv1_kernel_shape = 3 // conv_group, 27, (1, 1)
        rng = get_random_generator()
        conv1_W = rng.uniform(0, 1, (conv1_out_channels, conv1_in_channels, *conv1_kernel_shape)).astype(np.float32)
        conv1_B = rng.uniform(0, 1, conv1_out_channels).astype(np.float32)

        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(
            model_output_name, onnx.TensorProto.FLOAT, [1, conv1_out_channels, 10, 10]
        )

        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = create_initializer_tensor(
            name=conv1_W_initializer_tensor_name, tensor_array=conv1_W, data_type=onnx.TensorProto.FLOAT
        )
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = create_initializer_tensor(
            name=conv1_B_initializer_tensor_name, tensor_array=conv1_B, data_type=onnx.TensorProto.FLOAT
        )

        conv1_node = onnx.helper.make_node(
            name="Conv1",
            op_type="Conv",
            inputs=[model_input_name, conv1_W_initializer_tensor_name, conv1_B_initializer_tensor_name],
            outputs=[model_output_name],
            group=conv_group,
            kernel_shape=conv1_kernel_shape,
        )

        graph_def = onnx.helper.make_graph(
            nodes=[conv1_node],
            name="ConvNet",
            inputs=[X],
            outputs=[Y],
            initializer=[conv1_W_initializer_tensor, conv1_B_initializer_tensor],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], "one_depthwise_convolutional_model.dot")


class InputOutputModel(ONNXReferenceModel):
    def __init__(self):
        input_shape = [1, 3, 3, 3]
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name, onnx.TensorProto.FLOAT, input_shape)

        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name, onnx.TensorProto.FLOAT, input_shape)
        identity_node = onnx.helper.make_node(name="Identity", op_type="Identity", inputs=["X"], outputs=["Y"])
        graph_def = onnx.helper.make_graph(
            nodes=[identity_node],
            name="ConvNet",
            inputs=[X],
            outputs=[Y],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], "input_output_model.dot")


class IdentityConvolutionalModel(ONNXReferenceModel):
    def __init__(self, input_shape=None, inp_ch=3, out_ch=32, kernel_size=1, conv_w=None, conv_b=None):
        if input_shape is None:
            input_shape = [1, 3, 10, 10]

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name, onnx.TensorProto.FLOAT, input_shape)

        conv1_in_channels, conv1_out_channels, conv1_kernel_shape = inp_ch, out_ch, (kernel_size,) * 2
        rng = get_random_generator()
        conv1_W = conv_w
        if conv1_W is None:
            conv1_W = rng.uniform(0, 1, (conv1_out_channels, conv1_in_channels, *conv1_kernel_shape))
        conv1_W = conv1_W.astype(np.float32)

        conv1_B = conv_b
        if conv1_B is None:
            conv1_B = rng.uniform(0, 1, conv1_out_channels)
        conv1_B = conv1_B.astype(np.float32)

        model_identity_op_name = "Identity"
        model_conv_op_name = "Conv1"
        model_output_name = "Y"

        identity_node = onnx.helper.make_node(
            name=model_identity_op_name,
            op_type="Identity",
            inputs=[model_input_name],
            outputs=[model_input_name + "_X"],
        )

        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = create_initializer_tensor(
            name=conv1_W_initializer_tensor_name, tensor_array=conv1_W, data_type=onnx.TensorProto.FLOAT
        )
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = create_initializer_tensor(
            name=conv1_B_initializer_tensor_name, tensor_array=conv1_B, data_type=onnx.TensorProto.FLOAT
        )

        conv1_node = onnx.helper.make_node(
            name=model_conv_op_name,
            op_type="Conv",
            inputs=[model_input_name + "_X", conv1_W_initializer_tensor_name, conv1_B_initializer_tensor_name],
            outputs=[model_output_name],
            kernel_shape=conv1_kernel_shape,
        )

        Y = onnx.helper.make_tensor_value_info(
            model_output_name,
            onnx.TensorProto.FLOAT,
            [1, conv1_out_channels, input_shape[-2] - kernel_size + 1, input_shape[-1] - kernel_size + 1],
        )

        graph_def = onnx.helper.make_graph(
            nodes=[identity_node, conv1_node],
            name="ConvNet",
            inputs=[X],
            outputs=[Y],
            initializer=[conv1_W_initializer_tensor, conv1_B_initializer_tensor],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], "one_convolutional_model.dot")


@ALL_SYNTHETIC_MODELS.register()
class ShapeOfModel(ONNXReferenceModel):
    INPUT_NAME = "X"

    def __init__(self):
        input_shape = [1, 3, 32, 32]
        model_input_name = self.INPUT_NAME
        X = onnx.helper.make_tensor_value_info(model_input_name, onnx.TensorProto.FLOAT, input_shape)
        model_output_name = "Y"
        model_output_channels = 10
        Y = onnx.helper.make_tensor_value_info(
            model_output_name, onnx.TensorProto.FLOAT, [1, model_output_channels, 1, 1]
        )

        conv1_output_node_name = "Conv1_Y"
        conv1_in_channels, conv1_out_channels, conv1_kernel_shape = 3, 32, (3, 3)
        rng = get_random_generator()
        conv1_W = rng.uniform(0, 1, (conv1_out_channels, conv1_in_channels, *conv1_kernel_shape)).astype(np.float32)
        conv1_B = rng.uniform(0, 1, conv1_out_channels).astype(np.float32)

        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = create_initializer_tensor(
            name=conv1_W_initializer_tensor_name, tensor_array=conv1_W, data_type=onnx.TensorProto.FLOAT
        )
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = create_initializer_tensor(
            name=conv1_B_initializer_tensor_name, tensor_array=conv1_B, data_type=onnx.TensorProto.FLOAT
        )

        conv1_node = onnx.helper.make_node(
            name="Conv1",
            op_type="Conv",
            inputs=[model_input_name, conv1_W_initializer_tensor_name, conv1_B_initializer_tensor_name],
            outputs=[conv1_output_node_name],
            kernel_shape=conv1_kernel_shape,
        )

        relu1_output_node_name = "ReLU1_Y"
        relu1_node = onnx.helper.make_node(
            name="ReLU1",
            op_type="Relu",
            inputs=[conv1_output_node_name],
            outputs=[relu1_output_node_name],
        )

        # Shape subgraph
        shape_output_node_name = "Shape_Y"
        shape_node = onnx.helper.make_node(
            name="Shape",
            op_type="Shape",
            inputs=[relu1_output_node_name],
            outputs=[shape_output_node_name],
        )
        gather_output_node_name = "Gather_Y"
        gather_indices_tensor_name = "Gather_I"
        gather_indices_initializer_tensor = create_initializer_tensor(
            name=gather_indices_tensor_name, tensor_array=np.int64(2), data_type=onnx.TensorProto.INT64
        )
        gather_node = onnx.helper.make_node(
            name="Gather",
            op_type="Gather",
            axis=0,
            inputs=[shape_output_node_name, gather_indices_tensor_name],
            outputs=[gather_output_node_name],
        )
        cast_1_output_node_name = "Cast1_Y"
        cast_1_node = onnx.helper.make_node(
            name="Cast1",
            op_type="Cast",
            to=onnx.TensorProto.INT64,
            inputs=[gather_output_node_name],
            outputs=[cast_1_output_node_name],
        )
        cast_2_output_node_name = "Cast2_Y"
        cast_2_node = onnx.helper.make_node(
            name="Cast2",
            op_type="Cast",
            to=onnx.TensorProto.FLOAT,
            inputs=[cast_1_output_node_name],
            outputs=[cast_2_output_node_name],
        )
        sqrt_output_node_name = "Sqrt_Y"
        sqrt_node = onnx.helper.make_node(
            name="Sqrt",
            op_type="Sqrt",
            inputs=[cast_2_output_node_name],
            outputs=[sqrt_output_node_name],
        )
        reshape_output_node_name = "Reshape_Y"
        reshape_node = onnx.helper.make_node(
            name="Reshape",
            op_type="Reshape",
            inputs=[relu1_output_node_name, sqrt_output_node_name],
            outputs=[reshape_output_node_name],
        )

        conv2_in_channels, conv2_out_channels, conv2_kernel_shape = conv1_out_channels, model_output_channels, (1, 1)
        conv2_W = rng.uniform(0, 1, (conv2_out_channels, conv2_in_channels, *conv2_kernel_shape)).astype(np.float32)
        conv2_B = rng.uniform(0, 1, conv2_out_channels).astype(np.float32)

        conv2_W_initializer_tensor_name = "Conv2_W"
        conv2_W_initializer_tensor = create_initializer_tensor(
            name=conv2_W_initializer_tensor_name, tensor_array=conv2_W, data_type=onnx.TensorProto.FLOAT
        )
        conv2_B_initializer_tensor_name = "Conv2_B"
        conv2_B_initializer_tensor = create_initializer_tensor(
            name=conv2_B_initializer_tensor_name, tensor_array=conv2_B, data_type=onnx.TensorProto.FLOAT
        )

        conv2_node = onnx.helper.make_node(
            name="Conv2",
            op_type="Conv",
            inputs=[reshape_output_node_name, conv2_W_initializer_tensor_name, conv2_B_initializer_tensor_name],
            outputs=[model_output_name],
            kernel_shape=conv2_kernel_shape,
        )

        graph_def = onnx.helper.make_graph(
            nodes=[
                conv1_node,
                relu1_node,
                shape_node,
                gather_node,
                cast_1_node,
                cast_2_node,
                sqrt_node,
                reshape_node,
                conv2_node,
            ],
            name="ConvNet",
            inputs=[X],
            outputs=[Y],
            initializer=[
                conv1_W_initializer_tensor,
                conv1_B_initializer_tensor,
                gather_indices_initializer_tensor,
                conv2_W_initializer_tensor,
                conv2_B_initializer_tensor,
            ],
        )
        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], "shape_of_model.dot")


@ALL_SYNTHETIC_MODELS.register()
class Float64InputMulModel(ONNXReferenceModel):
    def __init__(self):
        input_shape = [1, 3, 10, 10]
        model_input_name = "X"
        model_reciprocal_op_name = "Reciprocal"
        model_cast_op_name = "Cast"
        model_cast_output = "Cast_Y"

        X = onnx.helper.make_tensor_value_info(model_input_name, onnx.TensorProto.DOUBLE, input_shape)

        reciprocal_node = onnx.helper.make_node(
            name=model_reciprocal_op_name,
            op_type="Reciprocal",
            inputs=[model_input_name],
            outputs=[model_input_name + "_X"],
        )

        cast_node = onnx.helper.make_node(
            name=model_cast_op_name,
            op_type="Cast",
            inputs=[model_input_name + "_X"],
            outputs=[model_cast_output],
            to=onnx.TensorProto.FLOAT,
        )

        tensor = np.array((1)).astype(np.float32)
        tensor_name = "Tensor"
        initializer_tensor = create_initializer_tensor(
            name=tensor_name, tensor_array=tensor, data_type=onnx.TensorProto.FLOAT
        )

        conv_output_node_name = "Conv1_Y"
        conv_in_channels, conv_out_channels, conv1_kernel_shape = 3, 32, (3, 3)
        rng = get_random_generator()
        conv_W = rng.uniform(0, 1, (conv_out_channels, conv_in_channels, *conv1_kernel_shape)).astype(np.float32)
        conv_B = rng.uniform(0, 1, conv_out_channels).astype(np.float32)

        conv_W_initializer_tensor_name = "Conv1_W"
        conv_W_initializer_tensor = create_initializer_tensor(
            name=conv_W_initializer_tensor_name, tensor_array=conv_W, data_type=onnx.TensorProto.FLOAT
        )
        conv_B_initializer_tensor_name = "Conv1_B"
        conv_B_initializer_tensor = create_initializer_tensor(
            name=conv_B_initializer_tensor_name, tensor_array=conv_B, data_type=onnx.TensorProto.FLOAT
        )

        conv_node = onnx.helper.make_node(
            name="Conv1",
            op_type="Conv",
            inputs=[model_cast_output, conv_W_initializer_tensor_name, conv_B_initializer_tensor_name],
            outputs=[conv_output_node_name],
            kernel_shape=conv1_kernel_shape,
        )

        Y = onnx.helper.make_tensor_value_info(conv_output_node_name, onnx.TensorProto.FLOAT, input_shape)

        graph_def = onnx.helper.make_graph(
            nodes=[reciprocal_node, cast_node, conv_node],
            name="Float64Net",
            inputs=[X],
            outputs=[Y],
            initializer=[initializer_tensor, conv_W_initializer_tensor, conv_B_initializer_tensor],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], "float64_model.dot")


@ALL_SYNTHETIC_MODELS.register()
class NonShapeModel(ONNXReferenceModel):
    def __init__(self):
        input_shape = [1, 3, 32, 32]
        model_input_name = "X"
        model_output_name = "Y"
        X = onnx.helper.make_tensor_value_info(model_input_name, onnx.TensorProto.FLOAT, input_shape)

        model_output_channels = 10
        Y = onnx.helper.make_tensor_value_info(model_output_name, onnx.TensorProto.FLOAT, [1, model_output_channels])

        conv1_output_node_name = "conv"
        conv1_in_channels, conv1_out_channels, conv1_kernel_shape = 3, 32, (3, 3)
        rng = get_random_generator()
        conv1_W = rng.uniform(0, 1, (conv1_out_channels, conv1_in_channels, *conv1_kernel_shape)).astype(np.float32)
        conv1_B = rng.uniform(0, 1, conv1_out_channels).astype(np.float32)

        conv1_W_initializer_tensor_name = "Conv_W"
        conv1_W_initializer_tensor = create_initializer_tensor(
            name=conv1_W_initializer_tensor_name, tensor_array=conv1_W, data_type=onnx.TensorProto.FLOAT
        )
        conv1_B_initializer_tensor_name = "Conv_B"
        conv1_B_initializer_tensor = create_initializer_tensor(
            name=conv1_B_initializer_tensor_name, tensor_array=conv1_B, data_type=onnx.TensorProto.FLOAT
        )

        conv1_node = onnx.helper.make_node(
            name="Conv",
            op_type="Conv",
            inputs=[model_input_name, conv1_W_initializer_tensor_name, conv1_B_initializer_tensor_name],
            outputs=[conv1_output_node_name],
            kernel_shape=conv1_kernel_shape,
        )

        relu1_output_node_name = "relu_1"
        relu1_node = onnx.helper.make_node(
            name="Relu1",
            op_type="Relu",
            inputs=[conv1_output_node_name],
            outputs=[relu1_output_node_name],
        )

        # Shape subgraph
        shape1_output_node_name = "shape_1"
        shape1_node = onnx.helper.make_node(
            name="Shape1",
            op_type="Shape",
            inputs=[relu1_output_node_name],
            outputs=[shape1_output_node_name],
        )
        shape2_output_node_name = "shape_2"
        shape2_node = onnx.helper.make_node(
            name="Shape2",
            op_type="Shape",
            inputs=[relu1_output_node_name],
            outputs=[shape2_output_node_name],
        )
        gather1_output_node_name = "gather_1"
        gather1_indices_tensor_name = "gather_1_w"
        gather1_indices_initializer_tensor = create_initializer_tensor(
            name=gather1_indices_tensor_name, tensor_array=np.int64(0), data_type=onnx.TensorProto.INT64
        )
        gather1_node = onnx.helper.make_node(
            name="Gather1",
            op_type="Gather",
            inputs=[shape1_output_node_name, gather1_indices_tensor_name],
            outputs=[gather1_output_node_name],
        )
        gather2_output_node_name = "gather_2"
        gather2_indices_tensor_name = "gather_2_w"
        gather2_indices_initializer_tensor = create_initializer_tensor(
            name=gather2_indices_tensor_name, tensor_array=np.int64(1), data_type=onnx.TensorProto.INT64
        )
        gather2_node = onnx.helper.make_node(
            name="Gather2",
            op_type="Gather",
            inputs=[shape2_output_node_name, gather2_indices_tensor_name],
            outputs=[gather2_output_node_name],
        )
        unsqueeze1_output_node_name = "unsqueeze_1"
        unsqueeze1_axes_tensor_name = "unsqueeze_1_a"
        unsqueeze1_axes_initializer_tensor = create_initializer_tensor(
            name=unsqueeze1_axes_tensor_name, tensor_array=np.int64([0]), data_type=onnx.TensorProto.INT64
        )
        unsqueeze1_node = onnx.helper.make_node(
            name="Unsqueeze1",
            op_type="Unsqueeze",
            inputs=[gather1_output_node_name, unsqueeze1_axes_tensor_name],
            outputs=[unsqueeze1_output_node_name],
        )
        unsqueeze2_output_node_name = "unsqueeze_2"
        unsqueeze2_axes_tensor_name = "unsqueeze_2_a"
        unsqueeze2_axes_initializer_tensor = create_initializer_tensor(
            name=unsqueeze2_axes_tensor_name, tensor_array=np.int64([0]), data_type=onnx.TensorProto.INT64
        )
        unsqueeze2_node = onnx.helper.make_node(
            name="Unsqueeze2",
            op_type="Unsqueeze",
            inputs=[gather2_output_node_name, unsqueeze2_axes_tensor_name],
            outputs=[unsqueeze2_output_node_name],
        )
        concat_output_node_name = "concat"
        concat_node = onnx.helper.make_node(
            name="Concat",
            op_type="Concat",
            inputs=[unsqueeze1_output_node_name, unsqueeze2_output_node_name],
            outputs=[concat_output_node_name],
            axis=0,
        )

        avg_pool_output_node_name = "global_average_pool"
        avg_pool_node = onnx.helper.make_node(
            name="GlobalAveragePool",
            op_type="GlobalAveragePool",
            inputs=[relu1_output_node_name],
            outputs=[avg_pool_output_node_name],
        )

        reshape_output_node_name = "reshape"
        reshape_node = onnx.helper.make_node(
            name="Reshape",
            op_type="Reshape",
            inputs=[avg_pool_output_node_name, concat_output_node_name],
            outputs=[reshape_output_node_name],
        )

        rng = np.random.default_rng(seed=0)
        shape = [conv1_out_channels, model_output_channels]
        gemm_w_tensor = create_initializer_tensor(
            name="W", tensor_array=rng.uniform(0, 1, shape).astype(np.float32), data_type=onnx.TensorProto.FLOAT
        )
        gemm_output_node_name = "gemm"
        gemm_node = onnx.helper.make_node(
            name="Gemm", op_type="Gemm", inputs=[reshape_output_node_name, "W"], outputs=[gemm_output_node_name]
        )

        relu2_node = onnx.helper.make_node(
            name="Relu2",
            op_type="Relu",
            inputs=[gemm_output_node_name],
            outputs=[model_output_name],
        )

        graph_def = onnx.helper.make_graph(
            nodes=[
                conv1_node,
                relu1_node,
                shape1_node,
                shape2_node,
                gather1_node,
                gather2_node,
                unsqueeze1_node,
                unsqueeze2_node,
                concat_node,
                avg_pool_node,
                reshape_node,
                gemm_node,
                relu2_node,
            ],
            name="NonShapeModel",
            inputs=[X],
            outputs=[Y],
            initializer=[
                conv1_W_initializer_tensor,
                conv1_B_initializer_tensor,
                gather1_indices_initializer_tensor,
                gather2_indices_initializer_tensor,
                unsqueeze1_axes_initializer_tensor,
                unsqueeze2_axes_initializer_tensor,
                gemm_w_tensor,
            ],
        )
        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], "non_shape_model.dot")


@ALL_SYNTHETIC_MODELS.register()
class MatMulWeightModel(ONNXReferenceModel):
    #         X       W
    #          \     /
    #           MatMul
    #             |
    #          softmax
    def __init__(self):
        model_input_name, model_output_name = "X", "Y"
        model_input_channels, model_output_channels = 10, 5
        input_shape = [1, model_input_channels]

        X = onnx.helper.make_tensor_value_info(model_input_name, onnx.TensorProto.FLOAT, input_shape)
        Y = onnx.helper.make_tensor_value_info(model_output_name, onnx.TensorProto.FLOAT, [1, model_output_channels])

        rng = np.random.default_rng(seed=0)
        shape = [model_input_channels, model_output_channels]
        w_tensor = create_initializer_tensor(
            name="W", tensor_array=rng.uniform(0, 1, shape).astype(np.float32), data_type=onnx.TensorProto.FLOAT
        )

        matmul_node = onnx.helper.make_node(
            name="MatMul", op_type="MatMul", inputs=[model_input_name, "W"], outputs=["logit"]
        )

        softmax_node = onnx.helper.make_node(
            name="Softmax",
            op_type="Softmax",
            inputs=["logit"],
            outputs=["Y"],
        )

        graph_def = onnx.helper.make_graph(
            nodes=[matmul_node, softmax_node],
            name="Net",
            inputs=[X],
            outputs=[Y],
            initializer=[w_tensor],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], "weight_matmul_model.dot")


@ALL_SYNTHETIC_MODELS.register()
class MatMulActivationModel(ONNXReferenceModel):
    #         X       Y
    #         |       |
    #          \     /
    #           MatMul
    #             |
    #          softmax
    def __init__(self):
        model_input_name_1, model_input_name_2, model_output_name = "X", "Y", "Z"
        channels = 10
        x_input_shape = [channels, 1]
        y_input_shape = [1, channels]

        X = onnx.helper.make_tensor_value_info(model_input_name_1, onnx.TensorProto.FLOAT, x_input_shape)
        Y = onnx.helper.make_tensor_value_info(model_input_name_2, onnx.TensorProto.FLOAT, y_input_shape)
        Z = onnx.helper.make_tensor_value_info(model_output_name, onnx.TensorProto.FLOAT, [channels, channels])

        matmul_node = onnx.helper.make_node(name="MatMul", op_type="MatMul", inputs=["X", "Y"], outputs=["logit"])

        softmax_node = onnx.helper.make_node(
            name="Softmax",
            op_type="Softmax",
            inputs=["logit"],
            outputs=["Z"],
        )

        graph_def = onnx.helper.make_graph(nodes=[matmul_node, softmax_node], name="Net", inputs=[X, Y], outputs=[Z])

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [x_input_shape, y_input_shape], "activation_matmul_model.dot")


@ALL_SYNTHETIC_MODELS.register()
class GEMMTransposeWeightModel(ONNXReferenceModel):
    #         X       W(Transposed)
    #         |       |
    #      Identity   |
    #          \     /
    #           Gemm
    #             |
    #          softmax
    def __init__(self):
        model_input_name, model_output_name = "X", "Y"
        model_input_channels, model_output_channels = 10, 5
        input_shape = [1, model_input_channels]

        X = onnx.helper.make_tensor_value_info(model_input_name, onnx.TensorProto.FLOAT, input_shape)
        Y = onnx.helper.make_tensor_value_info(model_output_name, onnx.TensorProto.FLOAT, [1, model_output_channels])

        rng = np.random.default_rng(seed=0)
        shape = [model_output_channels, model_input_channels]
        w_tensor = create_initializer_tensor(
            name="W", tensor_array=rng.uniform(0, 1, shape).astype(np.float32), data_type=onnx.TensorProto.FLOAT
        )

        identity_node = onnx.helper.make_node(
            name="Identity", op_type="Identity", inputs=[model_input_name], outputs=["identity"]
        )

        gemm_node = onnx.helper.make_node(
            name="Gemm", op_type="Gemm", inputs=["identity", "W"], outputs=["logit"], transB=1
        )

        softmax_node = onnx.helper.make_node(
            name="Softmax",
            op_type="Softmax",
            inputs=["logit"],
            outputs=["Y"],
        )

        graph_def = onnx.helper.make_graph(
            nodes=[identity_node, gemm_node, softmax_node],
            name="Net",
            inputs=[X],
            outputs=[Y],
            initializer=[w_tensor],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], "gemm_weight_transpose_model.dot")


@ALL_SYNTHETIC_MODELS.register()
class WeightPropagationMatMulModel(ONNXReferenceModel):
    #               Identity
    #                   |
    #         X     Identity
    #          \     /
    #           MatMul
    #             |     Constant
    #             |     /
    #           MatMul
    #             |
    #             Y
    def __init__(self):
        model_input_name, model_output_name = "X", "Y"
        model_input_channels = 10
        matmul_output_channels = 5
        input_shape = [1, model_input_channels]
        X = onnx.helper.make_tensor_value_info(model_input_name, onnx.TensorProto.FLOAT, input_shape)
        Y = onnx.helper.make_tensor_value_info(model_output_name, onnx.TensorProto.FLOAT, [1, model_input_channels])

        rng = np.random.default_rng(seed=0)
        shape = [model_input_channels, matmul_output_channels]

        # Create MatMul
        w_tensor = create_initializer_tensor(
            name="W_tensor", tensor_array=rng.uniform(0, 1, shape).astype(np.float32), data_type=onnx.TensorProto.FLOAT
        )
        identity_1 = onnx.helper.make_node(name="Identity_1", op_type="Identity", inputs=["W_tensor"], outputs=["i_1"])
        identity_2 = onnx.helper.make_node(name="Identity_2", op_type="Identity", inputs=["i_1"], outputs=["i_2"])
        matmul_1 = onnx.helper.make_node(
            name="MatMul_1", op_type="MatMul", inputs=[model_input_name, "i_2"], outputs=["mm_1"]
        )
        matmul_2_shape = (matmul_output_channels, model_input_channels)
        constant_data = rng.uniform(0, 1, matmul_2_shape).astype(np.float32)  # Randomly initialized weight tensor
        constant_initializer = onnx.helper.make_tensor(
            name="constant_data",
            data_type=onnx.TensorProto.FLOAT,
            dims=constant_data.shape,
            vals=constant_data.flatten(),
        )
        constant = onnx.helper.make_node("Constant", [], ["const"], name="constant", value=constant_initializer)
        matmul_2 = onnx.helper.make_node(
            name="MatMul_2", op_type="MatMul", inputs=["mm_1", "const"], outputs=[model_output_name]
        )

        graph_def = onnx.helper.make_graph(
            nodes=[identity_1, identity_2, matmul_1, constant, matmul_2],
            name="Net",
            inputs=[X],
            outputs=[Y],
            initializer=[w_tensor, constant_initializer],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], "weight_propagation_matmul_model.dot")


@ALL_SYNTHETIC_MODELS.register()
class WeightPropagationConvModel(ONNXReferenceModel):
    #     X             Reshape
    #     |               |
    #     |           Transpose
    #     |               |
    #     \            Identiy
    #       \            /
    #         \       /
    #            Conv
    #             |
    #             |         Constant
    #             |           /
    #             |       Reshape
    #             |        /
    #             |    Identity
    #             |     /
    #            Conv
    #             |
    #             |    Constant
    #             |   /
    #            Conv
    #             |
    #             |
    #             |
    def __init__(self):
        input_shape = (1, 1, 28, 28)  # Example shape, change as required
        Y = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 1, 28, 28])
        input_tensor = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, input_shape)
        rng = get_random_generator()

        # Layer 1: Convolution 1
        conv_output1 = "conv_output1"
        conv1_shape = (1, 1, 3, 3)
        conv1_weight = rng.uniform(0, 1, conv1_shape).astype(np.float32)  # Randomly initialized weight tensor

        conv1_weight_initializer = onnx.helper.make_tensor(
            name="conv1_weight",
            data_type=onnx.TensorProto.FLOAT,
            dims=conv1_weight.shape,
            vals=conv1_weight.flatten(),
        )

        # Layer 1: Identity -> Transpose -> Reshape
        identity_output1 = "identity_output1"
        transpose_output = "transpose_output"
        reshape_output = "reshape_output"
        reshape_1_tensor_name = "w_r_1"
        reshape_1_initializer_tensor = create_initializer_tensor(
            name=reshape_1_tensor_name,
            tensor_array=np.array(conv1_shape).astype(np.int64),
            data_type=onnx.TensorProto.INT64,
        )
        reshape_node = onnx.helper.make_node(
            "Reshape", ["conv1_weight", reshape_1_tensor_name], [reshape_output], name="reshape"
        )
        transpose_node = onnx.helper.make_node(
            "Transpose", [reshape_output], [transpose_output], name="transpose", perm=[0, 1, 3, 2]
        )
        identity_node1 = onnx.helper.make_node("Identity", [transpose_output], [identity_output1], name="identity1")

        conv1_node = onnx.helper.make_node(
            "Conv", ["input", identity_output1], [conv_output1], name="conv1", kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        )

        # Layer 3: Convolution 2
        identity_output2 = "identity_output2"
        reshape_output2 = "reshape_output2"
        constant_output = "constant_output"
        conv_output2 = "conv_output2"
        conv2_shape = (1, 1, 3, 3)
        conv2_node = onnx.helper.make_node(
            "Conv",
            [conv_output1, identity_output2],
            [conv_output2],
            name="conv2",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )

        # Layer 4: Identity -> Reshape -> Constant
        constant_data = rng.uniform(0, 1, conv2_shape).astype(np.float32)  # Randomly initialized weight tensor
        reshape_2_tensor_name = "w_r_2"

        reshape_2_initializer_tensor = create_initializer_tensor(
            name=reshape_2_tensor_name,
            tensor_array=np.array((1, 1, 3, 3)).astype(np.int64),
            data_type=onnx.TensorProto.INT64,
        )
        constant_initializer = onnx.helper.make_tensor(
            name="constant_data",
            data_type=onnx.TensorProto.FLOAT,
            dims=constant_data.shape,
            vals=constant_data.flatten(),
        )
        constant_node = onnx.helper.make_node(
            "Constant", [], [constant_output], name="constant", value=constant_initializer
        )
        reshape_node2 = onnx.helper.make_node(
            "Reshape", [constant_output, reshape_2_tensor_name], [reshape_output2], name="reshape2"
        )
        identity_node2 = onnx.helper.make_node("Identity", [reshape_output2], [identity_output2], name="identity2")

        # Layer 6: Convolution 3
        constant_output2 = "constant_output2"
        conv4_shape = (1, 1, 3, 3)
        constant_data2 = rng.uniform(0, 1, conv4_shape).astype(np.float32)  # Randomly initialized weight tensor
        constant_initializer2 = onnx.helper.make_tensor(
            name="constant_data2",
            data_type=onnx.TensorProto.FLOAT,
            dims=constant_data2.shape,
            vals=constant_data2.flatten(),
        )
        constant_2_node = onnx.helper.make_node(
            "Constant", [], [constant_output2], name="constant2", value=constant_initializer2
        )
        conv4_node = onnx.helper.make_node(
            "Conv",
            [conv_output2, constant_output2],
            ["output"],
            name="conv4",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )

        # Create the graph with all the nodes
        graph_def = onnx.helper.make_graph(
            [
                reshape_node,
                transpose_node,
                identity_node1,
                conv1_node,
                constant_node,
                reshape_node2,
                identity_node2,
                conv2_node,
                constant_2_node,
                conv4_node,
            ],
            "example_model",
            [input_tensor],
            [Y],
            [
                conv1_weight_initializer,
                constant_initializer,
                reshape_1_initializer_tensor,
                reshape_2_initializer_tensor,
                constant_initializer2,
            ],
        )

        # Create the model with the graph
        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], "weight_propagation_conv_model.dot")


@ALL_SYNTHETIC_MODELS.register()
class EmbeddingModel(ONNXReferenceModel):
    #               Constant
    #                   |
    #         X     Identity
    #          \     /
    #           Gather
    #             |
    #           Gather
    #             |
    #           MatMul
    #             |
    #             Y
    def __init__(self):
        model_input_name, model_output_name = "X", "Y"
        model_input_channels = 10
        model_output_channels = 10
        input_shape = [1, model_input_channels]
        X = onnx.helper.make_tensor_value_info(model_input_name, onnx.TensorProto.INT64, input_shape)
        Y = onnx.helper.make_tensor_value_info(model_output_name, onnx.TensorProto.FLOAT, [10, model_output_channels])

        rng = np.random.default_rng(seed=0)

        embedding_output_node_name = "Embedding_Y"
        embedding_weights_tensor_name = "Embedding_W"
        embedding_weights_tensor = create_initializer_tensor(
            name=embedding_weights_tensor_name,
            tensor_array=rng.uniform(0, 1, (10, 20)).astype(np.float32),
            data_type=onnx.TensorProto.FLOAT,
        )

        identity_output_name = "Identity_Y"
        identity_node = onnx.helper.make_node(
            name="Identity",
            op_type="Identity",
            inputs=[embedding_weights_tensor_name],
            outputs=[identity_output_name],
        )

        embedding_node = onnx.helper.make_node(
            name="Embedding",
            op_type="Gather",
            axis=0,
            inputs=[identity_output_name, model_input_name],
            outputs=[embedding_output_node_name],
        )

        gather_output_node_name = "Gather_Y"
        gather_indices_tensor_name = "Gather_I"
        gather_indices_initializer_tensor = create_initializer_tensor(
            name=gather_indices_tensor_name, tensor_array=np.int64(0), data_type=onnx.TensorProto.INT64
        )
        gather_node = onnx.helper.make_node(
            name="Gather",
            op_type="Gather",
            axis=0,
            inputs=[embedding_output_node_name, gather_indices_tensor_name],
            outputs=[gather_output_node_name],
        )

        shape = [20, model_output_channels]
        w_tensor_name = "W"
        w_tensor = create_initializer_tensor(
            name=w_tensor_name,
            tensor_array=rng.uniform(0, 1, shape).astype(np.float32),
            data_type=onnx.TensorProto.FLOAT,
        )

        matmul_node = onnx.helper.make_node(
            name="MatMul",
            op_type="MatMul",
            inputs=[gather_output_node_name, w_tensor_name],
            outputs=[model_output_name],
        )

        graph_def = onnx.helper.make_graph(
            nodes=[identity_node, embedding_node, gather_node, matmul_node],
            name="EmbeddingModel",
            inputs=[X],
            outputs=[Y],
            initializer=[embedding_weights_tensor, gather_indices_initializer_tensor, w_tensor],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], "embedding_model.dot")


@ALL_SYNTHETIC_MODELS.register()
class UnifiedEmbeddingModel(ONNXReferenceModel):
    #       X
    #      / \
    #     | Convert
    #     |     \
    #   MatMul  Gather
    #     |       |
    #  Reshape    |
    #     \      /
    #      Concat
    #         |
    #       MatMul
    #         |
    #         Y
    def __init__(self):
        model_input_name, model_output_name = "X", "Y"
        model_input_channels = 3
        model_output_channels = 6
        input_shape = [1, model_input_channels]
        X = onnx.helper.make_tensor_value_info(model_input_name, onnx.TensorProto.FLOAT, input_shape)
        Y = onnx.helper.make_tensor_value_info(model_output_name, onnx.TensorProto.FLOAT, [1, model_output_channels])

        rng = np.random.default_rng(seed=0)

        cast_output_name = "Cast_Y"
        cast_node = onnx.helper.make_node(
            name="Cast",
            op_type="Cast",
            to=onnx.TensorProto.INT64,
            inputs=[model_input_name],
            outputs=[cast_output_name],
        )

        embedding_output_name = "Embedding_Y"
        embedding_tensor_name = "Embedding_W"
        embedding_tensor = create_initializer_tensor(
            name=embedding_tensor_name,
            tensor_array=rng.uniform(0, 1, (4, 5)).astype(np.float32),
            data_type=onnx.TensorProto.FLOAT,
        )
        embedding_node = onnx.helper.make_node(
            name="Embedding",
            op_type="Gather",
            axis=0,
            inputs=[embedding_tensor_name, cast_output_name],
            outputs=[embedding_output_name],
        )

        matmul_1_tensor_name = "W_1"
        matmul_1_output_name = "MatMul_1_Y"
        matmul_1_tensor = create_initializer_tensor(
            name=matmul_1_tensor_name,
            tensor_array=rng.uniform(0, 1, (3, 3, 5)).astype(np.float32),
            data_type=onnx.TensorProto.FLOAT,
        )
        matmul_1_node = onnx.helper.make_node(
            name="MatMul_1",
            op_type="MatMul",
            inputs=[model_input_name, matmul_1_tensor_name],
            outputs=[matmul_1_output_name],
        )

        reshape_tensor_name = "R"
        reshape_tensor = create_initializer_tensor(
            name=reshape_tensor_name,
            tensor_array=np.array([1, 3, 5]).astype(np.float32),
            data_type=onnx.TensorProto.FLOAT,
        )
        reshape_output_name = "Reshape_Y"
        reshape_node = onnx.helper.make_node(
            name="Reshape",
            op_type="Reshape",
            inputs=[matmul_1_output_name, reshape_tensor_name],
            outputs=[reshape_output_name],
        )

        concat_output_name = "Concat_Y"
        concat_node = onnx.helper.make_node(
            name="Concat",
            op_type="Concat",
            inputs=[embedding_output_name, reshape_output_name],
            outputs=[concat_output_name],
            axis=0,
        )

        matmul_2_tensor_name = "W_2"
        matmul_2_tensor = create_initializer_tensor(
            name=matmul_2_tensor_name,
            tensor_array=rng.uniform(0, 1, (1, 5)).astype(np.float32),
            data_type=onnx.TensorProto.FLOAT,
        )
        matmul_2_node = onnx.helper.make_node(
            name="MatMul_2",
            op_type="MatMul",
            inputs=[concat_output_name, matmul_2_tensor_name],
            outputs=[model_output_name],
        )

        graph_def = onnx.helper.make_graph(
            nodes=[cast_node, embedding_node, matmul_1_node, reshape_node, concat_node, matmul_2_node],
            name="UnifiedEmbeddingModel",
            inputs=[X],
            outputs=[Y],
            initializer=[embedding_tensor, matmul_1_tensor, matmul_2_tensor, reshape_tensor],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], "unified_embedding_model.dot")


class RoPEModel(ONNXReferenceModel):
    def __init__(self):
        rng = np.random.default_rng(seed=0)

        input_shape = [1, 10]
        input_name = "model_in"
        model_in = onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.INT64, input_shape)

        output_shape = [1, 5, 10]
        cos_out_name = "cos_out"
        cos_out = onnx.helper.make_tensor_value_info(cos_out_name, onnx.TensorProto.FLOAT, output_shape)
        sin_out_name = "sin_out"
        sin_out = onnx.helper.make_tensor_value_info(sin_out_name, onnx.TensorProto.FLOAT, output_shape)

        unsqueeze_out_name = "un_out"
        unsqueeze_tensor_name = "un_tensor"
        unsqueeze_tensor = create_initializer_tensor(
            name=unsqueeze_tensor_name, tensor_array=np.int64([2]), data_type=onnx.TensorProto.INT64
        )
        unsqueeze_node = onnx.helper.make_node(
            name="unsqueeze",
            op_type="Unsqueeze",
            inputs=[input_name, unsqueeze_tensor_name],
            outputs=[unsqueeze_out_name],
        )

        cast_out_name = "cast_out"
        cast_node = onnx.helper.make_node(
            name="cast",
            op_type="Cast",
            to=onnx.TensorProto.FLOAT,
            inputs=[unsqueeze_out_name],
            outputs=[cast_out_name],
        )

        reshape_shape_name = "re_shape"
        reshape_shape = create_initializer_tensor(
            name=reshape_shape_name,
            tensor_array=np.array([1, 5]).astype(np.int64),
            data_type=onnx.TensorProto.INT64,
        )
        reshape_tensor_name = "re_tensor"
        reshape_tensor = create_initializer_tensor(
            name=reshape_tensor_name,
            tensor_array=rng.uniform(0, 1, (5)).astype(np.float32),
            data_type=onnx.TensorProto.FLOAT,
        )
        reshape_out_name = "re_out"
        reshape_node = onnx.helper.make_node(
            name="reshape",
            op_type="Reshape",
            inputs=[reshape_tensor_name, reshape_shape_name],
            outputs=[reshape_out_name],
        )

        matmul_out_name = "mm_out"
        matmul_node = onnx.helper.make_node(
            name="matmul",
            op_type="MatMul",
            inputs=[cast_out_name, reshape_out_name],
            outputs=[matmul_out_name],
        )

        transpose_out_name = "trans_out"
        transpose_node = onnx.helper.make_node(
            name="transpose",
            op_type="Transpose",
            inputs=[matmul_out_name],
            outputs=[transpose_out_name],
            perm=[0, 2, 1],
        )

        concat_out_name = "concat_out"
        concat_node = onnx.helper.make_node(
            name="concat",
            op_type="Concat",
            inputs=[transpose_out_name],
            outputs=[concat_out_name],
            axis=-1,
        )

        sin_node = onnx.helper.make_node(
            name="sin",
            op_type="Sin",
            inputs=[concat_out_name],
            outputs=[sin_out_name],
        )

        cos_node = onnx.helper.make_node(
            name="cos",
            op_type="Cos",
            inputs=[concat_out_name],
            outputs=[cos_out_name],
        )

        graph_def = onnx.helper.make_graph(
            nodes=[
                unsqueeze_node,
                cast_node,
                reshape_node,
                matmul_node,
                transpose_node,
                concat_node,
                sin_node,
                cos_node,
            ],
            name="RoPEModel",
            inputs=[model_in],
            outputs=[sin_out, cos_out],
            initializer=[
                unsqueeze_tensor,
                reshape_tensor,
                reshape_shape,
            ],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], "rope_model.dot")
