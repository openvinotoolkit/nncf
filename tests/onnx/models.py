"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from typing import List

import numpy as np
import onnx

from nncf.common.utils.registry import Registry


# pylint: disable=no-member

def create_initializer_tensor(name: str, tensor_array: np.ndarray,
                              data_type: onnx.TensorProto = onnx.TensorProto.FLOAT) -> onnx.TensorProto:
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())
    return initializer_tensor


OPSET_VERSION = 13
ALL_SYNTHETIC_MODELS = Registry('ONNX_SYNTHETIC_MODELS')


class ONNXReferenceModel:
    def __init__(self, onnx_model, input_shape: List[List[int]], graph_path):
        self.onnx_model = onnx_model
        self.input_shape = input_shape
        self.path_ref_graph = graph_path


@ALL_SYNTHETIC_MODELS.register()
class LinearModel(ONNXReferenceModel):
    INPUT_NAME = "X"

    def __init__(self):
        input_shape = [1, 3, 32, 32]
        model_input_name = self.INPUT_NAME
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               input_shape)
        model_output_name = "Y"
        model_output_channels = 10
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [1, model_output_channels, 1, 1])

        conv1_output_node_name = "Conv1_Y"
        conv1_in_channels, conv1_out_channels, conv1_kernel_shape = 3, 32, (3, 3)
        conv1_W = np.ones(shape=(conv1_out_channels, conv1_in_channels, *conv1_kernel_shape)).astype(np.float32)
        conv1_B = np.ones(shape=conv1_out_channels).astype(np.float32)

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
            name="Conv1",
            op_type="Conv",
            inputs=[
                model_input_name, conv1_W_initializer_tensor_name,
                conv1_B_initializer_tensor_name
            ],
            outputs=[conv1_output_node_name],
            kernel_shape=conv1_kernel_shape,
        )

        bn1_output_node_name = "BN1_Y"
        bn1_scale = np.random.randn(conv1_out_channels).astype(np.float32)
        bn1_bias = np.random.randn(conv1_out_channels).astype(np.float32)
        bn1_mean = np.random.randn(conv1_out_channels).astype(np.float32)
        bn1_var = np.random.rand(conv1_out_channels).astype(np.float32)
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
            name="BN1",
            op_type="BatchNormalization",
            inputs=[
                conv1_output_node_name, bn1_scale_initializer_tensor_name,
                bn1_bias_initializer_tensor_name, bn1_mean_initializer_tensor_name,
                bn1_var_initializer_tensor_name
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
        conv2_W = np.ones(shape=(conv2_out_channels, conv2_in_channels,
                                 *conv2_kernel_shape)).astype(np.float32)
        conv2_B = np.ones(shape=conv2_out_channels).astype(np.float32)

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
            kernel_shape=conv2_kernel_shape
        )

        graph_def = onnx.helper.make_graph(
            nodes=[conv1_node, bn1_node, relu1_node, avg_pool1_node, conv2_node],
            name="ConvNet",
            inputs=[X],
            outputs=[Y],
            initializer=[
                conv1_W_initializer_tensor, conv1_B_initializer_tensor,
                bn1_scale_initializer_tensor, bn1_bias_initializer_tensor,
                bn1_mean_initializer_tensor, bn1_var_initializer_tensor,
                conv2_W_initializer_tensor, conv2_B_initializer_tensor
            ],
        )
        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], 'linear_model.dot')


@ALL_SYNTHETIC_MODELS.register()
class MultiInputOutputModel(ONNXReferenceModel):
    def __init__(self):
        input_shape_1 = [1, 6, 3, 3]
        model_input_name_1 = "X_1"
        X_1 = onnx.helper.make_tensor_value_info(model_input_name_1,
                                                 onnx.TensorProto.FLOAT,
                                                 input_shape_1)
        input_shape_2 = [2, 6, 3, 3]
        model_input_name_2 = "X_2"
        X_2 = onnx.helper.make_tensor_value_info(model_input_name_2,
                                                 onnx.TensorProto.FLOAT,
                                                 input_shape_2)
        input_shape_3 = [3, 6, 3, 3]
        model_input_name_3 = "X_3"
        X_3 = onnx.helper.make_tensor_value_info(model_input_name_3,
                                                 onnx.TensorProto.FLOAT,
                                                 input_shape_3)

        model_output_name_1 = "Y_1"
        Y_1 = onnx.helper.make_tensor_value_info(model_output_name_1,
                                                 onnx.TensorProto.FLOAT,
                                                 [6, 6, 3, 3])

        model_output_name_2 = "Y_2"
        Y_2 = onnx.helper.make_tensor_value_info(model_output_name_2,
                                                 onnx.TensorProto.FLOAT,
                                                 [2, 6, 3, 3])

        concat_node = onnx.helper.make_node(
            name="Concat1",
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
            name="MultiInputOutputNet",
            inputs=[X_1, X_2, X_3],
            outputs=[Y_1, Y_2],
            initializer=[],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape_1, input_shape_2, input_shape_3], 'multi_input_output_model.dot')


@ALL_SYNTHETIC_MODELS.register()
class ModelWithIntEdges(ONNXReferenceModel):
    def __init__(self):
        model_input_name_1 = "X_1"
        input_shape = [1, 6, 3, 3]
        X_1 = onnx.helper.make_tensor_value_info(model_input_name_1,
                                                 onnx.TensorProto.FLOAT,
                                                 input_shape)

        model_output_name_1 = "Y_1"
        Y_1 = onnx.helper.make_tensor_value_info(model_output_name_1,
                                                 onnx.TensorProto.FLOAT,
                                                 [1, 6, 3, 3])

        shape_node_output_name = 'shape_output'
        # Output is int64
        shape_node = onnx.helper.make_node(
            name="Shape1",
            op_type="Shape",
            inputs=[
                model_input_name_1
            ],
            outputs=[shape_node_output_name]
        )

        constant_node = onnx.helper.make_node(
            name="Constant1",
            op_type="ConstantOfShape",
            inputs=[
                shape_node_output_name
            ],
            outputs=[model_output_name_1]
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
        super().__init__(model, [input_shape], 'int_edges_model.dot')


@ALL_SYNTHETIC_MODELS.register()
class OneConvolutionalModel(ONNXReferenceModel):
    def __init__(self):
        input_shape = [1, 3, 10, 10]
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               input_shape)

        conv1_in_channels, conv1_out_channels, conv1_kernel_shape = 3, 32, (1, 1)
        conv1_W = np.ones(shape=(conv1_out_channels, conv1_in_channels, *conv1_kernel_shape)).astype(np.float32)
        conv1_B = np.ones(shape=conv1_out_channels).astype(np.float32)

        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [1, conv1_out_channels, 10, 10])

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
            name="Conv1",
            op_type="Conv",
            inputs=[
                model_input_name, conv1_W_initializer_tensor_name,
                conv1_B_initializer_tensor_name
            ],
            outputs=[model_output_name],
            kernel_shape=conv1_kernel_shape,
        )

        graph_def = onnx.helper.make_graph(
            nodes=[conv1_node],
            name="ConvNet",
            inputs=[X],
            outputs=[Y],
            initializer=[
                conv1_W_initializer_tensor, conv1_B_initializer_tensor
            ],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], 'one_convolutional_model.dot')


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
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               input_shape)
        model_output_name = "Y"
        model_output_channels = 5
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [1, model_output_channels])

        w_tensor = create_initializer_tensor(
            name="W",
            tensor_array=np.random.standard_normal(
                [1, 1, model_input_channels, model_output_channels]),
            data_type=onnx.TensorProto.FLOAT)

        w_shape_tensor = create_initializer_tensor(
            name="w_shape",
            tensor_array=np.array([model_input_channels, model_output_channels]),
            data_type=onnx.TensorProto.INT64)

        z_tensor = create_initializer_tensor(
            name="z_tensor",
            tensor_array=np.random.standard_normal([1, model_input_channels]),
            data_type=onnx.TensorProto.FLOAT)

        reshaped_w_node = onnx.helper.make_node(
            name='Reshape',
            op_type="Reshape",
            inputs=["W", "w_shape"],
            outputs=["reshaped_w"],
        )

        added_x_node = onnx.helper.make_node(
            name='Add',
            op_type="Add",
            inputs=["X", "z_tensor"],
            outputs=["added_x"],
        )

        gemm_node = onnx.helper.make_node(
            name='Gemm',
            op_type='Gemm',
            inputs=['added_x', 'reshaped_w'],
            outputs=['logit']
        )

        softmax_node = onnx.helper.make_node(
            name='Softmax',
            op_type='Softmax',
            inputs=['logit'],
            outputs=['Y'],
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
        super().__init__(model, [input_shape], 'reshape_weight_model.dot')


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
        input_shape = output_shape = [1, 1, 5, 5]

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               input_shape)
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               output_shape)

        W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                        [1., 1., 1.],
                        [1., 1., 1.]]]]).astype(np.float32)

        w_tensor = create_initializer_tensor(
            name="W",
            tensor_array=W,
            data_type=onnx.TensorProto.FLOAT)

        relu_x_node = onnx.helper.make_node(
            name="Relu",
            op_type='Relu',
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
            name='Add',
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
        super().__init__(model, [input_shape], 'weight_sharing_model.dot')


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
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               input_shape)
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               output_shape)

        relu_x_node = onnx.helper.make_node(
            name="Relu",
            op_type='Relu',
            inputs=["X"],
            outputs=["relu_X"],
        )

        softmax_node = onnx.helper.make_node(
            name="Softmax",
            op_type="Softmax",
            inputs=["relu_X"],
            outputs=["softmax_1"]
        )

        mul_node = onnx.helper.make_node(
            name='Mul',
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
        super().__init__(model, [input_shape], 'one_input_port_quantizable_model.dot')


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
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               input_shape)
        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               output_shape)

        model_output_name1 = "Y1"
        Y1 = onnx.helper.make_tensor_value_info(model_output_name1,
                                                onnx.TensorProto.FLOAT,
                                                output_shape)

        relu_x_node = onnx.helper.make_node(
            name="Relu",
            op_type='Relu',
            inputs=["X"],
            outputs=["relu_X"],
        )

        identity_node = onnx.helper.make_node(
            name="Identity",
            op_type="Identity",
            inputs=["X"],
            outputs=["identity_1"]
        )

        softmax_node = onnx.helper.make_node(
            name="Softmax",
            op_type="Softmax",
            inputs=["identity_1"],
            outputs=["Y1"]
        )

        mul_node = onnx.helper.make_node(
            name='Mul',
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
        super().__init__(model, [input_shape], 'many_input_ports_quantizable_model.dot')


class OneDepthwiseConvolutionalModel(ONNXReferenceModel):
    def __init__(self):
        input_shape = [1, 3, 10, 10]
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               input_shape)
        conv_group = 3
        conv1_in_channels, conv1_out_channels, conv1_kernel_shape = 3 // conv_group, 27, (1, 1)

        conv1_W = np.ones(shape=(conv1_out_channels, conv1_in_channels, *conv1_kernel_shape)).astype(np.float32)
        conv1_B = np.ones(shape=conv1_out_channels).astype(np.float32)

        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [1, conv1_out_channels, 10, 10])

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
            name="Conv1",
            op_type="Conv",
            inputs=[
                model_input_name, conv1_W_initializer_tensor_name,
                conv1_B_initializer_tensor_name
            ],
            outputs=[model_output_name],
            group=conv_group,
            kernel_shape=conv1_kernel_shape,
        )

        graph_def = onnx.helper.make_graph(
            nodes=[conv1_node],
            name="ConvNet",
            inputs=[X],
            outputs=[Y],
            initializer=[
                conv1_W_initializer_tensor, conv1_B_initializer_tensor
            ],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], 'one_depthwise_convolutional_model.dot')


class InputOutputModel(ONNXReferenceModel):
    def __init__(self):
        input_shape = [1, 3, 3, 3]
        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               input_shape)

        model_output_name = "Y"
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               input_shape)
        identity_node = onnx.helper.make_node(
            name="Identity",
            op_type="Identity",
            inputs=["X"],
            outputs=["Y"]
        )
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
        super().__init__(model, [input_shape], 'input_output_model.dot')


class IdentityConvolutionalModel(ONNXReferenceModel):
    def __init__(self,
                 input_shape=None,
                 inp_ch=3,
                 out_ch=32,
                 kernel_size=1,
                 conv_w=None,
                 conv_b=None):
        if input_shape is None:
            input_shape = [1, 3, 10, 10]

        model_input_name = "X"
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               input_shape)

        conv1_in_channels, conv1_out_channels, conv1_kernel_shape = inp_ch, out_ch, (kernel_size,) * 2

        conv1_W = conv_w
        if conv1_W is None:
            conv1_W = np.ones(shape=(conv1_out_channels, conv1_in_channels, *conv1_kernel_shape))
        conv1_W = conv1_W.astype(np.float32)

        conv1_B = conv_b
        if conv1_B is None:
            conv1_B = np.ones(shape=conv1_out_channels)
        conv1_B = conv1_B.astype(np.float32)

        model_identity_op_name = 'Identity'
        model_conv_op_name = 'Conv1'
        model_output_name = "Y"

        identity_node = onnx.helper.make_node(
            name=model_identity_op_name,
            op_type="Identity",
            inputs=[model_input_name],
            outputs=[model_input_name + '_X']
        )

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
            name=model_conv_op_name,
            op_type="Conv",
            inputs=[
                model_input_name + '_X', conv1_W_initializer_tensor_name,
                conv1_B_initializer_tensor_name
            ],
            outputs=[model_output_name],
            kernel_shape=conv1_kernel_shape,
        )

        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [1, conv1_out_channels,
                                                input_shape[-2] - kernel_size + 1,
                                                input_shape[-1] - kernel_size + 1])

        graph_def = onnx.helper.make_graph(
            nodes=[identity_node, conv1_node],
            name="ConvNet",
            inputs=[X],
            outputs=[Y],
            initializer=[
                conv1_W_initializer_tensor, conv1_B_initializer_tensor
            ],
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        super().__init__(model, [input_shape], 'one_convolutional_model.dot')
