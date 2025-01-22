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

from abc import ABC
from abc import abstractmethod
from functools import partial
from typing import Callable, Optional, Tuple

import numpy as np
import openvino.runtime as ov
from openvino.runtime import opset13 as opset

from nncf.common.utils.registry import Registry
from tests.torch.test_models.inceptionv3 import inception_v3
from tests.torch.test_models.mobilenet import mobilenet_v2
from tests.torch.test_models.mobilenet_v3 import mobilenet_v3_small
from tests.torch.test_models.resnet import ResNet18
from tests.torch.test_models.ssd_mobilenet import ssd_mobilenet
from tests.torch.test_models.ssd_vgg import ssd_vgg300
from tests.torch.test_models.swin import SwinTransformerBlock

SYNTHETIC_MODELS = Registry("OV_SYNTHETIC_MODELS")


def get_torch_model_info(model_name: str) -> Tuple[Callable, Tuple[int]]:
    models = {
        "mobilenet-v2": (mobilenet_v2, (1, 3, 224, 224)),
        "mobilenet-v3-small": (mobilenet_v3_small, (1, 3, 224, 224)),
        "resnet-18": (ResNet18, (1, 3, 224, 224)),
        "inception-v3": (inception_v3, (1, 3, 224, 224)),
        "ssd-vgg-300": (ssd_vgg300, (1, 3, 300, 300)),
        "ssd-mobilenet": (ssd_mobilenet, (1, 3, 300, 300)),
        "swin-block": (partial(SwinTransformerBlock, dim=8, input_resolution=[4, 4], num_heads=2), (1, 16, 8)),
    }
    return models[model_name]


class OVReferenceModel(ABC):
    def __init__(self, **kwargs):
        self._rng = np.random.default_rng(seed=0)
        self.ref_model_name = f"{self.__class__.__name__}"
        self.ref_graph_name = f"{self.ref_model_name}.dot"
        self.ov_model = self._create_ov_model(**kwargs)

    @abstractmethod
    def _create_ov_model(self) -> ov.Model:
        pass


@SYNTHETIC_MODELS.register()
class LinearModel(OVReferenceModel):
    def _create_ov_model(self, input_shape=None, reshape_shape=None, matmul_w_shape=None, add_shape=None):
        if input_shape is None:
            input_shape = [1, 3, 4, 2]
        if reshape_shape is None:
            reshape_shape = (1, 3, 2, 4)
        if matmul_w_shape is None:
            matmul_w_shape = (4, 5)
        if add_shape is None:
            add_shape = (1, 3, 2, 4)

        input_1 = opset.parameter(input_shape, name="Input")
        reshape = opset.reshape(input_1, reshape_shape, special_zero=False, name="Reshape")
        data = self._rng.random(matmul_w_shape).astype(np.float32) - 0.5
        matmul = opset.matmul(reshape, data, transpose_a=False, transpose_b=False, name="MatMul")
        add = opset.add(reshape, self._rng.random(add_shape).astype(np.float32), name="Add")
        r1 = opset.result(matmul, name="Result_MatMul")
        # TODO(KodiaqQ): Remove this after fix - CVS-100010
        r1.get_output_tensor(0).set_names(set(["Result_MatMul"]))
        r2 = opset.result(add, name="Result_Add")
        r2.get_output_tensor(0).set_names(set(["Result_Add"]))
        model = ov.Model([r1, r2], [input_1])
        return model


@SYNTHETIC_MODELS.register()
class ConvModel(OVReferenceModel):
    def _create_ov_model(self):
        input_1 = opset.parameter([1, 3, 4, 2], name="Input_1")
        mean = self._rng.random((1, 3, 1, 1)).astype(np.float32)
        scale = self._rng.random((1, 3, 1, 1)).astype(np.float32) + 1e-4
        subtract = opset.subtract(input_1, mean, name="Sub")
        kernel = self._rng.random((3, 3, 1, 1)).astype(np.float32) / scale - 0.5
        strides = [1, 1]
        pads = [0, 0]
        dilations = [1, 1]
        conv = opset.convolution(subtract, kernel, strides, pads, pads, dilations, name="Conv")
        bias = opset.constant(np.zeros((1, 3, 1, 1)), dtype=np.float32, name="Bias")
        conv_add = opset.add(conv, bias, name="Conv_Add")
        relu = opset.relu(conv_add, name="Relu")

        input_2 = opset.parameter([1, 3, 2, 4], name="Input_2")
        add = opset.add(input_2, (-1) * mean, name="Add")
        multiply = opset.multiply(add, 1 / scale, name="Mul")
        transpose = opset.transpose(multiply, [0, 1, 3, 2], name="Transpose")

        cat = opset.concat([relu, transpose], axis=0)
        result = opset.result(cat, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_1, input_2])
        return model


class DepthwiseConv3DModel(OVReferenceModel):
    def _create_ov_model(self):
        input_1 = opset.parameter([1, 3, 7], name="Input_1")
        kernel = self._rng.random((3, 1, 1, 5)).astype(np.float32)
        strides = [1]
        pads = [0]
        dilations = [1]
        conv = opset.group_convolution(input_1, kernel, strides, pads, pads, dilations, name="DepthwiseConv3D")
        bias = self._rng.random((1, 3, 1)).astype(np.float32)
        add = opset.add(conv, bias, name="Add")

        result = opset.result(add, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_1])
        return model


class DepthwiseConv4DModel(OVReferenceModel):
    def _create_ov_model(self):
        input_1 = opset.parameter([1, 3, 5, 5], name="Input_1")
        kernel = self._rng.random((3, 1, 1, 3, 3)).astype(np.float32)
        strides = [1, 1]
        pads = [0, 0]
        dilations = [1, 1]
        conv = opset.group_convolution(input_1, kernel, strides, pads, pads, dilations, name="DepthwiseConv4D")
        bias = self._rng.random((1, 3, 1, 1)).astype(np.float32)
        add = opset.add(conv, bias, name="Add")
        relu = opset.relu(add, name="Relu")

        result = opset.result(relu, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_1])
        return model


class DepthwiseConv5DModel(OVReferenceModel):
    def _create_ov_model(self):
        input_1 = opset.parameter([1, 3, 7, 6, 5], name="Input_1")
        kernel = self._rng.random((3, 1, 1, 5, 4, 3)).astype(np.float32)
        strides = [1, 1, 1]
        pads = [0, 0, 0]
        dilations = [1, 1, 1]
        conv = opset.group_convolution(input_1, kernel, strides, pads, pads, dilations, name="DepthwiseConv5D")
        bias = self._rng.random((1, 3, 1, 1, 1)).astype(np.float32)
        add = opset.add(conv, bias, name="Add")

        result = opset.result(add, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_1])
        return model


class QuantizedModel(OVReferenceModel):
    @staticmethod
    def _create_fq_node(parent_node, name):
        # OV bug with FQ element types after fusing preprocessing
        return opset.fake_quantize(
            parent_node, np.float32(-1), np.float32(1), np.float32(-1), np.float32(1), 256, name=name
        )

    def _create_ov_model(self):
        input_1 = opset.parameter([1, 3, 14, 28], name="Input_1")
        conv_1_fq_input = self._create_fq_node(input_1, name="Conv_1/fq_input_0")

        mean = self._rng.random((1, 3, 1, 1)).astype(np.float32)
        scale = self._rng.random((1, 3, 1, 1)).astype(np.float32) + 1e-4
        kernel = self._rng.random((3, 3, 1, 1)).astype(np.float32) / scale
        strides = [1, 1]
        pads = [0, 0]
        dilations = [1, 1]
        conv_1_fq_weights = self._create_fq_node(kernel, name="Conv_1/fq_weights_0")
        conv_1 = opset.convolution(conv_1_fq_input, conv_1_fq_weights, strides, pads, pads, dilations, name="Conv_1")
        relu_1 = opset.relu(conv_1, name="Relu_1")

        input_2 = opset.parameter([1, 3, 28, 14], name="Input_2")
        multiply = opset.multiply(input_2, 1 / scale, name="Mul")
        add_1 = opset.add(multiply, (-1) * mean, name="Add_1")
        transpose_fq_input = self._create_fq_node(add_1, name="Transpose/fq_input_0")
        transpose = opset.transpose(transpose_fq_input, [0, 1, 3, 2], name="Transpose")

        cat_fq_input = self._create_fq_node(relu_1, name="Concat_1/fq_input_0")
        cat_1 = opset.concat([cat_fq_input, transpose], axis=1, name="Concat_1")

        kernel = self._rng.random((12, 6, 1, 1)).astype(np.float32)
        conv_2_fq_weights = self._create_fq_node(kernel, name="Conv_2/fq_weights_0")
        conv_2 = opset.convolution(cat_1, conv_2_fq_weights, strides, pads, pads, dilations, name="Conv_2")
        relu_2 = opset.relu(conv_2, name="Relu_2")

        kernel = self._rng.random((6, 12, 1, 1)).astype(np.float32)
        conv_3_fq_input = self._create_fq_node(relu_2, name="Conv_3/fq_input_0")
        conv_3_fq_weights = self._create_fq_node(kernel, name="Conv_3/fq_weights_0")
        conv_3 = opset.convolution(conv_3_fq_input, conv_3_fq_weights, strides, pads, pads, dilations, name="Conv_3")

        mean = self._rng.random((1, 6, 1, 1)).astype(np.float32)
        add_2_const = opset.constant((-1) * mean)
        add_2_fq_weights = self._create_fq_node(add_2_const, name="Add_2/fq_weights_0")
        add_2 = opset.add(cat_1, add_2_fq_weights, name="Add_2")

        cat_2 = opset.concat([conv_3, add_2], axis=1, name="Concat_2")

        reshape = opset.reshape(cat_2, (-1, 2352), True)
        matmul_constant = self._rng.random((100, 2352)).astype(np.float32)
        matmul = opset.matmul(reshape, matmul_constant, False, True)
        result = opset.result(matmul, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_1, input_2])
        return model


@SYNTHETIC_MODELS.register()
class WeightsModel(OVReferenceModel):
    def _create_ov_model(self):
        input_1 = opset.parameter([1, 3, 5, 5], name="Input_1")
        kernel_data = self._rng.random((3, 3, 1, 1)).astype(np.float32)
        kernel = opset.constant(kernel_data, dtype=np.float32, name="conv_weights_0")
        strides = [1, 1]
        pads = [0, 0]
        dilations = [1, 1]
        conv = opset.convolution(input_1, kernel, strides, pads, pads, dilations, name="Conv")
        kernel_data_2 = self._rng.random((3, 3, 1, 1)).astype(np.float32)
        kernel_2 = opset.constant(kernel_data_2, dtype=np.float32, name="conv_weights_1")
        output_shape = [1, 1]
        conv_tr = opset.convolution_backprop_data(
            conv, kernel_2, output_shape, strides, pads, pads, dilations, name="Conv_backprop"
        )

        weights_1 = opset.constant(self._rng.random((1, 4)), dtype=np.float32, name="weights_1")
        matmul_1 = opset.matmul(conv_tr, weights_1, transpose_a=False, transpose_b=False, name="MatMul_1")
        weights_0 = opset.constant(self._rng.random((1, 1)), dtype=np.float32, name="weights_0")
        matmul_0 = opset.matmul(weights_0, matmul_1, transpose_a=False, transpose_b=False, name="MatMul_0")
        matmul = opset.matmul(matmul_0, matmul_1, transpose_a=False, transpose_b=True, name="MatMul")
        matmul_const = opset.matmul(weights_1, weights_0, transpose_a=True, transpose_b=False, name="MatMul_const")

        add = opset.add(matmul_const, matmul)
        result = opset.result(add, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_1])
        return model


@SYNTHETIC_MODELS.register()
class MatMul2DModel(OVReferenceModel):
    def _create_ov_model(self):
        input_shape = [3, 5]
        input_1 = opset.parameter(input_shape, name="Input")
        data = self._rng.random((5, 2)).astype(np.float32)
        matmul = opset.matmul(input_1, data, transpose_a=False, transpose_b=False, name="MatMul")
        add = opset.add(matmul, self._rng.random((1, 2)).astype(np.float32), name="Add")
        result = opset.result(add, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_1])
        return model


@SYNTHETIC_MODELS.register()
class ScaleShiftReluModel(OVReferenceModel):
    def _create_ov_model(self):
        input_shape = [3, 5]
        input_1 = opset.parameter(input_shape, name="Input")
        data = self._rng.random((5, 2)).astype(np.float32)
        matmul = opset.matmul(input_1, data, transpose_a=False, transpose_b=False, name="MatMul")
        multiply = opset.multiply(matmul, self._rng.random((1, 2)).astype(np.float32), name="Mul")
        add = opset.add(multiply, self._rng.random((1, 2)).astype(np.float32), name="Add")
        relu = opset.relu(add, name="Relu")
        data_2 = self._rng.random((2, 4)).astype(np.float32)
        matmul_2 = opset.matmul(relu, data_2, transpose_a=False, transpose_b=False, name="MatMul2")
        result = opset.result(matmul_2, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_1])
        return model


class FPModel(OVReferenceModel):
    def __init__(self, const_dtype: ov.Type = ov.Type.f32, input_dtype: ov.Type = ov.Type.f32):
        self.const_dtype = const_dtype
        self.input_dtype = input_dtype
        super().__init__()

    def _create_ov_model(self):
        input_shape = [1, 3, 4, 2]
        input_1 = opset.parameter(input_shape, name="Input", dtype=self.input_dtype)
        data = opset.constant(value=self._rng.random((1, 3, 4, 5)), dtype=self.const_dtype, name="MatMul_const")
        if self.const_dtype != self.input_dtype:
            data = opset.convert(data, self.input_dtype.to_string())
        matmul = opset.matmul(input_1, data, transpose_a=True, transpose_b=False, name="MatMul")
        bias = opset.constant(value=self._rng.random((1, 3, 1, 1)), dtype=self.const_dtype, name="MatMul_bias")
        if self.const_dtype != self.input_dtype:
            bias = opset.convert(bias, self.input_dtype.to_string())
        add = opset.add(matmul, bias, name="Add")
        result = opset.result(add, name="Result_Add")
        result.get_output_tensor(0).set_names(set(["Result_Add"]))
        model = ov.Model([result], [input_1])
        return model


@SYNTHETIC_MODELS.register()
class ComparisonBinaryModel(OVReferenceModel):
    def _create_ov_model(self):
        input_shape = [1, 3, 4, 2]
        input_1 = opset.parameter(input_shape, name="Input")
        data = self._rng.random(input_shape).astype(np.float32)

        mask = opset.greater_equal(input_1, data, name="GreaterEqual")
        indices = opset.convert(mask, np.int64, name="Convert")
        gather = opset.gather(input_1, indices, axis=0, batch_dims=0)

        add = opset.add(input_1, gather, name="Add")
        result = opset.result(add, name="Result_Add")
        result.get_output_tensor(0).set_names(set(["Result_Add"]))
        model = ov.Model([result], [input_1])
        return model


@SYNTHETIC_MODELS.register()
class DynamicModel(OVReferenceModel):
    def _create_ov_model(self):
        dynamic_axis = ov.Dimension(1, 99)
        input_1_shape = ov.PartialShape([dynamic_axis, 3, 4, 2])
        input_2_shape = ov.PartialShape([dynamic_axis, 3, 2, 4])

        input_1 = opset.parameter(input_1_shape, name="Input_1")
        mean = self._rng.random((1, 3, 1, 1)).astype(np.float32)
        scale = self._rng.random((1, 3, 1, 1)).astype(np.float32) + 1e-4
        subtract = opset.subtract(input_1, mean, name="Sub")
        kernel = self._rng.random((3, 3, 1, 1)).astype(np.float32) / scale - 0.5
        strides = [1, 1]
        pads = [0, 0]
        dilations = [1, 1]
        conv = opset.convolution(subtract, kernel, strides, pads, pads, dilations, name="Conv")
        bias = opset.constant(np.zeros((1, 3, 1, 1)), dtype=np.float32, name="Bias")
        conv_add = opset.add(conv, bias, name="Conv_Add")
        relu = opset.relu(conv_add, name="Relu")

        input_2 = opset.parameter(input_2_shape, name="Input_2")
        add = opset.add(input_2, (-1) * mean, name="Add")
        multiply = opset.multiply(add, 1 / scale, name="Mul")
        transpose = opset.transpose(multiply, [0, 1, 3, 2], name="Transpose")

        cat = opset.concat([relu, transpose], axis=0)
        result = opset.result(cat, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_1, input_2])
        return model


@SYNTHETIC_MODELS.register()
class ShapeOfModel(OVReferenceModel):
    def _create_ov_model(self):
        input_1 = opset.parameter([1, 3, 4, 2], name="Input")
        scale = self._rng.random((1, 3, 1, 1)).astype(np.float32) + 1e-4
        kernel = self._rng.random((3, 3, 1, 1)).astype(np.float32) / scale - 0.5
        strides = [1, 1]
        pads = [0, 0]
        dilations = [1, 1]
        conv_1 = opset.convolution(input_1, kernel, strides, pads, pads, dilations, name="Conv_1")
        bias_1 = opset.constant(np.zeros((1, 3, 1, 1)), dtype=np.float32, name="Bias_1")
        conv_add_1 = opset.add(conv_1, bias_1, name="Conv_Add_1")

        # ShapeOf subgraph
        shape_of_1 = opset.shape_of(conv_add_1, name="ShapeOf_1")
        gather = opset.gather(shape_of_1, indices=np.int64([2, 3]), axis=np.int64(0))
        cat = opset.concat([np.int64([0]), np.int64([0]), gather], axis=0)
        reshape_1 = opset.reshape(conv_add_1, output_shape=cat, special_zero=True, name="Reshape_1")
        transpose = opset.transpose(reshape_1, input_order=np.int64([0, 1, 3, 2]), name="Transpose")

        conv_2 = opset.convolution(transpose, kernel, strides, pads, pads, dilations, name="Conv_2")
        bias_2 = opset.constant(np.zeros((1, 3, 1, 1)), dtype=np.float32, name="Bias_2")
        conv_add_2 = opset.add(conv_2, bias_2, name="Conv_Add_2")

        # ShapeOf subgraph
        shape_of_2 = opset.shape_of(conv_add_2, name="ShapeOf_2")
        convert_1 = opset.convert(shape_of_2, destination_type="f32", name="Convert_1")
        multiply = opset.multiply(convert_1, np.float32([1, 1, 1, 1]), name="Multiply")
        convert_2 = opset.convert(multiply, destination_type="i64", name="Convert_2")
        reshape_2 = opset.reshape(conv_add_2, output_shape=convert_2, special_zero=True, name="Reshape_2")

        conv_3 = opset.convolution(reshape_2, kernel, strides, pads, pads, dilations, name="Conv_3")
        bias_3 = opset.constant(np.zeros((1, 3, 1, 1)), dtype=np.float32, name="Bias_3")
        conv_add_3 = opset.add(conv_3, bias_3, name="Conv_Add_3")

        result = opset.result(conv_add_3, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_1])
        return model


@SYNTHETIC_MODELS.register()
class ConvNotBiasModel(OVReferenceModel):
    def _create_ov_model(self):
        input_1 = opset.parameter([1, 3, 4, 2], name="Input_1")
        mean = self._rng.random((1, 3, 1, 1)).astype(np.float32)
        scale = self._rng.random((1, 3, 1, 1)).astype(np.float32) + 1e-4
        subtract = opset.subtract(input_1, mean, name="Sub")
        kernel = self._rng.random((3, 3, 1, 1)).astype(np.float32) / scale - 0.5
        strides = [1, 1]
        pads = [0, 0]
        dilations = [1, 1]
        conv = opset.convolution(subtract, kernel, strides, pads, pads, dilations, name="Conv")
        not_bias = opset.constant(np.zeros((1, 3, 4, 2)), dtype=np.float32, name="NotBias")
        conv_add = opset.add(conv, not_bias, name="Conv_Add")
        relu = opset.relu(conv_add, name="Relu")

        result = opset.result(relu, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_1])
        return model


class MatMul2DNotBiasModel(OVReferenceModel):
    def _create_ov_model(self):
        input_shape = [2, 5, 4, 3]
        input_1 = opset.parameter(input_shape, name="Input")
        data = self._rng.random((3, 4)).astype(np.float32)
        matmul = opset.matmul(input_1, data, transpose_a=False, transpose_b=False, name="MatMul")
        add = opset.add(matmul, self._rng.random((1, 5, 4, 4)).astype(np.float32), name="Add")
        result = opset.result(add, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_1])
        return model


@SYNTHETIC_MODELS.register()
class LSTMModel(OVReferenceModel):
    def _create_ov_model(self):
        input_1 = opset.parameter([1, 1, 128], name="Input")
        squeeze = opset.squeeze(input_1, np.int64(1), name="Squeeze_1")
        data = self._rng.random((256, 128)).astype(np.float32)
        matmul_1 = opset.matmul(squeeze, data, transpose_a=False, transpose_b=True, name="MatMul_1")

        variable_1 = "variable_id_1"
        data = self._rng.random((1, 64)).astype(np.float32)
        read_value_1 = opset.read_value(data, variable_1, name="ReadValue_1")
        data = self._rng.random((1, 64)).astype(np.float32)
        add = opset.add(read_value_1, data, name="Add")
        data = self._rng.random((256, 64)).astype(np.float32)
        matmul_2 = opset.matmul(add, data, transpose_a=False, transpose_b=True, name="MatMul_2")

        add_1 = opset.add(matmul_1, matmul_2, name="Add_1")
        split_1 = opset.split(add_1, axis=1, num_splits=4)
        split_outputs = split_1.outputs()

        sigmoid_1 = opset.sigmoid(split_outputs[0], name="Sigmoid_1")
        sigmoid_2 = opset.sigmoid(split_outputs[1], name="Sigmoid_2")
        sigmoid_3 = opset.sigmoid(split_outputs[2], name="Sigmoid_3")
        tanh_1 = opset.tanh(split_outputs[3], name="Tanh_1")

        data = self._rng.random((1, 64)).astype(np.float32)
        multiply_1 = opset.multiply(sigmoid_1, data, name="Multiply_1")
        multiply_2 = opset.multiply(tanh_1, sigmoid_2, name="Multiply_2")

        add_2 = opset.add(multiply_1, multiply_2, name="Add_2")
        tanh_2 = opset.tanh(add_2, name="Tanh_2")

        multiply_3 = opset.multiply(sigmoid_3, tanh_2, name="Multiply_3")
        assign_1 = opset.assign(multiply_3, variable_1, name="Assign_1")

        data = self._rng.random((128, 64)).astype(np.float32)
        matmul_3 = opset.matmul(multiply_3, data, transpose_a=False, transpose_b=True, name="MatMul_3")

        result = opset.result(matmul_3, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model(results=[result], sinks=[assign_1], parameters=[input_1], name="LSTMModel")
        return model


@SYNTHETIC_MODELS.register()
class LSTMSequenceModel(OVReferenceModel):
    def _create_ov_model(self):
        x = opset.parameter([1, 2, 16], name="X")
        initial_hidden_state = opset.parameter([1, 1, 128], name="initial_hidden_state")
        initial_cell_state = opset.parameter([1, 1, 128], name="initial_cell_state")
        seq_len = opset.constant(np.array([2]), dtype=np.int32)

        W = opset.constant(np.zeros(([1, 512, 16])), dtype=np.float32)
        R = opset.constant(np.zeros(([1, 512, 128])), dtype=np.float32)
        B = opset.constant(np.zeros(([1, 512])), dtype=np.float32)

        lstm = opset.lstm_sequence(
            x, initial_hidden_state, initial_cell_state, seq_len, W, R, B, 128, "FORWARD", name="LSTMSequence"
        )
        data = self._rng.random((1, 1, 128, 3)).astype(np.float32)
        matmul = opset.matmul(lstm.output(1), data, transpose_a=False, transpose_b=False, name="MatMul")

        result = opset.result(matmul, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model(results=[result], parameters=[x, initial_hidden_state, initial_cell_state])
        return model


class GRUSequenceModel(OVReferenceModel):
    def _create_ov_model(self, linear_before_reset=True):
        hidden_size = 128

        x = opset.parameter([3, 2, 16], name="X")
        initial_hidden_state = opset.parameter([3, 1, hidden_size], name="initial_hidden_state")
        seq_len = opset.constant(np.array([1, 2, 3]), dtype=np.int32)

        scale_factor = 4 if linear_before_reset else 3
        W = opset.constant(np.zeros(([1, 3 * hidden_size, 16])), dtype=np.float32)
        R = opset.constant(np.zeros(([1, 3 * hidden_size, hidden_size])), dtype=np.float32)
        B = opset.constant(np.zeros(([1, scale_factor * hidden_size])), dtype=np.float32)

        gru = opset.gru_sequence(
            x,
            initial_hidden_state,
            seq_len,
            W,
            R,
            B,
            hidden_size,
            direction="FORWARD",
            linear_before_reset=linear_before_reset,
            name="GRUSequence",
        )
        data = self._rng.random((3, 1, hidden_size, 3)).astype(np.float32)
        matmul = opset.matmul(gru.output(0), data, transpose_a=False, transpose_b=False, name="MatMul")

        result = opset.result(matmul, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model(results=[result], parameters=[x, initial_hidden_state])
        return model


class MatmulSoftmaxMatmulBlock(OVReferenceModel):
    def _create_ov_model(self):
        input_1 = opset.parameter([1, 1, 1], name="Input")
        squeeze = opset.squeeze(input_1, np.int64(1), name="Squeeze_1")
        data = self._rng.random((1, 1)).astype(np.float32)
        matmul_1 = opset.matmul(input_1, data, transpose_a=False, transpose_b=True, name="MatMul_1")
        softmax_1 = opset.softmax(matmul_1, axis=1, name="Softmax_1")

        matmul_2 = opset.matmul(softmax_1, squeeze, transpose_a=False, transpose_b=True, name="MatMul_2")

        result = opset.result(matmul_2, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_1])
        return model


class SimpleSplitModel(OVReferenceModel):
    def _create_ov_model(self, input_shape=None, splits=None):
        if input_shape is None:
            input_shape = [1, 9, 4, 4]
            splits = 3
        input_1 = opset.parameter(input_shape, name="Input")
        split = opset.split(input_1, 1, splits, name="Split")
        results = []
        for idx, output in enumerate(split.outputs()):
            results.append(opset.result(output, name=f"Result_{idx}"))

        model = ov.Model(results, [input_1])
        return model


@SYNTHETIC_MODELS.register()
class SharedConvModel(OVReferenceModel):
    def _create_ov_model(self, input_name="Input", input_shape=(1, 3, 3, 3), kernel=None) -> ov.Model:
        input_1 = opset.parameter(input_shape, name=input_name)
        if kernel is None:
            c_in = input_shape[1]
            kernel = self._rng.random((3, c_in, 1, 1))
        const_kernel = opset.constant(kernel, np.float32, name="Shared_conv_w")
        strides = [1, 1]
        pads = [0, 0]
        dilations = [1, 1]
        conv_1 = opset.convolution(input_1, const_kernel, strides, pads, pads, dilations, name="Conv_1")
        conv_2 = opset.convolution(input_1, const_kernel, strides, pads, pads, dilations, name="Conv_2")
        result_1 = opset.result(conv_1, name="Result_1")
        result_2 = opset.result(conv_2, name="Result_2")
        model = ov.Model([result_1, result_2], [input_1])
        return model


class SplitConcatModel(OVReferenceModel):
    def _create_ov_model(self, input_name) -> ov.Model:
        input_1 = opset.parameter([1, 3, 3, 3], name=input_name)
        split = opset.split(input_1, 1, 3, name="split")
        add_const = np.array(1).astype(np.float32)
        add_1 = opset.add(split.output(0), add_const, name="add_1")
        add_2 = opset.add(split.output(1), add_const, name="add_2")
        add_3 = opset.add(split.output(2), add_const, name="add_3")
        concat = opset.concat([add_1, add_2, add_3], 1, name="concat")
        add_4 = opset.add(concat, add_const, name="add_4")
        add_5 = opset.add(concat, add_const, name="add_5")
        result_1 = opset.result(add_4, name="result_1")
        result_2 = opset.result(add_5, name="result_2")
        model = ov.Model([result_1, result_2], [input_1])
        return model


@SYNTHETIC_MODELS.register()
class IntegerModel(OVReferenceModel):
    def _create_ov_model(self, dim1=1, dim2=7, dim3=6, max_input_value=2, add_batch_dimension=False):
        input_1 = opset.parameter([dim1, dim2, dim1], name="Input")
        convert_1 = opset.convert(input_1, destination_type="i64", name="Convert_1")

        gather_1 = opset.gather(convert_1, 0, axis=0, batch_dims=0)
        gather_1.set_friendly_name("Gather_1")

        gather_2_data = opset.constant(
            self._rng.random((max_input_value + 1, dim3)), dtype=np.float32, name="gather_2_data"
        )
        gather_2 = opset.gather(gather_2_data, gather_1, axis=0, batch_dims=0)
        gather_2.set_friendly_name("Gather_2")

        gather_3 = opset.gather(gather_2, 2, axis=0, batch_dims=0)
        if add_batch_dimension:
            gather_3 = opset.unsqueeze(gather_3, 0)
        gather_3.set_friendly_name("Gather_3")

        matmul_1_data = opset.constant(self._rng.random((dim3, dim3)), dtype=np.float32, name="matmul_1_data")
        matmul_1 = opset.matmul(gather_3, matmul_1_data, transpose_a=False, transpose_b=True, name="MatMul_1")

        gather_4 = opset.gather(input_1, 0, axis=2, batch_dims=0)
        if add_batch_dimension:
            gather_4 = opset.unsqueeze(gather_4, 0)
        gather_4.set_friendly_name("Gather_4")

        matmul_2_data = opset.constant(self._rng.random((dim3, dim2)), dtype=np.float32, name="matmul_2_data")
        matmul_2 = opset.matmul(gather_4, matmul_2_data, transpose_a=False, transpose_b=True, name="MatMul_2")
        add_1 = opset.add(matmul_1, matmul_2, name="Add_1")

        result = opset.result(add_1, name="Result")
        model = ov.Model([result], [input_1])
        return model


@SYNTHETIC_MODELS.register()
class SeBlockModel(OVReferenceModel):
    def _create_ov_model(self):
        input_shape = [1, 3, 5, 6]
        strides = [1, 1]
        pads = [0, 0]
        dilations = [1, 1]

        input_1 = opset.parameter(input_shape, name="Input")
        kernel0 = self._rng.random((3, 3, 1, 1)).astype(np.float32)
        conv0 = opset.convolution(input_1, kernel0, strides, pads, pads, dilations, name="Conv0")
        add0 = opset.add(conv0, self._rng.random((1, 3, 1, 1)).astype(np.float32), name="Add0")

        reduce_mean = opset.reduce_mean(add0, np.array([2, 3]), keep_dims=True)
        kernel = self._rng.random((5, 3, 1, 1)).astype(np.float32)
        conv = opset.convolution(reduce_mean, kernel, strides, pads, pads, dilations, name="Conv")
        add = opset.add(conv, self._rng.random((1, 5, 1, 1)).astype(np.float32), name="Add")
        relu = opset.relu(add, name="Relu")

        kernel2 = self._rng.random((3, 5, 1, 1)).astype(np.float32)
        conv2 = opset.convolution(relu, kernel2, strides, pads, pads, dilations, name="Conv2")
        add2 = opset.add(conv2, self._rng.random((1, 3, 1, 1)).astype(np.float32), name="Add2")
        sigmoid = opset.sigmoid(add2, name="Sigmoid")
        multiply = opset.multiply(add0, sigmoid, name="Mul")

        data = self._rng.random((6, 5)).astype(np.float32)
        matmul = opset.matmul(multiply, data, transpose_a=False, transpose_b=False, name="MatMul")
        result_1 = opset.result(matmul, name="Result")
        model = ov.Model([result_1], [input_1])
        return model


class ZeroRankEltwiseModel(OVReferenceModel):
    def _create_ov_model(self):
        input_shape = [1, 3, 5, 6]

        input_1 = opset.parameter(input_shape, name="Input")
        add = opset.add(input_1, np.array(1.0, dtype=np.float32), name="Add")
        result_1 = opset.result(add, name="Result")
        model = ov.Model([result_1], [input_1])
        return model


class ParallelEdgesModel(OVReferenceModel):
    def _create_ov_model(self) -> ov.Model:
        input_shape = [1, 3, 3]

        input_1 = opset.parameter(input_shape, name="Input")
        mm = opset.matmul(input_1, input_1, False, False, name="Mm")
        add = opset.add(input_1, np.array(1.0, dtype=np.float32), name="Add")
        result_0 = opset.result(mm, name="Result_mm")
        result_1 = opset.result(add, name="Result_add")
        model = ov.Model([result_0, result_1], [input_1])
        return model


@SYNTHETIC_MODELS.register()
class UnifiedEmbeddingModel(OVReferenceModel):
    def _create_ov_model(self):
        input_1 = opset.parameter([1, 3], name="Input")
        convert_1 = opset.convert(input_1, destination_type="i64", name="Convert_1")

        gather_1_data = opset.constant(self._rng.random((4, 5)), dtype=np.float32, name="gather_1_data")
        gather_1 = opset.gather(gather_1_data, convert_1, axis=0, batch_dims=0)
        gather_1.set_friendly_name("Gather_1")

        matmul_1_data = opset.constant(self._rng.random((3, 3, 5)), dtype=np.float32, name="matmul_1_data")
        matmul_1 = opset.matmul(input_1, matmul_1_data, transpose_a=False, transpose_b=False, name="MatMul_1")
        reshape_1 = opset.reshape(matmul_1, [1, 3, 5], special_zero=False, name="Reshape_1")

        concat_1 = opset.concat([gather_1, reshape_1], axis=1)

        matmul_2_data = opset.constant(self._rng.random((1, 5)), dtype=np.float32, name="matmul_2_data")
        matmul_2 = opset.matmul(concat_1, matmul_2_data, transpose_a=False, transpose_b=True, name="MatMul_2")

        result = opset.result(matmul_2, name="Result")
        model = ov.Model([result], [input_1])
        return model


@SYNTHETIC_MODELS.register()
class GroupNormalizationModel(OVReferenceModel):
    def _create_ov_model(self):
        groups_num = 2
        channels = 4
        input_1 = opset.parameter([1, groups_num, 3, 4, 4], name="Input_1")

        kernel = self._rng.random((channels, groups_num, 3, 3, 3)).astype(np.float32)
        strides = [1, 1, 1]
        pads = [0, 0, 0]
        dilations = [1, 1, 1]
        conv = opset.convolution(input_1, kernel, strides, pads, pads, dilations, name="Conv")
        bias = opset.constant(np.zeros((1, 1, 3, 1, 1)), dtype=np.float32, name="Bias")
        conv_add = opset.add(conv, bias, name="Conv_Add")

        scale = self._rng.random(channels).astype(np.float32)
        bias = self._rng.random(channels).astype(np.float32)
        group_norm = opset.group_normalization(conv_add, scale, bias, num_groups=channels, epsilon=1e-5)

        relu = opset.relu(group_norm, name="Relu")

        mean = self._rng.random((1, channels, 1, 1, 1)).astype(np.float32)
        scale = self._rng.random((1, channels, 1, 1, 1)).astype(np.float32)
        multiply = opset.multiply(relu, 1 / scale, name="Mul")
        add = opset.add(multiply, (-1) * mean, name="Add")

        result = opset.result(add, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_1])
        return model


class IfModel(OVReferenceModel):
    def _create_ov_model(self):
        input_1 = opset.parameter([1, 3, 4, 2], name="Input_1")
        input_2 = opset.parameter([1, 3, 2, 4], name="Input_2")
        input_3 = opset.parameter([], dtype=bool, name="Cond_input")

        then_body = ConvModel().ov_model
        else_body = ConvModel().ov_model

        if_node = opset.if_op(input_3)
        if_node.set_then_body(then_body)
        if_node.set_else_body(else_body)
        if_node.set_input(input_1.outputs()[0], then_body.get_parameters()[0], else_body.get_parameters()[0])
        if_node.set_input(input_2.outputs()[0], then_body.get_parameters()[1], else_body.get_parameters()[1])
        if_node.set_output(then_body.results[0], else_body.results[0])
        result = opset.result(if_node, name="Result")
        model = ov.Model([result], [input_1, input_2, input_3])
        return model


class SequentialMatmulModel(OVReferenceModel):
    """
    Model for mixed precision weight compression.
    Matrices with outliers are defined in such a way that there is a different nf4, int8, relative error.
    rel_error = nf4_error / int8_error
    The maximum relative error is achieved with not maximum outlier 10000, because nf4 better copes with outliers.

    [[   0.    1.    2.]
    [   3.    4.    5.]
    [   6.    7. 1000.]]
        nf4 error = 28
        int8 error = 13
        rel_error=2

    [[ 0.  1.  2.]
    [ 3.  4.  5.]
    [ 6.  7. 10000.]]
        nf4 error = 28
        int8 error = 40
        rel_error= 0.7

    [[ 0.  1.  2.]
    [ 3.  4.  5.]
    [ 6.  7. 10.]]
        nf4 error = 0.06
        int8 error = 16
        rel_error= 0.03
    """

    def _create_ov_model(self):
        input_node = opset.parameter([1, 3, 3], name="Input_1")
        main_values = [10000, 1000, 1, 10, 10000]

        last_node = input_node
        for i, main_value in enumerate(main_values):
            weights_data = np.arange(0, 9).reshape(3, 3)
            weights_data[-1, -1] = main_value
            current_weights = opset.constant(weights_data, dtype=np.float32, name=f"weights_{i}")
            current_node = opset.matmul(
                last_node, current_weights, transpose_a=False, transpose_b=True, name=f"MatMul_{i}"
            )
            last_node = current_node

        result = opset.result(last_node, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_node])
        return model


class IdentityMatmul(OVReferenceModel):
    def _create_ov_model(self, weights_dtype: Optional[ov.Type] = None, activation_dtype: Optional[ov.Type] = None):
        """
        :param: weights_dtype: precision of weights
        :param: activation_dtype: precision of activations
        """
        weights_dtype = ov.Type.f32 if weights_dtype is None else weights_dtype
        activation_dtype = ov.Type.f32 if activation_dtype is None else activation_dtype

        input_node = opset.parameter([1, 3, 3], dtype=activation_dtype, name="Input_1")
        weights_data = np.eye(3) * 255
        current_weights = opset.constant(weights_data, dtype=weights_dtype, name="weights")
        if weights_dtype != activation_dtype:
            current_weights = opset.convert(current_weights, activation_dtype, name="weights/convert")
        matmul_node = opset.matmul(input_node, current_weights, transpose_a=False, transpose_b=True, name="MatMul")
        result = opset.result(matmul_node, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_node])
        return model


class GatherWithTwoReductionAxes(OVReferenceModel):
    def _create_ov_model(self):
        input_1 = opset.parameter([2, 3], name="Input")
        convert_1 = opset.convert(input_1, destination_type="i64", name="Convert_1")

        gather_1_data = opset.constant(self._rng.random((3, 2, 1)), dtype=np.float32, name="gather_1_data")
        gather_1 = opset.gather(gather_1_data, convert_1, axis=0, batch_dims=0)
        gather_1.set_friendly_name("Gather_1")

        result = opset.result(gather_1, name="Result")
        model = ov.Model([result], [input_1])
        return model


class GatherAndMatmulShareData(OVReferenceModel):
    def _create_ov_model(self):
        input_1 = opset.parameter([2, 3], name="Input")
        convert_1 = opset.convert(input_1, destination_type="i64", name="Convert_1")

        shared_data = opset.constant(self._rng.random((2, 2)), dtype=np.float32, name="shared_data")
        gather_1 = opset.gather(shared_data, convert_1, axis=0, batch_dims=0)
        gather_1.set_friendly_name("Gather_1")

        gather_2_data = opset.constant(self._rng.random((2, 1)), dtype=np.float32, name="gather_2_data")
        gather_2 = opset.gather(gather_2_data, convert_1, axis=0, batch_dims=0)
        gather_2.set_friendly_name("Gather_2")

        matmul_1_data = opset.constant(self._rng.random((2, 3)), dtype=np.float32, name="matmul_1_data")
        matmul_1 = opset.matmul(input_1, matmul_1_data, transpose_a=False, transpose_b=True, name="MatMul_1")

        matmul_2 = opset.matmul(matmul_1, shared_data, transpose_a=False, transpose_b=True, name="MatMul_2")

        result = opset.result(matmul_2, name="  Result")
        model = ov.Model([result, gather_2, gather_1], [input_1])
        return model


class ScaledDotProductAttentionModel(OVReferenceModel):
    def _create_ov_model(self):
        input_ = opset.parameter([1, 1, 1, 64], name="Input_1")
        attn_mask = opset.parameter([1, 1, 1, 1], name="Input_2")
        x = opset.reshape(input_, [64], False)
        x = opset.reshape(x, [1, 1, 1, 64], False)

        # Parallel edges are not supported by PTQ for now.
        # Ref 148498
        inputs = []
        for _ in range(3):
            x_ = opset.reshape(x, [64], False)
            x_ = opset.reshape(x_, [1, 1, 1, 64], False)
            inputs.append(x_)

        attn = opset.scaled_dot_product_attention(*inputs, attn_mask)
        result = opset.result(attn, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_, attn_mask])
        return model


class LinearQuantizedModel(OVReferenceModel):
    @staticmethod
    def _create_fq_node(parent_node, name):
        return opset.fake_quantize(
            parent_node, np.float32(-1), np.float32(1), np.float32(-1), np.float32(1), 256, name=name
        )

    def _create_ov_model(self):
        inputs = opset.parameter((1, 3, 4, 2), name="Input")

        w = self._rng.random((2, 5), dtype=np.float32)
        x = opset.matmul(
            self._create_fq_node(inputs, "FQ_Inputs"),
            self._create_fq_node(w, "FQ_Weights_0"),
            transpose_a=False,
            transpose_b=False,
            name="MatMul_0",
        )
        x = opset.relu(x, name="ReLu_0")

        w = self._rng.random((5, 2), dtype=np.float32)
        x = opset.matmul(
            self._create_fq_node(x, "FQ_ReLu_0"),
            self._create_fq_node(w, "FQ_Weights_1"),
            transpose_a=False,
            transpose_b=False,
            name="MatMul_1",
        )
        x = opset.relu(x, name="ReLu_1")

        x = opset.result(x, name="Result")
        x.get_output_tensor(0).set_names(set(["Result"]))

        model = ov.Model([x], [inputs])
        return model


class AWQMatmulModel(OVReferenceModel):
    """
    Model for testing AWQ algorithm. Contains MatMul->Multiply->MatMul pattern.
    """

    @staticmethod
    def get_weights(weights_data, is_int8, name):
        if not is_int8:
            return opset.constant(weights_data, dtype=np.float32, name=name)
        else:
            qw = opset.constant(weights_data, dtype=np.uint8, name="qw_" + name)
            qw = opset.convert(qw, destination_type=np.float32)

            zp = opset.constant(np.array([2**7]), dtype=np.uint8, name="zp_" + name)
            zp = opset.convert(zp, destination_type=np.float32)

            scale = opset.constant(
                np.ones((weights_data.shape[0], 1), dtype=np.float32), dtype=np.float32, name="scale_" + name
            )
            return (qw - zp) * scale

    def _create_ov_model(self, n_extra_dims: int = 1, is_int8=False):
        input_node = opset.parameter([1] * n_extra_dims + [-1, 8], name="Input_1")

        weights_data1 = np.arange(0, 64).reshape(8, 8)
        weights_data1[:] = 2.0
        weights1 = self.get_weights(weights_data1, is_int8, name="weights_1")
        node1 = opset.matmul(input_node, weights1, transpose_a=False, transpose_b=True, name="MatMul_1")

        weights_data2 = np.arange(0, 64).reshape(8, 8)
        weights_data2[:] = 3.0
        weights2 = self.get_weights(weights_data2, is_int8, name="weights_2")
        node2 = opset.matmul(input_node, weights2, transpose_a=False, transpose_b=True, name="MatMul_2")

        node_multiply = opset.multiply(node1, node2, name="Multiply")

        weights_data3 = np.arange(0, 64).reshape(8, 8)
        weights_data3[:] = 4.0
        weights3 = self.get_weights(weights_data3, is_int8, name="weights_3")
        node3 = opset.matmul(node_multiply, weights3, transpose_a=False, transpose_b=True, name="MatMul_3")

        weights_data4 = np.arange(0, 64).reshape(8, 8)
        weights_data4[:] = 2.0
        weights4 = self.get_weights(weights_data4, is_int8, name="weights_4")
        node4 = opset.matmul(node3, weights4, transpose_a=False, transpose_b=True, name="MatMul_4")

        weights_data5 = np.arange(0, 64).reshape(8, 8)
        weights_data5[:] = 3.0
        weights5 = self.get_weights(weights_data5, is_int8, name="weights_5")
        node5 = opset.matmul(node3, weights5, transpose_a=False, transpose_b=True, name="MatMul_5")

        node_multiply_2 = opset.multiply(node4, node5, name="Multiply_2")

        weights_data6 = np.arange(0, 64).reshape(8, 8)
        weights_data6[:] = 4.0
        weights6 = self.get_weights(weights_data6, is_int8, name="weights_6")
        node6 = opset.matmul(node_multiply_2, weights6, transpose_a=False, transpose_b=True, name="MatMul_6")

        result = opset.result(node6, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_node])
        return model


class AWQActMatmulModel(OVReferenceModel):
    """
    Model for testing AWQ algorithm. Contains MatMul->Multiply->MatMul pattern.
    """

    def _create_ov_model(self, is_int8=False, with_multiply=False, n_layers=8):
        input_node = opset.parameter([1, 8, 8], name="Input_1")
        weights_data = np.arange(0, 64).reshape(8, 8) - 32
        weights = AWQMatmulModel.get_weights(weights_data, is_int8, name="weights_emb")
        out_node = opset.matmul(input_node, weights, transpose_a=False, transpose_b=True, name="MatMul_emb")

        for i in range(n_layers):
            weights_data = np.arange(0, 64).reshape(8, 8) - 32
            weights = AWQMatmulModel.get_weights(weights_data, is_int8, name=f"weights_1_{i}")
            mm1 = opset.matmul(out_node, weights, transpose_a=False, transpose_b=True, name=f"MatMul_1_{i}")
            node1 = opset.relu(mm1, name=f"ReLU_{i}")

            if with_multiply:
                weights_data = np.arange(0, 64).reshape(8, 8) - 32
                weights = AWQMatmulModel.get_weights(weights_data, is_int8, name=f"weights_2_{i}")
                mm2 = opset.matmul(out_node, weights, transpose_a=False, transpose_b=True, name=f"MatMul_2_{i}")

                alpha = np.array([1.5], dtype=np.float32)
                alpha = opset.constant(alpha, dtype=np.float32)
                lambda_value = np.array([1.5], dtype=np.float32)
                lambda_value = opset.constant(lambda_value, dtype=np.float32)
                node2 = opset.selu(mm2, alpha, lambda_value, name=f"SeLU_{i}")

                node_multiply = opset.multiply(node1, node2, name=f"Multiply_{i}")
            else:
                node_multiply = node1

            out_node = node_multiply

        weights_data = np.arange(0, 64).reshape(8, 8) - 32
        weights = AWQMatmulModel.get_weights(weights_data, is_int8, name="weights_lm_head")
        out_node = opset.matmul(out_node, weights, transpose_a=False, transpose_b=True, name="MatMul_lm_head")

        result = opset.result(out_node, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input_node])
        return model


class DuplicatedNamesModel(OVReferenceModel):
    """
    Model with duplicated node names (friendly_name field).
    """

    def _create_ov_model(self):
        main_shape = [1, 2]
        input_1 = opset.parameter(main_shape, name="Parameter")

        add_1_data = self._rng.random(main_shape).astype(np.float32)
        add_1 = opset.add(input_1, add_1_data, name="Duplicate")

        matmul_1_data = self._rng.random(main_shape).astype(np.float32)
        matmul_1 = opset.matmul(add_1, matmul_1_data, transpose_a=False, transpose_b=True, name="Duplicate")

        result = opset.result(matmul_1, name="Result")
        model = ov.Model([result], [input_1])
        return model


class ModelNamedConsts(OVReferenceModel):
    """
    Model with named constant nodes (friendly_name field).
    """

    def _create_ov_model(self):
        main_shape = [1, 2]
        input_1 = opset.parameter(main_shape, name="Parameter")

        add_1_data = self._rng.random(main_shape).astype(np.float32)
        add_1_const = opset.constant(add_1_data, name="Constant_16")
        add_1 = opset.add(input_1, add_1_const, name="Add_1")

        matmul_1_data = self._rng.random(main_shape).astype(np.float32)
        matmul_1_const = opset.constant(matmul_1_data, name="Constant_1")
        matmul_1 = opset.matmul(add_1, matmul_1_const, transpose_a=False, transpose_b=True, name="MatMul_1")

        result = opset.result(matmul_1, name="Result")
        model = ov.Model([result], [input_1])
        return model


class StatefulModel(OVReferenceModel):
    """
    Stateful model for testing.
    Borrowed from https://github.com/openvinotoolkit/openvino/blob/0c552b7b152c341b5e545d131bd032fcb3cb6b86/src/bindings/python/tests/utils/helpers.py#L212
    """

    def __init__(self, stateful=True):
        super().__init__(stateful=stateful)

    def _create_ov_model(self, stateful=True):
        input_shape = [1, 8]
        data_type = np.float32
        input_data = opset.parameter(input_shape, name="input_data", dtype=data_type)
        init_val = opset.constant(np.zeros(input_shape), data_type)
        if stateful:
            rv = opset.read_value(init_val, "var_id_667", data_type, input_shape)
            add = opset.add(rv, input_data, name="MemoryAdd")
            node = opset.assign(add, "var_id_667")
            scale_val = opset.constant(np.ones(input_shape), data_type)
            scale = opset.multiply(add, scale_val, name="Scale")
            result = opset.result(scale, name="Result")
            result.get_output_tensor(0).set_names(set(["Result"]))
            model = ov.Model(results=[result], sinks=[node], parameters=[input_data], name="TestModel")
        else:
            bias = opset.constant(np.zeros(input_shape), data_type)
            add = opset.add(input_data, bias, name="Add")
            scale_val = opset.constant(np.ones(input_shape), data_type)
            scale = opset.multiply(add, scale_val, name="Scale")
            result = opset.result(scale, name="Result")
            result.get_output_tensor(0).set_names(set(["Result"]))
            model = ov.Model(results=[result], parameters=[input_data], name="TestModel")

        return model


class IfModel_2(OVReferenceModel):
    def _create_ov_model(self):
        input_1 = opset.parameter([1, 3, 4, 2], name="Input_1")
        input_2 = opset.parameter([], dtype=bool, name="Cond_input")

        then_body = ConvNotBiasModel().ov_model
        else_body = FPModel().ov_model

        if_node = opset.if_op(input_2)
        if_node.set_then_body(then_body)
        if_node.set_else_body(else_body)
        if_node.set_input(input_1.outputs()[0], then_body.get_parameters()[0], else_body.get_parameters()[0])
        if_node.set_output(then_body.results[0], else_body.results[0])
        result = opset.result(if_node, name="Result")
        model = ov.Model([result], [input_1, input_2])
        return model


class PreluModel(OVReferenceModel):
    def _create_ov_model(self):
        input = opset.parameter([1, 3, 4, 2], name="Input")
        prelu = opset.prelu(input, slope=1)
        result = opset.result(prelu, name="Result")
        result.get_output_tensor(0).set_names(set(["Result"]))
        model = ov.Model([result], [input])
        return model


class UnifiedScalesModel(OVReferenceModel):
    def _create_ov_model(self):
        input = opset.parameter([1, 3, 4, 2], name="Input")
        multiply = opset.multiply(input, self._rng.random((1, 2)).astype(np.float32), name="Mul")
        sin = opset.sin(multiply, name="Sin")
        cos = opset.cos(multiply, name="Cos")
        concat = opset.concat([sin, cos], axis=0)
        kernel = self._rng.random((3, 3, 1, 1)).astype(np.float32)
        strides = [1, 1]
        pads = [0, 0]
        dilations = [1, 1]
        conv = opset.convolution(concat, kernel, strides, pads, pads, dilations, name="Conv")
        result = opset.result(conv, name="Result")
        model = ov.Model([result], [input])
        return model


class RoPEModel(OVReferenceModel):
    def _create_ov_model(self):
        position_ids = opset.parameter([1, 10], name="position_ids")

        unsqueeze = opset.unsqueeze(position_ids, 0, name="unsqueeze")
        convert = opset.convert(unsqueeze, ov.Type.f32, name="convert")

        broadcast_data = self._rng.random((1, 5, 1)).astype(np.float32)
        broadcast_shape = [1, 5, 1]
        broadcast = opset.broadcast(broadcast_data, broadcast_shape, name="broadcast")

        matmul = opset.matmul(broadcast, convert, transpose_a=False, transpose_b=False, name="MatMul")
        transpose = opset.transpose(matmul, [0, 2, 1], name="transpose")
        concat = opset.concat([transpose], axis=0, name="concat")
        sin = opset.sin(concat, name="sin")
        cos = opset.cos(concat, name="cos")
        sin_result = opset.result(sin, name="sin_result")
        cos_result = opset.result(cos, name="cos_result")

        model = ov.Model([sin_result, cos_result], [position_ids])
        return model
