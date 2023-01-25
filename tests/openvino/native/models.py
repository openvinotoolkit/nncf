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

import numpy as np
from abc import ABC, abstractmethod
import openvino.runtime as ov
from openvino.runtime import opset9 as opset

from nncf.common.utils.registry import Registry

SYNTHETIC_MODELS = Registry('OV_SYNTHETIC_MODELS')


class OVReferenceModel(ABC):
    def __init__(self):
        self._rng = np.random.default_rng(seed=0)
        self.ref_graph_name = f'{self.__class__.__name__}.dot'
        self.ov_model = self._create_ov_model()

    @abstractmethod
    def _create_ov_model(self) -> ov.Model:
        pass


@SYNTHETIC_MODELS.register()
class LinearModel(OVReferenceModel):
    def _create_ov_model(self):
        input_shape = [1, 3, 4, 2]
        input_1 = opset.parameter(input_shape, name="Input")
        reshape = opset.reshape(input_1, (1, 3, 2, 4), special_zero=False, name='Reshape')
        data = self._rng.random((1, 3, 4, 5)).astype(np.float32)
        matmul = opset.matmul(reshape, data, transpose_a=False, transpose_b=False, name="MatMul")
        add = opset.add(reshape, self._rng.random((1, 3, 2, 4)).astype(np.float32), name="Add")
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
        kernel = self._rng.random((3, 3, 1, 1)).astype(np.float32) / scale
        strides = [1, 1]
        pads = [0, 0]
        dilations = [1, 1]
        conv = opset.convolution(subtract, kernel, strides, pads, pads, dilations, name="Conv")
        relu = opset.relu(conv, name="Relu")

        input_2 = opset.parameter([1, 3, 2, 4], name="Input_2")
        add = opset.add(input_2, (-1) * mean, name="Add")
        multiply = opset.multiply(add, 1 / scale, name="Mul")
        transpose = opset.transpose(multiply, [0, 1, 3, 2], name="Transpose")

        cat = opset.concat([relu, transpose], axis=0)
        result = opset.result(cat, name="Result")
        model = ov.Model([result], [input_1, input_2])
        return model


class QuantizedModel(OVReferenceModel):
    @staticmethod
    def _create_fq_node(parent_node, name):
        # OV bug with FQ element types after fusing preprocessing
        return opset.fake_quantize(parent_node,
             np.float32(-1), np.float32(1), np.float32(-1), np.float32(1), 256, name=name)

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
        model = ov.Model([result], [input_1, input_2])
        return model


class WeightsModel(OVReferenceModel):
    def _create_ov_model(self):
        input_1 = opset.parameter([1, 3, 5, 5], name="Input_1")
        kernel = self._rng.random((3, 3, 1, 1)).astype(np.float32)
        strides = [1, 1]
        pads = [0, 0]
        dilations = [1, 1]
        conv = opset.convolution(input_1, kernel, strides, pads, pads, dilations, name="Conv")
        kernel_2 = self._rng.random((3, 3, 1, 1)).astype(np.float32)
        output_shape = [1, 1]
        conv_tr = opset.convolution_backprop_data(
            conv, kernel_2, output_shape, strides, pads, pads, dilations, name="Conv_backprop")

        weights_1 = self._rng.random((1, 3, 1, 4)).astype(np.float32)
        matmul_1 = opset.matmul(conv_tr, weights_1, transpose_a=False, transpose_b=False, name="MatMul_1")
        weights_0 = self._rng.random((1, 3, 1, 1)).astype(np.float32)
        matmul_0 = opset.matmul(weights_0, matmul_1, transpose_a=False, transpose_b=False, name="MatMul_0")
        matmul = opset.matmul(matmul_0, matmul_1, transpose_a=False, transpose_b=True, name="MatMul")
        matmul_const = opset.matmul(weights_1, weights_0, transpose_a=True, transpose_b=False, name="MatMul_const")

        add = opset.add(matmul_const, matmul)
        result = opset.result(add, name="Result")
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
        result_1 = opset.result(add, name="Result")
        model = ov.Model([result_1], [input_1])
        return model


class FP16Model(OVReferenceModel):
    def __init__(self, precision='FP32'):
        self.precision = np.float32 if precision == 'FP32' else np.float16
        super().__init__()

    def _create_ov_model(self):
        input_shape = [1, 3, 4, 2]
        input_1 = opset.parameter(input_shape, name="Input")
        data = self._rng.random((1, 3, 4, 5)).astype(self.precision)
        if self.precision == np.float16:
            data = opset.convert(data, np.float32)
        matmul = opset.matmul(input_1, data, transpose_a=True, transpose_b=False, name="MatMul")
        bias = self._rng.random((1, 3, 2, 5)).astype(self.precision)
        if self.precision == np.float16:
            bias = opset.convert(bias, np.float32)
        convert_2 = opset.convert(bias, np.float32)
        add = opset.add(matmul, convert_2, name="Add")
        r1 = opset.result(add, name="Result_Add")
        model = ov.Model([r1], [input_1])
        return model
