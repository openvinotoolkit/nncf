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
import openvino.runtime as ov
from openvino.runtime import opset9 as opset


class OVReferenceModel:
    def __init__(self, ov_model: ov.Model):
        self.ov_model = ov_model
        self.ref_graph_name = f'{self.__class__.__name__}.dot'


class LinearModel(OVReferenceModel):
    def __init__(self):
        input_shape = [1, 3, 4, 2]
        input_1 = opset.parameter(input_shape, name="Input")
        reshape = opset.reshape(input_1, (1, 3, 2, 4), special_zero=False, name='Reshape')
        data = np.random.rand(1, 3, 4, 5).astype(np.float32)
        matmul = opset.matmul(reshape, data, transpose_a=False, transpose_b=False, name="MatMul")
        add = opset.add(reshape, np.random.rand(1, 3, 2, 4).astype(np.float32), name="Add")
        r1 = opset.result(matmul, name="Result_MatMul")
        # TODO(KodiaqQ): Remove this after fix - CVS-100010
        r1.get_output_tensor(0).set_names(set(["Result_MatMul"]))
        r2 = opset.result(add, name="Result_Add")
        r2.get_output_tensor(0).set_names(set(["Result_Add"]))
        model = ov.Model([r1, r2], [input_1])

        super().__init__(model)


class ConvModel(OVReferenceModel):
    def __init__(self):
        input_1 = opset.parameter([1, 3, 4, 2], name="Input_1")
        mean = np.random.rand(1, 3, 1, 1).astype(np.float32)
        scale = np.random.rand(1, 3, 1, 1).astype(np.float32) + 1e-4
        subtract = opset.subtract(input_1, mean, name="Sub")
        kernel = np.random.rand(3, 3, 1, 1).astype(np.float32) / scale
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

        super().__init__(model)


class QuantizedModel(OVReferenceModel):
    def __init__(self):
        input_1 = opset.parameter([1, 3, 14, 28], name="Input_1")
        conv_1_fq_input = opset.fake_quantize(input_1, -1, 1, -1, 1, 256, name="Conv_1/fq_input_0")

        mean = np.random.rand(1, 3, 1, 1).astype(np.float32)
        scale = np.random.rand(1, 3, 1, 1).astype(np.float32) + 1e-4
        kernel = np.random.rand(3, 3, 1, 1).astype(np.float32) / scale
        strides = [1, 1]
        pads = [0, 0]
        dilations = [1, 1]
        conv_1_fq_weights = opset.fake_quantize(kernel, -1, 1, -1, 1, 256, name="Conv_1/fq_weights_0")
        conv_1 = opset.convolution(conv_1_fq_input, conv_1_fq_weights, strides, pads, pads, dilations, name="Conv_1")
        relu_1 = opset.relu(conv_1, name="Relu_1")

        input_2 = opset.parameter([1, 3, 28, 14], name="Input_2")
        multiply = opset.multiply(input_2, 1 / scale, name="Mul")
        add_1 = opset.add(multiply, (-1) * mean, name="Add_1")
        transpose_fq_input = opset.fake_quantize(add_1, -1, 1, -1, 1, 256, name="Transpose/fq_input_0")
        transpose = opset.transpose(transpose_fq_input, [0, 1, 3, 2], name="Transpose")

        cat_fq_input = opset.fake_quantize(relu_1, -1, 1, -1, 1, 256, name="Concat_1/fq_input_0")
        cat_1 = opset.concat([cat_fq_input, transpose], axis=1, name="Concat_1")

        kernel = np.random.rand(12, 6, 1, 1).astype(np.float32)
        conv_2_fq_weights = opset.fake_quantize(kernel, -1, 1, -1, 1, 256, name="Conv_2/fq_weights_0")
        conv_2 = opset.convolution(cat_1, conv_2_fq_weights, strides, pads, pads, dilations, name="Conv_2")
        relu_2 = opset.relu(conv_2, name="Relu_2")

        kernel = np.random.rand(6, 12, 1, 1).astype(np.float32)
        conv_3_fq_input = opset.fake_quantize(relu_2, -1, 1, -1, 1, 256, name="Conv_3/fq_input_0")
        conv_3_fq_weights = opset.fake_quantize(kernel, -1, 1, -1, 1, 256, name="Conv_3/fq_weights_0")
        conv_3 = opset.convolution(conv_3_fq_input, conv_3_fq_weights, strides, pads, pads, dilations, name="Conv_3")

        mean = np.random.rand(1, 6, 1, 1).astype(np.float32)
        add_2_const = opset.constant((-1) * mean)
        add_2_fq_weights = opset.fake_quantize(add_2_const, -1, 1, -1, 1, 256, name="Add_2/fq_weights_0")
        add_2 = opset.add(cat_1, add_2_fq_weights, name="Add_2")

        cat_2 = opset.concat([conv_3, add_2], axis=1, name="Concat_2")

        reshape = opset.reshape(cat_2, (-1, 2352), True)
        matmul_constant = np.random.rand(100, 2352).astype(np.float32)
        matmul = opset.matmul(reshape, matmul_constant, False, True)
        result = opset.result(matmul, name="Result")
        model = ov.Model([result], [input_1, input_2])

        super().__init__(model)
