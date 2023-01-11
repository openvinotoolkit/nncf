"""
 Copyright (c) 2022 Intel Corporation
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
