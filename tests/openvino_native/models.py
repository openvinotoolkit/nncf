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


class OVReferenceModel:
    def __init__(self, ov_model: ov.Model):
        self.ov_model = ov_model


class LinearModel(OVReferenceModel):
    def __init__(self):
        input_shape = [1, 3, 4, 2]
        input_1 = ov.opset9.parameter(input_shape, name="Input")
        reshape = ov.opset9.reshape(input_1, (1, 3, 2, 4), special_zero=False, name='Reshape')
        data = np.random.rand(1, 3, 4, 5).astype(np.float32)
        matmul = ov.opset9.matmul(reshape, data, transpose_a=False, transpose_b=False, name="MatMul")
        add = ov.opset9.add(reshape, np.random.rand(1, 3, 2, 4).astype(np.float32), name="Add")
        r1 = ov.opset9.result(matmul, name="Result_Matmul")
        r2 = ov.opset9.result(add, name="Result_Add")
        model = ov.Model([r1, r2], [input_1])

        super().__init__(model)
