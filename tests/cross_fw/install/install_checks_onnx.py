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

from typing import List

import numpy as np

from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantizationParameters
from nncf.data.dataset import Dataset
from tests.onnx.models import LinearModel


class DatasetToTest:
    def __init__(self, input_shape: List[int], input_name: str):
        self._input_shape = input_shape
        self._input_name = input_name

    def __iter__(self):
        yield {self._input_name: np.random.random(self._input_shape).astype(np.float32)}

    def __len__(self):
        return 1


model = LinearModel()
onnx_model = model.onnx_model
input_shape = model.input_shape[0]
input_name = onnx_model.graph.input[0].name
dataset_to_test = DatasetToTest(input_shape, input_name)

quantization_parameters = PostTrainingQuantizationParameters()
quantization = PostTrainingQuantization(quantization_parameters)
quantized_model = quantization.apply(onnx_model, dataset=Dataset(dataset_to_test))
