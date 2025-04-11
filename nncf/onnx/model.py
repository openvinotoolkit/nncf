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

import copy
from typing import Dict

import numpy as np
import onnx

from nncf.onnx.graph.model_utils import extract_raw_data_from_model
from nncf.onnx.graph.model_utils import insert_raw_data_into_model


class ONNXModel:
    def __init__(self, model: onnx.ModelProto, data: Dict[str, np.ndarray]):
        """
        :param model: The ONNX model without raw data loaded into its initializer tensors.
        :param data: A dictionary where the keys are the names of the initializer tensors and
            the values are NumPy arrays representing the `raw_data` field for each corresponding tensor.
        """
        self._model = model
        self._data = data

    @property
    def model_proto(self) -> onnx.ModelProto:
        return self._model

    @property
    def tensors(self) -> Dict[str, np.ndarray]:
        return self._data

    @classmethod
    def from_model(cls, model: onnx.ModelProto) -> "ONNXModel":
        """
        Creates an instance of ONNXModel from a given ONNX model.

        :param model: The ONNX model.
        :return: An ONNXModel instance containing the model and the extracted raw data from the
            initializer tensors.
        """
        # The `extract_raw_data_from_model()` method modifies the model passed to it,
        # so we should create a copy of the original model here.
        copy_model = copy.deepcopy(model)
        data = extract_raw_data_from_model(copy_model)

        # TODO(andrey-churkin): Complete all preprocessing here
        copy_model = onnx.shape_inference.infer_shapes(copy_model)

        return cls(copy_model, data)

    def export(self) -> onnx.ModelProto:
        """
        Exports the ONNXModel instance to an ONNX model, inserting the raw data into
        its initializer tensors.

        :return: The ONNX model with the raw data inserted into its initializer tensors.
        """
        copy_model = copy.deepcopy(self._model)
        insert_raw_data_into_model(copy_model, self._data)
        return copy_model
