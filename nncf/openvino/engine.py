# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Tuple, Union

import numpy as np
import openvino.runtime as ov

from nncf.common.engine import Engine
from nncf.parameters import TargetDevice


class OVNativeEngine(Engine):
    """
    Implementation of the engine for OpenVINO backend.

    OVNativeEngine uses
    [OpenVINO Runtime](https://docs.openvino.ai/latest/openvino_docs_OV_UG_OV_Runtime_User_Guide.html)
    to infer the model.
    """

    def __init__(self, model: ov.Model, target_device: TargetDevice = TargetDevice.CPU):
        if target_device == TargetDevice.ANY:
            target_device = TargetDevice.CPU

        ie = ov.Core()
        self.compiled_model = ie.compile_model(model, target_device.value)
        self.input_tensor_names = set()
        self.number_of_inputs = len(model.inputs)
        for model_input in model.inputs:
            self.input_tensor_names.update(model_input.get_names())

    def _check_input_data_format(
        self, input_data: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray], Dict[str, np.ndarray]]
    ) -> None:
        """
        Checks correspondence of the model input names and the passed data.
        If there is a mismatch, the method throws a more specific and readable error than
        original error raised by the compiled model.

        :param input_data: Provided inputs to infer the model.
        """
        actual_num_inputs = 1 if isinstance(input_data, np.ndarray) else len(input_data)
        if actual_num_inputs != self.number_of_inputs:
            raise RuntimeError(f"Model expects {self.number_of_inputs} inputs, but {actual_num_inputs} are provided.")
        if isinstance(input_data, dict):
            for name in input_data:
                if isinstance(name, str) and name not in self.input_tensor_names:
                    raise RuntimeError(f"Missing a required input: {name} to run the model.")

    def infer(
        self, input_data: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray], Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Runs model on the provided input via OpenVINO Runtime.
        Returns the dictionary of model outputs by node names.

        :param input_data: Inputs for the model.
        :return output_data: Model's output.
        """
        self._check_input_data_format(input_data)
        model_outputs = self.compiled_model(input_data)

        output_data = {}
        for tensor, value in model_outputs.items():
            for tensor_name in tensor.get_names():
                output_data[tensor_name] = value
        return output_data
