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

from typing import Dict

import numpy as np
import openvino.runtime as ov

from nncf.common.engine import Engine
from nncf.experimental.openvino_native.tensor import OVNNCFTensor


class OVNativeEngine(Engine):
    """
    Implementation of the engine for OpenVINO_NATIVE backend.

    OVNativeEngine uses
    [OpenVINO Runtime](https://docs.openvino.ai/latest/openvino_docs_OV_UG_OV_Runtime_User_Guide.html)
    to infer the model.
    """

    def __init__(self, model, target_device='CPU'):
        self.input_names = set()

        ie = ov.Core()
        self.compiled_model = ie.compile_model(model=model, device_name=target_device)
        for inp in model.get_parameters():
            self.input_names.add(inp.get_friendly_name())

    def infer(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Runs model on the provided input via OpenVINO Runtime.
        Returns the dictionary of model outputs by node names.
        :param input_data: inputs for the model.
        :return output_data: models outputs.
        """
        model_outputs = self.compiled_model(
            {k: v.tensor for k, v in input_data.items() if k in self.input_names})

        return {
            out.get_node().get_friendly_name(): OVNNCFTensor(data)
            for out, data in model_outputs.items()
        }
