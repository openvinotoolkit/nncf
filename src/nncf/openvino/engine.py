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

from typing import Optional, Union

import numpy as np
import openvino as ov
from openvino import Type
from openvino.properties.hint import inference_precision

from nncf.common.engine import Engine
from nncf.openvino.graph.model_utils import model_has_state

RESET_STATE_KEY = "reset_state"


class OVCompiledModelEngine(Engine):
    """
    Implementation of the engine to infer OpenVINO compiled model.

    OVCompiledModelEngine uses
    [OpenVINO Runtime](https://docs.openvino.ai/latest/openvino_docs_OV_UG_OV_Runtime_User_Guide.html)
    to infer the compiled model.
    """

    def __init__(self, compiled_model: ov.CompiledModel, reset_state: bool):
        self.infer_request = compiled_model.create_infer_request()
        if reset_state and not hasattr(self.infer_request, "reset_state"):
            msg = "The model is not stateful, but reset_state=True was passed."
            raise ValueError(msg)
        self.reset_state = reset_state

    def infer(
        self, input_data: Union[np.ndarray, list[np.ndarray], tuple[np.ndarray], dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        """
        Runs model on the provided input via OpenVINO Runtime.
        Returns the dictionary of model outputs by node names.

        :param input_data: Inputs for the model.
        :return output_data: Model's output.
        """
        if isinstance(input_data, dict) and input_data.get(RESET_STATE_KEY, False):
            self.infer_request.reset_state()
            input_data = input_data.copy()
            del input_data[RESET_STATE_KEY]
        elif self.reset_state:
            self.infer_request.reset_state()

        model_outputs = self.infer_request.infer(input_data, share_inputs=True)

        output_data = {}
        for tensor, value in model_outputs.items():
            for tensor_name in tensor.get_names():
                output_data[tensor_name] = value
        return output_data


class OVNativeEngine(Engine):
    """
    Implementation of the engine for OpenVINO backend.

    OVNativeEngine uses
    [OpenVINO Runtime](https://docs.openvino.ai/latest/openvino_docs_OV_UG_OV_Runtime_User_Guide.html)
    to infer the model.
    """

    def __init__(self, model: ov.Model, use_fp32_precision: bool = True, reset_state: Optional[bool] = None):
        """
        :param model: Model.
        :param use_fp32_precision: A flag that determines whether to force the engine to use FP32
            precision during inference.
        """
        config = None
        if use_fp32_precision:
            config = {inference_precision: Type.f32}
        ie = ov.Core()
        stateful = model_has_state(model)
        if reset_state and not stateful:
            msg = "The model is not stateful, but reset_state=True was passed."
            raise ValueError(msg)
        if reset_state is None:
            reset_state = stateful
        compiled_model = ie.compile_model(model, device_name="CPU", config=config)
        self.engine = OVCompiledModelEngine(compiled_model, reset_state)

    def infer(
        self, input_data: Union[np.ndarray, list[np.ndarray], tuple[np.ndarray], dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        """
        Runs model on the provided input via OpenVINO Runtime.
        Returns the dictionary of model outputs by node names.

        :param input_data: Inputs for the model.
        :return output_data: Model's output.
        """
        return self.engine.infer(input_data)
