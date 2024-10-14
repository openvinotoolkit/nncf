# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import openvino.runtime as ov
from openvino.properties.hint import inference_precision

from nncf.common.engine import Engine
from nncf.openvino.graph.model_utils import model_has_state
from nncf.openvino.quantization.backend_parameters import BackendParameters


def get_compile_config(backend_params: Optional[Dict[str, Any]] = None):
    """
    Returns config to compile model.

    :params backend_params: Backend-specific params.
    :return: Config to compile model.
    """
    config = None
    if backend_params:
        inference_precision_hint = backend_params.get(BackendParameters.INFERENCE_PRECISION_HINT, None)
        if inference_precision_hint:
            config = {inference_precision: inference_precision_hint}
    return config


class OVCompiledModelEngine(Engine):
    """
    Implementation of the engine to infer OpenVINO compiled model.

    OVCompiledModelEngine uses
    [OpenVINO Runtime](https://docs.openvino.ai/latest/openvino_docs_OV_UG_OV_Runtime_User_Guide.html)
    to infer the compiled model.
    """

    def __init__(self, compiled_model: ov.CompiledModel, stateful: bool):
        self.infer_request = compiled_model.create_infer_request()
        self.reset_state = stateful and hasattr(self.infer_request, "reset_state")

    def infer(
        self, input_data: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray], Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Runs model on the provided input via OpenVINO Runtime.
        Returns the dictionary of model outputs by node names.

        :param input_data: Inputs for the model.
        :return output_data: Model's output.
        """
        if self.reset_state:
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

    def __init__(self, model: ov.Model, backend_params: Optional[Dict[str, Any]] = None):
        ie = ov.Core()
        stateful = model_has_state(model)
        config = get_compile_config(backend_params)
        compiled_model = ie.compile_model(model, device_name="CPU", config=config)
        self.engine = OVCompiledModelEngine(compiled_model, stateful)

    def infer(
        self, input_data: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray], Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Runs model on the provided input via OpenVINO Runtime.
        Returns the dictionary of model outputs by node names.

        :param input_data: Inputs for the model.
        :return output_data: Model's output.
        """
        return self.engine.infer(input_data)
