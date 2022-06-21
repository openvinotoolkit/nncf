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
from typing import Tuple

from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.onnx.samplers import create_onnx_sampler
from nncf.experimental.post_training.api.sampler import Sampler

import onnxruntime as rt
import numpy as np
import onnx
import tempfile


class ONNXEngine(Engine):
    """
    Engine for ONNX backend using ONNXRuntime to infer the model.
    """

    def __init__(self, **rt_session_options):
        super().__init__()
        self._inputs_transforms = lambda input_data: input_data.astype(np.float32)
        self.sess = None
        self.rt_session_options = rt_session_options

        # TODO: Do not force it to use CPUExecutionProvider
        # OpenVINOExecutionProvider raises the following error.
        # onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1
        # : FAIL : This is an invalid model. Error: Duplicate definition of name (data).
        self.rt_session_options['providers'] = ['CPUExecutionProvider']

    def get_sampler(self) -> Sampler:
        # TODO (Nikita Malinin): Replace range calling with the max length variable
        return self.sampler if self.sampler else create_onnx_sampler(self.dataset, range(len(self.dataset)))

    def set_model(self, model: str) -> None:
        """
        Creates ONNXRuntime InferenceSession for the onnx model with the location at 'model'.

        :param model: onnx.ModelProto model instance
        """
        super().set_model(model)
        with tempfile.NamedTemporaryFile() as temporary_model:
            onnx.save(model, temporary_model.name)
            self.sess = rt.InferenceSession(temporary_model.name, **self.rt_session_options)

    def infer(self, input_data: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
        """
        Runs model on the provided input_data via ONNXRuntime InferenceSession.
        Returns the dictionary of model outputs by node names.

        :param input_data: inputs for the model transformed with the inputs_transforms
        :return output_data: models output after outputs_transforms
        """
        # TODO (Nikita Malinin): Need to change input_data to Dict values by input names
        input_name = self.sess.get_inputs()[0].name
        output_tensors = self.sess.run([], {input_name: self._inputs_transforms(input_data)})
        model_outputs = self.sess.get_outputs()
        output_tensors = self._outputs_transforms(output_tensors)
        return output_tensors, model_outputs
