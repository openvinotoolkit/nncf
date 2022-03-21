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

from nncf.experimental.post_training.api.engine import Engine

import onnxruntime as rt
import numpy as np


class ONNXEngine(Engine):
    """
    Engine for ONNX backend using ONNXRuntime to infer the model.
    """

    def __init__(self, **rt_session_options):
        super().__init__()
        self.sess = None
        self.rt_session_options = rt_session_options
        if 'providers' not in self.rt_session_options:
            self.rt_session_options['providers'] = ['OpenVINOExecutionProvider']

    def set_model(self, model: str) -> None:
        """
        Creates ONNXRuntime InferenceSession for the onnx model with the location at 'model'.
        """
        super().set_model(model)
        self.sess = rt.InferenceSession(model, **self.rt_session_options)

    def _infer(self, _input: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Runs InferenceSession on the provided '_input'.
        Returns the model outputs and corresponding node names in the model.
        """
        output = {}
        input_name = self.sess.get_inputs()[0].name
        output_tensor = self.sess.run([], {input_name: _input.astype(np.float32)})
        model_outputs = self.sess.get_outputs()
        for i, model_output in enumerate(model_outputs):
            output[model_output.name] = output_tensor[i]
        return output
