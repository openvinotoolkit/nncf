"""
 Copyright (c) 2023 Intel Corporation
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
import onnxruntime as rt

from nncf.common.engine import Engine


class ONNXEngine(Engine):
    """
    Engine for ONNX backend using ONNXRuntime to infer the model.
    """

    def __init__(self, model, **rt_session_options):
        self.input_names = set()
        rt_session_options['providers'] = ['CPUExecutionProvider']
        serialized_model = model.SerializeToString()
        self.sess = rt.InferenceSession(serialized_model, **rt_session_options)

        for inp in self.sess.get_inputs():
            self.input_names.add(inp.name)

    def infer(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Runs model on the provided input via ONNXRuntime InferenceSession.
        Returns the dictionary of model outputs by node names.
        :param input_data: inputs for the model
        :return output_data: models outputs
        """
        output_tensors = self.sess.run([], {k: v for k, v in input_data.items() if k in self.input_names})
        model_outputs = self.sess.get_outputs()

        return {output.name: tensor for tensor, output in zip(output_tensors, model_outputs)}
