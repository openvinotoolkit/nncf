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

from nncf.experimental.post_training.api.engine import Engine

import onnx
import onnxruntime as rt
import numpy as np


class ONNXEngine(Engine):
    def __init__(self, providers: List[str] = None):
        if providers is None:
            self.providers = ['OpenVINOExecutionProvider']
        else:
            self.providers = providers

    def set_model(self, model: str) -> None:
        """
        Because ORT must load model from the file, we should provide the path.
        """
        onnx_model = onnx.load(model)
        # onnx.checker.check_model(onnx_model)
        self.model = model
        self.sess = rt.InferenceSession(self.model, providers=self.providers)

    def infer_model(self, input_: np.ndarray) -> List[np.ndarray]:
        # feed Dict TF
        input_name = self.sess.get_inputs()[0].name
        input_tensor = input_.cpu().detach().numpy()
        output_tensor = self.sess.run([], {input_name: input_tensor.astype(np.float32)})
        return output_tensor
