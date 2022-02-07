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
from typing import List
from typing import Tuple

from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.api.dataloader import DataLoader
from nncf.experimental.post_training.sampler import Sampler
from nncf.experimental.post_training.sampler import RandomSampler

import onnx
import onnxruntime as rt
import numpy as np


class ONNXEngine(Engine):
    def __init__(self, dataloader: DataLoader, sampler: Sampler = None, providers: List[str] = None):
        super().__init__(dataloader)
        if sampler is None:
            # self.sampler = RandomSampler()
            self.sampler = None
        if providers is None:
            self.providers = ['OpenVINOExecutionProvider']
        else:
            self.providers = providers

    def set_model(self, model: str) -> None:
        """
        Because ORT must load model from the file, we should provide the path.
        """
        self.model = model
        self.sess = rt.InferenceSession(self.model, providers=self.providers)

    def infer(self, i: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        # feed Dict TF
        output = {}
        _input, target = self.dataloader[i]
        input_name = self.sess.get_inputs()[0].name
        input_tensor = _input.cpu().detach().numpy()
        output_tensor = self.sess.run([], {input_name: input_tensor.astype(np.float32)})
        model_outputs = self.sess.get_outputs()
        for i, model_output in enumerate(model_outputs):
            output[model_output.name] = output_tensor[i]
        return output, target
