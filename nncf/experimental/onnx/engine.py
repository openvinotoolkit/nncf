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

from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.onnx.samplers import create_onnx_sampler

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

    def transform_input(self, inputs):
        return inputs.astype(np.float32)

    def compute_statistics(self, statistics_layout) -> Dict[str, List[np.ndarray]]:
        if not self.is_model_set():
            raise RuntimeError('The {} tried to compute statistics, '
                               'while the model was not set.'.format(self.__class__))

        sampler = self.sampler if self.sampler else create_onnx_sampler(self.data_loader)
        output = {}
        for sample in sampler:
            _input, _ = sample
            input_name = self.sess.get_inputs()[0].name
            output_tensor = self.sess.run([], {input_name: self.transform_input(_input)})
            model_outputs = self.sess.get_outputs()
            for out_id, model_output in enumerate(model_outputs):
                if model_output.name not in output:
                    output[model_output.name] = []
                # TODO (Nikita Malinin): Add backend-specific statistics aggregator usage
                output[model_output.name].append(output_tensor[out_id])
        return output

    def compute_metrics(self, metrics_per_sample=False):
        if not self.is_model_set():
            raise RuntimeError('The {} tried to compute statistics, '
                               'while the model was not set.'.format(self.__class__))

        # TODO (Nikita Malinin): Add per-sample metrics calculation
        sampler = self.sampler if self.sampler else create_onnx_sampler(self.data_loader, range(len(self.data_loader)))
        for sample in sampler:
            input_data, target  = sample
            input_name = self.sess.get_inputs()[0].name
            output_tensors = self.sess.run([], {input_name: self.transform_input(input_data)})
            self.metrics.update(output_tensors, target)
        return self.metrics.avg_value
