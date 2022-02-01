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

from nncf.experimental.post_training.initialization.quantizer_range_finder import QuantizerRangeFinderAlgorithm
from nncf.experimental.post_training.compressed_model import CompressedModel

from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
import onnx
import onnxruntime as rt
import tempfile
import numpy as np
import math


class StatisticsCollector:
    def __init__(self):
        self.tensors = []
        self.global_min = []
        self.global_max = []
        self.min_sum = 0
        self.max_sum = 0
        self.min_avg = 0
        self.max_avg = 0
        self.counter = 0

    def update(self, tensors):
        if len(self.global_min) == 0:
            self.global_min = [math.inf] * len(tensors)
            self.global_max = [-math.inf] * len(tensors)
        for i, tensor in enumerate(tensors):
            np_tensor = np.array(tensor)
            min_val = np.min(np_tensor)
            max_val = np.max(np_tensor)
            self.global_min[i] = min(self.global_min[i], min_val)
            self.global_max[i] = max(self.global_max[i], max_val)

        # min_val = np.min(tensor)
        # max_val = np.max(tensor)
        # self.min_sum += min_val
        # self.max_sum += max_val
        # self.counter += 1
        # self.min_avg = self.min_sum / self.counter
        # self.max_avg = self.max_sum / self.counter
        # self.global_min = min(self.global_min, min_val)
        # self.global_max = max(self.global_max, max_val)


class ONNXQuantizerRangeFinderAlgorithm(QuantizerRangeFinderAlgorithm):
    """
    The base class for all post-training quantization initialization algorithms.
    """

    def __init__(self, dataloader, engine, **kwargs):
        super().__init__(engine, dataloader)
        self.model_transformer = kwargs['model_transformer']

    def apply(self, model: CompressedModel):
        num_iters = 3
        statistics_collector = StatisticsCollector()
        onnx_model = model.original_model
        activations_outputs = []
        for transformation in model.transformations:
            if not transformation.is_weights:
                activations_outputs.append(transformation.target_point)

        model_output = list(enumerate_model_node_outputs(onnx_model))[-1]
        model_with_intermediate_outputs = select_model_inputs_outputs(onnx_model,
                                                                      outputs=[*activations_outputs, model_output])
        with tempfile.NamedTemporaryFile() as temporary_model:
            onnx.save(model_with_intermediate_outputs, temporary_model.name)
            self.engine.set_model(temporary_model.name)
            for i, (input_, *other) in enumerate(self.dataloader):
                if i == num_iters:
                    break
                outputs = self.engine.infer_model(input_)
                statistics_collector.update(outputs)


        self.model_transformer._apply_transformation()
