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

from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
import onnx

from nncf.experimental.post_training.statistics.aggregator import StatisticsAggregator

from nncf.experimental.onnx.samplers import create_onnx_sampler
from nncf.experimental.onnx.engine import ONNXEngine
from nncf.experimental.post_training.api.dataset import Dataset


class ONNXStatisticsAggregator(StatisticsAggregator):
    # TODO (Nikita Malinin): Remove ONNXStatisticsAggregator & create the common backend-agnostic solution

    def __init__(self, engine: ONNXEngine, dataset: Dataset):
        super().__init__(engine, dataset)

    def collect_statistics(self, model: onnx.ModelProto) -> None:
        # TODO (Nikita Malinin): Need to update adding output process with the backend-specific graph transformer
        layers_to_collect_statistics = list(self.layers_statistics.keys())
        model_outputs = []
        for output in list(model.graph.output):
            model_outputs.append(output.name)
        model_with_intermediate_outputs = select_model_inputs_outputs(model,
                                                                      outputs=[*layers_to_collect_statistics,
                                                                               *model_outputs])
        max_number_samples = 0
        for _, v in self.layers_statistics.items():
            max_number_samples = max(max_number_samples, v.num_samples)

        sampler = create_onnx_sampler(self.dataset, range(max_number_samples))

        self.engine.set_model(model_with_intermediate_outputs)
        self.engine.set_sampler(sampler)
        self.engine.compute_statistics(self.layers_statistics)
