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

from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
import onnx
# pylint: disable=no-member
import tempfile

from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase

from nncf.experimental.post_training.statistics.aggregator import StatisticsAggregator

from nncf.experimental.onnx.samplers import create_onnx_sampler
from nncf.experimental.onnx.engine import ONNXEngine
from nncf.experimental.post_training.api.dataloader import DataLoader


class ONNXStatisticsAggregator(StatisticsAggregator):
    def __init__(self, engine: ONNXEngine, dataloader: DataLoader):
        super().__init__(engine, dataloader)

    def collect_statistics(self, model: onnx.ModelProto) -> None:
        layers_to_collect_statistics = list(self.layers_statistics.keys())
        model_output = list(enumerate_model_node_outputs(model))[-1]
        model_with_intermediate_outputs = select_model_inputs_outputs(model,
                                                                      outputs=[*layers_to_collect_statistics,
                                                                               model_output])
        max_number_samples = 0
        for _, v in self.layers_statistics.items():
            max_number_samples = max(max_number_samples, v.num_samples)

        with tempfile.NamedTemporaryFile() as temporary_model:
            onnx.save(model_with_intermediate_outputs, temporary_model.name)
            self.engine.set_model(temporary_model.name)
            sampler = create_onnx_sampler(self.dataloader)
            for i, sample in enumerate(sampler):
                if i == max_number_samples:
                    break
                # Currently, there is no an usage of target
                _input, _ = sample
                output = self.engine.infer(_input)
                self._agregate_statistics(output, self.layers_statistics)

    def _agregate_statistics(self, output, layers_statistics: Dict[str, TensorStatisticCollectorBase]):
        for k, v in layers_statistics.items():
            tensor = output[k]
            v.register_input(tensor)
