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

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.experimental.post_training.api.dataset import Dataset
from nncf.experimental.post_training.statistics.aggregator import StatisticsAggregator
from nncf.experimental.onnx.samplers import create_onnx_sampler
from nncf.experimental.onnx.engine import ONNXEngine
from nncf.experimental.onnx.graph.onnx_graph import ONNXGraph


class ONNXStatisticsAggregator(StatisticsAggregator):
    # TODO (Nikita Malinin): Remove ONNXStatisticsAggregator & create the common backend-agnostic solution

    def __init__(self, engine: ONNXEngine, dataset: Dataset):
        super().__init__(engine, dataset)

    def collect_statistics(self, model: onnx.ModelProto) -> None:
        # TODO (Nikita Malinin): Need to update adding output process with the backend-specific graph transformer
        onnx_graph = ONNXGraph(model)
        model_outputs = [output.name for output in onnx_graph.get_model_outputs()]
        extra_model_outputs = []
        for node_name, statistic_points in self.statistic_points.items():
            for statistic_point in statistic_points:
                if NNCFGraphNodeType.INPUT_NODE in statistic_point.target_point.target_node_name:
                    edge_name = onnx_graph.get_model_inputs()
                else:
                    if statistic_point.target_point.type == TargetType.POST_LAYER_OPERATION:
                        edge_name = onnx_graph.get_node_edges(node_name)['output'][0]
                    elif statistic_point.target_point.type == TargetType.PRE_LAYER_OPERATION:
                        edge_name = onnx_graph.get_node_edges(node_name)['input'][0]
                    else:
                        raise RuntimeError
                extra_model_outputs.append(edge_name)

        model_with_intermediate_outputs = select_model_inputs_outputs(model,
                                                                      outputs=[*extra_model_outputs,
                                                                               *model_outputs])

        sampler = create_onnx_sampler(self.dataset, self.max_number_samples)

        self.engine.set_model(model_with_intermediate_outputs)
        self.engine.set_sampler(sampler)
        self.engine.compute_statistics(self.statistic_points)
