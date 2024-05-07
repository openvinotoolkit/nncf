# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import openvino.runtime as ov

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.data.dataset import Dataset
from nncf.experimental.tensor.tensor import Tensor
from nncf.openvino.engine import OVNativeEngine
from nncf.openvino.graph.model_transformer import OVModelTransformer
from nncf.openvino.graph.node_utils import get_parameter_node_name
from nncf.openvino.graph.node_utils import get_result_node_name
from nncf.openvino.graph.transformations.commands import OVModelExtractionCommandV2
from nncf.quantization.algorithms.layerwise.iterator import LayerwiseIterator
from nncf.quantization.algorithms.layerwise.scheduler import LayerwiseStep
from nncf.quantization.algorithms.layerwise.scheduler import NodeOutputPort


class OVLayerwiseIterator(LayerwiseIterator):
    def __init__(
        self,
        model: ov.Model,
        graph: NNCFGraph,
        schedule: List[LayerwiseStep],
        dataset: Dataset,
        subset_size: int = 100,
        cache: Optional[Dict[NodeOutputPort, List[Tensor]]] = None,
    ):
        self._model = model
        self._graph = graph
        self._schedule = schedule
        self._dataset = dataset
        self._subset_size = subset_size
        self._cache = cache

        self._step_index = 0
        self._queue = []

        self._model_transformer = OVModelTransformer(model)
        self._cache_lifetime = {input_id: self.calculate_lifetime(input_id) for input_id in cache}
        self._model_input_ids = [(node.node_name, 0) for node in graph.get_input_nodes()]
        self._grap_vs_model_inputs_map = {
            input.node.get_friendly_name(): next(iter(input.names)) for input in model.inputs
        }

    def extract_model(self, input_ids: List[NodeOutputPort], output_ids: List[NodeOutputPort]) -> ov.Model:
        """
        Returns the backend-specific model that bounded by the specified input & output layers.

        :param inputs: List with inputs.
        :param outputs: List with the outputs.
        :return: Extracted backend-specific model.
        """
        transformation_layout = TransformationLayout()
        model_extraction_command = OVModelExtractionCommandV2(input_ids, output_ids)
        transformation_layout.register(model_extraction_command)
        extracted_model = self._model_transformer.transform(transformation_layout)
        return extracted_model

    def create_feed_dicts(self, input_ids: List[NodeOutputPort]) -> List[Dict]:
        """
        Creates the list of the dictionaries that contains the input data for the model execution.

        :param model: TModel instance.
        :param subgraph_data: A dictionary with the necessary data for current node.
        :param statistic_points: StatisticPointsContainer instance.
        :return: List of the dictionaries with the input data.
        """
        subset_size = self._subset_size
        for input_id in input_ids:
            subset_size = min(subset_size, len(self._cache[input_id]))

        feed_dicts = []
        for idx in range(subset_size):
            feed_dict = {}
            for input_id in input_ids:
                if input_id in self._model_input_ids:
                    input_name = self._grap_vs_model_inputs_map[input_id.node_name]
                else:
                    input_name = get_parameter_node_name(input_id.node_name, input_id.output_port)
                feed_dict[input_name] = self._cache[input_id][idx].data
            feed_dicts.append(feed_dict)
        return feed_dicts

    def run_model(self, model: ov.Model, feed_dicts: List, output_ids: Dict) -> None:
        """
        Updates the self._fp_inputs with the new statistics for the next layers
        after the correction of the bias for the current.

        :param model: Backend-specific subgraph.
        :param feed_dicts: List of dictionaries with the input data for the subgraph.
        :param subgraph_data: A dictionary with the needed list of the statistic nodes that will be updated.
        """
        outputs = defaultdict(list)
        engine = OVNativeEngine(model)
        for feed_dict in feed_dicts:
            new_output = engine.infer(feed_dict)
            for output_id in output_ids:
                output_name = get_result_node_name(output_id.node_name, output_id.output_port)
                outputs[output_id].append(Tensor(new_output[output_name]))
        return outputs

    def calculate_lifetime(self, input_id) -> int:
        for idx in range(len(self._schedule) - 1, self._step_index - 1, -1):
            if input_id in self._schedule[idx].subgraph_inputs:
                return idx
        return -1

    def update_cache(self, outputs: Dict[NodeOutputPort, List[Tensor]]) -> None:
        updated_cache = {}
        updated_cache_lifetime = {}
        for cached_input_id, value in self._cache.items():
            input_step_index = self._cache_lifetime[cached_input_id]
            if input_step_index > self._step_index:
                updated_cache[cached_input_id] = value
                updated_cache_lifetime[cached_input_id] = input_step_index

        for output_id, value in outputs.items():
            output_step_index = self.calculate_lifetime(output_id)
            if output_step_index > self._step_index:
                updated_cache[output_id] = value
                updated_cache_lifetime[output_id] = output_step_index

        self._cache = updated_cache
        self._cache_lifetime = updated_cache_lifetime

    def collect_output_tensors(self, step: LayerwiseStep) -> Dict[NodeOutputPort, List[Tensor]]:
        outputs = {}
        for output_id in step.subgraph_outputs:
            if output_id not in self._cache:
                break
            outputs[output_id] = self._cache[output_id]
        else:
            return outputs

        subgraph_model_input_ids = []
        for input_id in step.subgraph_inputs:
            if input_id in self._cache:
                continue
            if input_id in self._model_input_ids:
                subgraph_model_input_ids.append(input_id)
            else:
                raise RuntimeError(
                    f"{input.node_name}:{input.output_port} is not found in the input cache and is not a model input"
                )

        if subgraph_model_input_ids:
            subgraph_inputs = self._model_input_ids
            subgraph_outputs = step.subgraph_outputs
            for step in self._schedule[self._step_index + 1 :]:
                for input_id in step.subgraph_inputs:
                    if (
                        input_id in self._model_input_ids
                        and input_id not in self._cache
                        and input_id not in subgraph_outputs
                    ):
                        subgraph_outputs.append(input_id)
            feed_dicts = self._dataset.get_inference_data(range(self._subset_size))
        else:
            subgraph_inputs = step.subgraph_inputs
            subgraph_outputs = step.subgraph_outputs
            feed_dicts = self.create_feed_dicts(step.subgraph_inputs)
        extracted_model = self.extract_model(subgraph_inputs, subgraph_outputs)
        return self.run_model(extracted_model, feed_dicts, subgraph_outputs)

    def __next__(self) -> Tuple[NNCFNode, Dict[int, List[Tensor]]]:
        if not self._queue:
            if self._step_index >= len(self._schedule):
                raise StopIteration

            step = self._schedule[self._step_index]
            outputs = self.collect_output_tensors(step)

            self.update_cache(outputs)

            for target_node, target_inputs in step.target_node_map.items():
                inputs = {input_port_id: outputs[output_id] for input_port_id, output_id in target_inputs.items()}
                self._queue.append((target_node, inputs))
            self._step_index += 1

        return self._queue.pop(0)
