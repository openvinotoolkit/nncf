# Copyright (c) 2025 Intel Corporation
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
from itertools import islice
from typing import Dict, List, Optional, Tuple

import openvino.runtime as ov

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.data.dataset import Dataset
from nncf.openvino.engine import OVNativeEngine
from nncf.openvino.graph.model_transformer import OVModelTransformer
from nncf.openvino.graph.node_utils import get_parameter_node_name
from nncf.openvino.graph.node_utils import get_result_node_name
from nncf.openvino.graph.transformations.commands import OVStateLessModelExtractionCommand
from nncf.quantization.algorithms.layerwise.iterator import LayerwiseIterator
from nncf.quantization.algorithms.layerwise.scheduler import LayerwiseStep
from nncf.quantization.algorithms.layerwise.scheduler import NodeOutputPort
from nncf.tensor import Tensor


class OVLayerwiseIterator(LayerwiseIterator):
    """
    OVLayerwiseIterator is a class for iterating through layers of an OpenVINO model in a layer-wise manner.
    """

    def __init__(
        self,
        model: ov.Model,
        graph: NNCFGraph,
        schedule: List[LayerwiseStep],
        dataset: Dataset,
        subset_size: int = 100,
        cache: Optional[Dict[NodeOutputPort, List[Tensor]]] = None,
    ):
        """
        :param model: The OpenVINO model to iterate over.
        :param graph: The NNCF graph representation of the model.
        :param schedule: The schedule of steps for layer-wise iteration.
        :param dataset: The dataset to use for model inference.
        :param subset_size: The size of the subset of the dataset to use for each iteration.
        :param cache: The cache for storing tensors.
        """
        self._model = model
        self._graph = graph
        self._schedule = schedule
        self._dataset = dataset
        self._subset_size = subset_size
        self._cache: Dict[NodeOutputPort, List[Tensor]] = {}
        self._cache_lifetime: Dict[NodeOutputPort, int] = {}

        self._step_index = 0
        self._queue: List[Tuple[NNCFNode, Dict[int, List[Tensor]]]] = []

        if cache is not None:
            self.update_cache(cache)

        self._model_transformer = OVModelTransformer(model)
        self._model_input_ids = [NodeOutputPort(node.node_name, 0) for node in graph.get_input_nodes()]
        self._graph_vs_model_inputs_map = {
            input.node.get_friendly_name(): next(iter(input.names)) for input in model.inputs
        }

    def extract_model(self, input_ids: List[NodeOutputPort], output_ids: List[NodeOutputPort]) -> ov.Model:
        """
        Extracts submodel by the specified input & output layers.

        :param input_ids: List of input node IDs.
        :param output_ids: List of output node IDs.
        :return: Extracted OpenVINO model.
        """
        transformation_layout = TransformationLayout()
        model_extraction_command = OVStateLessModelExtractionCommand(input_ids, output_ids)
        transformation_layout.register(model_extraction_command)
        extracted_model = self._model_transformer.transform(transformation_layout)
        return extracted_model

    def create_feed_dicts(self, input_ids: List[NodeOutputPort]) -> List[Dict]:
        """
        Creates a list of dictionaries containing the input data for model execution.

        :param input_ids: List of input node IDs.
        :return: List of dictionaries with the input data.
        """
        subset_size = self._subset_size
        for input_id in input_ids:
            subset_size = min(subset_size, len(self._cache[input_id]))

        feed_dicts = []
        for idx in range(subset_size):
            feed_dict = {}
            for input_id in input_ids:
                if input_id in self._model_input_ids:
                    input_name = self._graph_vs_model_inputs_map[input_id.node_name]
                else:
                    input_name = get_parameter_node_name(input_id.node_name, input_id.output_port)
                feed_dict[input_name] = self._cache[input_id][idx].data
            feed_dicts.append(feed_dict)
        return feed_dicts

    def run_model(
        self, model: ov.Model, feed_dicts: List[Dict], output_ids: List[NodeOutputPort]
    ) -> Dict[NodeOutputPort, List[Tensor]]:
        """
        Runs the submodel with the given input data and collects the output tensors.

        :param model: The OpenVINO submodel to run.
        :param feed_dicts: List of dictionaries containing the input data for the submodel.
        :param output_ids: List of output node IDs to collect outputs from.
        :return: Dictionary of output node IDs to their corresponding tensors.
        """
        outputs = defaultdict(list)
        engine = OVNativeEngine(model)
        for feed_dict in feed_dicts:
            new_output = engine.infer(feed_dict)
            for output_id in output_ids:
                output_name = get_result_node_name(output_id.node_name, output_id.output_port)
                outputs[output_id].append(Tensor(new_output[output_name]))
        return outputs

    def calculate_lifetime(self, input_id: NodeOutputPort) -> int:
        """
        Calculates the lifetime of a tensor in the cache.

        :param input_id: The input node ID.
        :return: The step index at which the tensor is last used.
        """
        for idx in range(len(self._schedule) - 1, self._step_index - 1, -1):
            if input_id in self._schedule[idx].subgraph_inputs:
                return idx
        return -1

    def update_cache(self, outputs: Dict[NodeOutputPort, List[Tensor]]) -> None:
        """
        Updates the cache with new output tensors and removes expired ones.

        :param outputs: Dictionary of output node IDs to their corresponding tensors.
        """
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
        """
        Collects the output tensors for a given step.

        :param step: The current layer-wise step.
        :return: Dictionary of output node IDs to their corresponding tensors.
        """
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
            subgraph_outputs = [output_id for output_id in step.subgraph_outputs]
            for step in self._schedule[self._step_index + 1 :]:
                for input_id in step.subgraph_inputs:
                    if (
                        input_id in self._model_input_ids
                        and input_id not in self._cache
                        and input_id not in subgraph_outputs
                    ):
                        subgraph_outputs.append(input_id)
            feed_dicts = islice(self._dataset.get_inference_data(), self._subset_size)
        else:
            subgraph_inputs = step.subgraph_inputs
            subgraph_outputs = step.subgraph_outputs
            feed_dicts = self.create_feed_dicts(step.subgraph_inputs)
        extracted_model = self.extract_model(subgraph_inputs, subgraph_outputs)
        return self.run_model(extracted_model, feed_dicts, subgraph_outputs)

    def __next__(self) -> Tuple[NNCFNode, Dict[int, List[Tensor]]]:
        """
        Advances to the next layer in the iteration.

        :return: A tuple containing the target node and a dictionary of input port IDs to tensors.
        :raises StopIteration: If the iteration has reached the end of the schedule.
        """
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
