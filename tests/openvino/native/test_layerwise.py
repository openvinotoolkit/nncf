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

from unittest.mock import MagicMock

import openvino as ov
import torch

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.data.dataset import Dataset
from nncf.quantization.algorithms.layerwise.openvino_iterator import OVLayerwiseIterator
from nncf.quantization.algorithms.layerwise.scheduler import LayerwiseStep
from nncf.quantization.algorithms.layerwise.scheduler import NodeOutputPort
from nncf.tensor import Tensor
from tests.cross_fw.test_templates.test_layerwise import TemplateTestLayerwiseEngine


class TestOVLayerwiseEngine(TemplateTestLayerwiseEngine):
    @staticmethod
    def backend_specific_model(model: bool, tmp_dir: str):
        ov_model = ov.convert_model(model, example_input=torch.rand(model.INPUT_SIZE))
        return ov_model

    @staticmethod
    def get_transform_fn():
        def transform_fn(data_item):
            tensor, _ = data_item
            return tensor

        return transform_fn


class TestOVLayerwiseIterator:
    def setup_mocks(self, schedule):
        self.subset_size = 10
        self.iterator = OVLayerwiseIterator(
            model=MagicMock(spec=ov.Model),
            graph=MagicMock(spec=NNCFGraph),
            schedule=schedule,
            dataset=MagicMock(spec=Dataset),
            subset_size=self.subset_size,
        )

    def test_create_feed_dicts(self):
        schedule = [
            LayerwiseStep(
                target_node_map={MagicMock(spec=NNCFNode): {0: NodeOutputPort("node1", 0)}},
                subgraph_inputs=[NodeOutputPort("input1", 0)],
                subgraph_outputs=[NodeOutputPort("node1", 0)],
            ),
            LayerwiseStep(
                target_node_map={MagicMock(spec=NNCFNode): {0: NodeOutputPort("node2", 0)}},
                subgraph_inputs=[NodeOutputPort("node1", 0)],
                subgraph_outputs=[NodeOutputPort("node2", 0)],
            ),
        ]
        self.setup_mocks(schedule)

        input_ids = [NodeOutputPort("input1", 0)]
        self.iterator._cache = {NodeOutputPort("input1", 0): [Tensor(i) for i in range(self.subset_size)]}
        self.iterator._model_input_ids = [NodeOutputPort("input1", 0)]
        self.iterator._graph_vs_model_inputs_map = {"input1": "model_input1"}

        feed_dicts = self.iterator.create_feed_dicts(input_ids)
        assert len(feed_dicts) == self.subset_size
        for i, feed_dict in enumerate(feed_dicts):
            assert feed_dict["model_input1"] == i

    def test_update_cache(self):
        schedule = [
            LayerwiseStep(
                target_node_map={MagicMock(spec=NNCFNode): {0: NodeOutputPort("node1", 0)}},
                subgraph_inputs=[NodeOutputPort("input1", 0)],
                subgraph_outputs=[NodeOutputPort("node1", 0)],
            ),
            LayerwiseStep(
                target_node_map={MagicMock(spec=NNCFNode): {0: NodeOutputPort("node2", 0)}},
                subgraph_inputs=[NodeOutputPort("node1", 0)],
                subgraph_outputs=[NodeOutputPort("node2", 0)],
            ),
        ]
        self.setup_mocks(schedule)

        outputs = [
            {NodeOutputPort("input1", 0): [Tensor(0)], NodeOutputPort("node1", 0): [Tensor(1)]},
            {NodeOutputPort("node2", 0): [Tensor(2)]},
        ]
        self.iterator._cache = {}
        self.iterator._cache_lifetime = {}
        self.iterator._step_index = 0

        for step_index, output in enumerate(outputs):
            self.iterator._step_index = step_index
            self.iterator.update_cache(output)
            if step_index == 0:
                assert len(self.iterator._cache) == 1
                assert self.iterator._cache[NodeOutputPort("node1", 0)][0].data == 1
                assert self.iterator._cache_lifetime[NodeOutputPort("node1", 0)] == 1
        assert not self.iterator._cache

    def test_collect_output_tensors(self):
        schedule = [
            LayerwiseStep(
                target_node_map={MagicMock(spec=NNCFNode): {0: NodeOutputPort("node1", 0)}},
                subgraph_inputs=[NodeOutputPort("input1", 0)],
                subgraph_outputs=[NodeOutputPort("node1", 0)],
            ),
            LayerwiseStep(
                target_node_map={MagicMock(spec=NNCFNode): {0: NodeOutputPort("node2", 0)}},
                subgraph_inputs=[NodeOutputPort("node1", 0), NodeOutputPort("input2", 0)],
                subgraph_outputs=[NodeOutputPort("node2", 0)],
            ),
        ]
        self.setup_mocks(schedule)

        self.iterator._model_input_ids = [NodeOutputPort("input1", 0), NodeOutputPort("input2", 0)]
        self.iterator._graph_vs_model_inputs_map = {"input1": "model_input1", "input2": "model_input2"}
        self.iterator.extract_model = MagicMock()
        self.iterator.run_model = MagicMock()

        # empty cache
        step = schedule[0]
        self.iterator.collect_output_tensors(step)
        assert self.iterator.extract_model.call_args[0][0] == self.iterator._model_input_ids
        assert self.iterator.extract_model.call_args[0][1] == schedule[1].subgraph_inputs

        # partial cache
        self.iterator._cache = {NodeOutputPort("input2", 0): [Tensor(1)]}
        self.iterator.collect_output_tensors(step)
        assert self.iterator.extract_model.call_args[0][0] == self.iterator._model_input_ids
        assert self.iterator.extract_model.call_args[0][1] == [NodeOutputPort("node1", 0)]

        # full cache
        self.iterator._cache = {NodeOutputPort("node1", 0): [Tensor(1)]}
        outputs = self.iterator.collect_output_tensors(step)
        assert self.iterator.extract_model.call_count == 2
        assert len(outputs) == 1
        assert outputs[NodeOutputPort("node1", 0)][0].data == 1

    def test_order(self):
        schedule = [
            LayerwiseStep(
                target_node_map={MagicMock(spec=NNCFNode): {0: NodeOutputPort("node1", 0)}},
                subgraph_inputs=[NodeOutputPort("input1", 0)],
                subgraph_outputs=[NodeOutputPort("node1", 0)],
            ),
            LayerwiseStep(
                target_node_map={MagicMock(spec=NNCFNode): {0: NodeOutputPort("node2", 0)}},
                subgraph_inputs=[NodeOutputPort("node1", 0)],
                subgraph_outputs=[NodeOutputPort("node2", 0)],
            ),
        ]
        self.setup_mocks(schedule)

        self.iterator._cache = {NodeOutputPort("node1", 0): [Tensor(1)], NodeOutputPort("node2", 0): [Tensor(1)]}
        self.iterator._cache_lifetime = {NodeOutputPort("node1", 0): 1, NodeOutputPort("node2", 0): 1}
        for idx, target_node_inputs in enumerate(self.iterator):
            assert target_node_inputs[0] in schedule[idx].target_node_map
