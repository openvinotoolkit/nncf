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

import pytest

from nncf.quantization.algorithms.layerwise.scheduler import LayerwiseScheduler
from tests.cross_fw.test_templates.models import NNCFGraphCAWithBias


@pytest.mark.parametrize("add_additional_outputs", [False, True])
@pytest.mark.parametrize("collect_inputs", [False, True])
def test_remove_nodes_and_reconnect_graph(add_additional_outputs, collect_inputs):
    conv_metatype = "CONV_METATYPE"
    add_metatype = "ADD_METATYPE"

    nncf_graph = NNCFGraphCAWithBias(conv_metatype, add_metatype).nncf_graph

    target_nodes = [node for node in nncf_graph.get_all_nodes() if node.metatype == conv_metatype]

    layerwise_scheduler = LayerwiseScheduler(add_additional_outputs)
    schedule = layerwise_scheduler.schedule(nncf_graph, target_nodes, collect_inputs)
    for step in schedule:
        assert len(step.target_node_map) == 1
        for node, io_id in step.target_node_map.items():
            assert node in target_nodes
            assert len(io_id) == 1
            if collect_inputs:
                for io_port, node_output_port in io_id.items():
                    assert io_port == 0
                    assert node_output_port.node_name == nncf_graph.get_previous_nodes(node)[0].node_name
                    assert node_output_port.output_port == 0
            else:
                for io_port, node_output_port in io_id.items():
                    assert io_port == 0
                    assert node_output_port.node_name == node.node_name
                    assert node_output_port.output_port == 0
