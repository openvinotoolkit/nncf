# Copyright (c) 2023 Intel Corporation
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

from nncf.common.graph.layer_attributes import MultipleInputLayerAttributes
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.graph import get_inputs_for_graph_with_several_connected_components
from nncf.torch.graph.operator_metatypes import PTModuleConv2dMetatype
from tests.post_training.test_templates.models import NNCFGraphToTestConstantFiltering


class DummyConstantMetatype(OperatorMetatype):
    pass


@pytest.mark.parametrize("node_between_const_and_target", [False, True])
def test_get_inputs_for_graph_with_several_connected_components(node_between_const_and_target):
    nncf_graph = NNCFGraphToTestConstantFiltering(
        DummyConstantMetatype,
        PTModuleConv2dMetatype,
        MultipleInputLayerAttributes(2, 3),
        node_between_const_and_target,
        PTNNCFGraph,
    ).nncf_graph
    ref_input_nodes_names = ["/Input_2_0", "/ReadVariable_0", "/Conv2_0", "/Input_1_0", "/Concat_with_missed_input_0"]
    input_nodes = get_inputs_for_graph_with_several_connected_components(nncf_graph)
    assert len(ref_input_nodes_names) == len(input_nodes)
    for node in input_nodes:
        assert node.node_name in ref_input_nodes_names
