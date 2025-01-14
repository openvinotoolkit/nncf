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

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.pruning.utils import is_batched_linear


@pytest.mark.parametrize(
    "batched,has_output_edges,res", [(False, True, False), (True, True, True), (True, False, False)]
)
def test_is_batched_linear(batched, has_output_edges, res):
    graph = NNCFGraph()
    linear = graph.add_nncf_node(
        "linear", "linear", "linear", LinearLayerAttributes(True, in_features=5, out_features=5)
    )
    if has_output_edges:
        last_linear = graph.add_nncf_node(
            "last_linear", "linear", "linear", LinearLayerAttributes(True, in_features=5, out_features=5)
        )
        tensor_shape = [5, 5] if not batched else [5, 5, 5]
        graph.add_edge_between_nncf_nodes(linear.node_id, last_linear.node_id, tensor_shape, 0, 0, Dtype.FLOAT)
    assert is_batched_linear(linear, graph) == res
