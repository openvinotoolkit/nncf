# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import chain

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.torch.function_hook.graph.graph_utils import TensorMeta
from nncf.torch.function_hook.nncf_graph.layer_attributes import PT2OpLayerAttributes
from nncf.torch.graph.transformations.commands import PTTargetPoint


class PTNNCFGraph(NNCFGraph):
    def get_input_shape_for_insertion_point(self, insertion_point: PTTargetPoint) -> tuple[int]:
        nncf_node = self.get_node_by_name(insertion_point.target_node_name)
        if insertion_point.input_port_id is not None:  # PRE_LAYER_OPERATION + OPERATION_WITH_WEIGHTS
            return self.get_input_edge_by_port_id(nncf_node, insertion_point.input_port_id).tensor_shape
        return next(edge.tensor_shape for edge in self.get_output_edges(nncf_node))

    def get_nodes_with_missed_input_edges(self) -> list[NNCFNode]:
        """
        Returns a list of NNCFNodes that have at least one expected input edge missed.
        Requires MultipleInputLayerAttributes for nodes with several inputs and
        right `num_expected_input_edges` parameter setted for nncf nodes metatypes.

        :return: List of NNCFNodes that are identified as disconnected.
        """
        input_nodes = set()

        # Check expected number of input edges by counting TensorMeta in op_args and op_kwargs.
        for node in self.get_all_nodes():
            input_edges = len(self.get_input_edges(node))
            if not isinstance(node.layer_attributes, PT2OpLayerAttributes):
                continue
            num_expected_input_edges = 0
            for val in chain(node.layer_attributes.op_args, node.layer_attributes.op_kwargs.values()):
                if isinstance(val, TensorMeta):
                    num_expected_input_edges += 1
                if isinstance(val, (list, tuple)):
                    num_expected_input_edges += sum(isinstance(v, TensorMeta) for v in val)
            if input_edges < num_expected_input_edges:
                input_nodes.add(node)

        return list(input_nodes)
