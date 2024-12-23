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

from typing import Dict, List, Optional, Type, Union

import openvino.runtime
from openvino.runtime import opset13 as opset

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.torch.sparsify_activations.sparsify_activations_impl import SparsifyActivationsAlgoBackend
from nncf.openvino.graph.metatypes import openvino_metatypes as om
from nncf.openvino.graph.model_transformer import OVModelTransformer
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.statistics.collectors import OVAbsQuantileReducer
from nncf.torch.nncf_network import NNCFNetwork

ACTIVATIONS_SPARSIFIER_PREFIX = "activations_sparsifier"


class OVSparsifyActivationsAlgoBackend(SparsifyActivationsAlgoBackend):
    """
    OpenVINO backend for the activation sparsification algorithm.
    """

    @property
    def supported_metatypes(self) -> List[Type[OperatorMetatype]]:
        return [om.OVMatMulMetatype]

    def abs_quantile_reducer(self, quantile: Optional[Union[float, List[float]]] = None) -> OVAbsQuantileReducer:
        return OVAbsQuantileReducer(quantile=quantile)

    def target_point(self, target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(TargetType.PRE_LAYER_OPERATION, target_node_name, port_id=port_id)

    def insert_sparsifiers(
        self,
        model: openvino.Model,
        graph: NNCFGraph,
        threshold_by_node: Dict[NNCFNode, float],
    ) -> NNCFNetwork:
        name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)
        for nncf_node, threshold in threshold_by_node.items():
            activation_port_id = self.get_activation_port_id(nncf_node, graph)
            matmul_node = name_to_node_mapping[nncf_node.node_name]
            dense_activation = matmul_node.input(activation_port_id).get_source_output().get_node()

            dtype = dense_activation.get_element_type()
            threshold_const = opset.constant(threshold, dtype=dtype, name=f"{matmul_node.name}/sparsity_threshold")
            zero_const = opset.constant(0.0, dtype=dtype)

            less_mask = opset.less_equal(opset.abs(dense_activation), threshold_const)
            sparse_activation = opset.select(
                less_mask, zero_const, dense_activation, name=f"{matmul_node.name}/sparse_input"
            )
            matmul_node.input(activation_port_id).replace_source_output(sparse_activation.output(0))

        return model

    @staticmethod
    def get_activation_port_id(matmul_node: NNCFNode, nncf_graph: NNCFGraph) -> int:
        return 0
        n_inputs = len(nncf_graph.get_input_edges(matmul_node))
        if n_inputs != 2:
            raise RuntimeError(f"Expected node to have two inputs, but found {n_inputs} for node {matmul_node}.")

        is_const_node_on_port = [
            nncf_graph.get_input_edges(matmul_node)[i].from_node.node_type == "Constant" for i in range(2)
        ]
        if is_const_node_on_port[0] != is_const_node_on_port[1]:
            assert not is_const_node_on_port[0], matmul_node.node_name
            return 1 if is_const_node_on_port[0] else 0

        # Try to match compressed constant subgraph
        for i in range(2):
            node = nncf_graph.get_input_edges(matmul_node)[i].from_node
            if node.node_type == "Convert":
                node = nncf_graph.get_input_edges(node)[0].from_node
            if node.node_type == "Reshape":
                node = nncf_graph.get_input_edges(node)[0].from_node
            if node.node_type == "Multiply":
                node = nncf_graph.get_input_edges(node)[0].from_node
                if node.node_type == "Subtract":
                    node = nncf_graph.get_input_edges(node)[0].from_node
                if node.node_type == "Convert":
                    node = nncf_graph.get_input_edges(node)[0].from_node
                else:
                    continue
            if node.node_type == "Constant":
                assert i == 1, matmul_node.node_name
                return int(i == 0)

        raise RuntimeError(f"Could not find activation port id for node {matmul_node}.")
