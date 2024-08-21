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

from typing import Dict, List, Type, Optional, Union

from openvino.runtime import opset13 as opset
import openvino.runtime

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.torch.sparsify_activations.sparsify_activations_impl import SparsifyActivationsAlgoBackend
from nncf.openvino.graph.model_transformer import OVModelTransformer
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.statistics.collectors import OVAbsQuantileReducer
from nncf.tensor.functions.torch_numeric import quantile
from nncf.openvino.graph.metatypes import openvino_metatypes as om
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
            sparse_activation = opset.select(less_mask, zero_const, dense_activation, name=f"{matmul_node.name}/sparse_input")
            matmul_node.input(activation_port_id).replace_source_output(sparse_activation.output(0))

        return model

    @staticmethod
    def get_activation_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> int:
        return 0

        # Code below won't work for the case of compressed weight constant
        constant_ports = node.layer_attributes.get_const_port_ids()
        activation_ports = [
            e.input_port_id for e in nncf_graph.get_input_edges(node) if e.input_port_id not in constant_ports
        ]
        assert len(activation_ports) == 1
        return activation_ports[0]
