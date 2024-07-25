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

from typing import Tuple

import torch.fx

import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph import NNCFNode
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.logging import nncf_logger
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import PT_OPERATOR_METATYPES


class GraphConverter:
    """
    Builds the NNCFGraph from an torch.fx.GraphModule instance.
    """

    @staticmethod
    def _get_node_type_and_metatype(node: torch.fx.Node) -> Tuple[str, om.OperatorMetatype]:
        """
        Retrieves node's type and metatype.

        :param node: Given node.
        :return: Node's type and metatype.
        """
        if node.op == "placeholder":
            node_type = "input"
            node_metatype = om.PTInputNoopMetatype
        elif node.op == "output":
            node_type = "output"
            node_metatype = om.PTOutputNoopMetatype
        elif node.op == "get_attr":
            node_type = "get_attr"
            node_metatype = om.PTConstNoopMetatype
        elif node.op in ("call_function",):
            if hasattr(node.target, "overloadpacket"):
                node_type = str(node.target.overloadpacket).split(".")[1]
            elif node.target.__name__ == "getitem":
                node_type = "__getitem__"
            else:
                # TODO(dlyakhov): get correct nodes types from this nodes as well
                node_type = str(node.target)
            node_metatype = PT_OPERATOR_METATYPES.get_operator_metatype_by_op_name(node_type)
        else:
            node_type = node.op
            node_metatype = UnknownMetatype
        if node_metatype is UnknownMetatype:
            nncf_logger.debug(f"Unknown metatype for node: {node}")
        return node_type, node_metatype

    @staticmethod
    def create_nncf_graph(model: torch.fx.GraphModule) -> PTNNCFGraph:
        """
        Creates NNCFGraph from GraphModule.
        All nodes from model which have valid metatype are added to NNCFGraph.
        Then, corresponding edges are added to the NNCFGraph with shape, type, output and input port ids.

        :param model: torch fx GraphModule.
        :return: NNCFGraph.
        """

        nncf_graph = PTNNCFGraph()

        for source_node in model.graph.nodes:
            node_type, node_metatype = GraphConverter._get_node_type_and_metatype(source_node)

            nncf_graph.add_nncf_node(
                node_name=source_node.name,
                node_type=node_type,
                node_metatype=node_metatype,
            )

        for source_node in model.graph.nodes:
            source_nncf_node = nncf_graph.get_node_by_name(source_node.name)
            for idx, dist_node in enumerate(source_node.users):
                dist_node_id = nncf_graph.get_node_by_name(dist_node.name).node_id
                input_port_id, output_port_id, tensor_shape = GraphConverter.get_edge_params(
                    model, source_node, source_nncf_node, dist_node, idx
                )

                nncf_graph.add_edge_between_nncf_nodes(
                    source_nncf_node.node_id,
                    dist_node_id,
                    tensor_shape=tensor_shape,
                    input_port_id=input_port_id,
                    output_port_id=output_port_id,
                    dtype=Dtype.FLOAT,
                )

        return nncf_graph

    @staticmethod
    def get_edge_params(
        model: torch.fx.GraphModule,
        source_node: torch.fx.Node,
        source_nncf_node: NNCFNode,
        dist_node: torch.fx.Node,
        output_idx: int,
    ) -> Tuple[int, int, Tuple[int, ...]]:
        """
        Retrieves edge params from the given source_node and dist_node pair.

        :param model: A torch.fx.GraphModule instance.
        :param source_node: Source node in format of torch.fx.Node.
        :param source_nncf_node: Source node in format of NNCFNode.
        :param dist_node: Distance node in format of torch.fx.Node.
        :param output_idx: Output indes of the source_node.
        :return: Tuple of edge parameters: edge input port id, edge output port id and
            edge tensor shape.
        """
        output_port_id = 0
        if source_node.op in ("get_attr",):
            tensor_shape = tuple(getattr(model, source_node.target).shape)
        elif "val" in source_node.meta:
            if source_nncf_node.metatype is om.PTBatchNormMetatype:
                tensor = source_node.meta["val"][0]
            elif source_nncf_node.metatype is om.PTSplitMetatype:
                tensor = source_node.meta["val"][output_idx]
                # Assume every split outputs corresponds to an unique output_port_id
                output_port_id = output_idx
            else:
                tensor = source_node.meta["val"]
            tensor_shape = tuple(tensor.shape)
        else:
            # TODO(dlyakhov): Refactor algorithms to always have knowns edges shapes.
            nncf_logger.debug(f"Edge shape between {source_node.name} and {dist_node.name} is unknown.")
            tensor_shape = None

        input_port_id = dist_node.all_input_nodes.index(source_node)
        return input_port_id, output_port_id, tensor_shape
