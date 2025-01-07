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

from collections import Counter
from typing import Tuple

import torch.fx

import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph import NNCFNode
from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.logging import nncf_logger
from nncf.experimental.torch.fx.node_utils import get_tensor_constant_from_node
from nncf.torch.dynamic_graph.layer_attributes_handlers import apply_args_defaults
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import PT_OPERATOR_METATYPES


class GraphConverter:
    """
    Builds the NNCFGraph from an torch.fx.GraphModule instance.
    """

    def _get_layer_attributes(
        node: torch.fx.Node, metatype: om.OperatorMetatype, model: torch.fx.GraphModule
    ) -> BaseLayerAttributes:
        """
        Collects layer attributes for the given node.

        :param node: Given node.
        :param metatype: Given node metatype.
        :param model: Target GraphModule instance.
        :return: Given node layer attributes.
        """
        if metatype in [om.PTConv1dMetatype, om.PTConv2dMetatype, om.PTConv3dMetatype]:
            conv_default_args = [(arg.name, arg.default_value) for arg in node.target._schema.arguments]
            kwargs = apply_args_defaults(node.args, node.kwargs, conv_default_args)

            weight_node = kwargs["weight"]
            if weight_node.op != "get_attr":
                # Convs with constant subgraphs or two inputs are not supported yet.
                return None
            weight = get_tensor_constant_from_node(weight_node, model)
            return ConvolutionLayerAttributes(
                weight_requires_grad=False,
                in_channels=weight.shape[0],
                out_channels=weight.shape[1],
                kernel_size=list(weight.shape[2:]),
                stride=kwargs["stride"],
                dilations=kwargs["dilation"],
                groups=kwargs["groups"],
                padding_values=kwargs["padding"],
                transpose=False,
            )
        return None

    def _map_fx_unique_metatypes(node: torch.fx.Node, metatype: om.OperatorMetatype) -> om.OperatorMetatype:
        """
        Attempts to retrieve correct subtype for the given node.

        :param node: Given node.
        :param metatype: Given node metatype.
        :param model: Target GraphModule instance.
        :return: Correct FX metatype of the given node if it is exist or the original node metatype otherwise.
        """
        if metatype in [om.PTEmbeddingMetatype]:
            weight_node = node.args[0]
            if weight_node.op == "get_attr":
                return om.PTAtenEmbeddingMetatype

        return metatype

    @staticmethod
    def _get_node_type_and_metatype(
        node: torch.fx.Node, model: torch.fx.GraphModule
    ) -> Tuple[str, om.OperatorMetatype]:
        """
        Retrieves node's type and metatype.

        :param node: Given node.
        :param model: Given GraphModule.
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

        if node_metatype.get_subtypes():
            layer_attrs = GraphConverter._get_layer_attributes(node, node_metatype, model)
            node_subtype = node_metatype.determine_subtype(layer_attrs)
            node_metatype = node_subtype or node_metatype
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

        const_targets_counter = Counter([node.target for node in model.graph.nodes if node.op == "get_attr"])
        for source_node in model.graph.nodes:
            node_type, node_metatype = GraphConverter._get_node_type_and_metatype(source_node, model)
            node_metatype = GraphConverter._map_fx_unique_metatypes(source_node, node_metatype)
            is_shared_node = source_node.op in ("get_attr",) and (
                const_targets_counter[source_node.target] > 1 or len(source_node.users) > 1
            )

            nncf_graph.add_nncf_node(
                node_name=source_node.name, node_type=node_type, node_metatype=node_metatype, is_shared=is_shared_node
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
        :param output_idx: Output index of the source_node.
        :return: Tuple of edge parameters: edge input port id, edge output port id and
            edge tensor shape.
        """
        output_port_id = 0
        tensor_shape = None
        if source_node.op in ("get_attr",):
            tensor_shape = tuple(get_tensor_constant_from_node(source_node, model).shape)
        elif "val" in source_node.meta:
            if source_nncf_node.metatype is om.PTBatchNormMetatype and isinstance(
                source_node.meta["val"], (tuple, list)
            ):
                tensor = source_node.meta["val"][0]
            elif source_nncf_node.metatype in [om.PTSplitMetatype, om.PTMaxMetatype, om.PTMinMetatype]:
                tensor = source_node.meta["val"][output_idx]
                # Assume every outputs corresponds to an unique output_port_id
                output_port_id = output_idx
            else:
                tensor = source_node.meta["val"]
            if isinstance(tensor, torch.Tensor):
                tensor_shape = tuple(tensor.shape)

        if tensor_shape is None:
            # TODO(dlyakhov): Refactor algorithms to always have knowns edges shapes.
            nncf_logger.debug(f"Edge shape between {source_node.name} and {dist_node.name} is unknown.")

        input_port_id = dist_node.all_input_nodes.index(source_node)
        return input_port_id, output_port_id, tensor_shape
