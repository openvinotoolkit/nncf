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

from itertools import chain

import torch.fx

import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph import NNCFGraph
from nncf.experimental.torch_fx.operator_metatypes import FX_OPERATOR_METATYPES


class GraphConverter:
    """
    Builds the NNCFGraph from an OpenVINO model.
    """

    @staticmethod
    def _get_leaf_node(module: torch.nn.Module, node: torch.fx.Node) -> torch.nn.Module:
        py_obj = module
        assert isinstance(node.target, str)
        atoms = node.target.split(".")
        for atom in atoms:
            if not hasattr(py_obj, atom):
                raise RuntimeError(str(py_obj) + " does not have attribute " + atom + "!")
            py_obj = getattr(py_obj, atom)
        return py_obj

    @staticmethod
    def create_nncf_graph(model: torch.fx.GraphModule) -> NNCFGraph:
        """
        Creates NNCFGraph from OpenVINO Model.
        All nodes from model which have valid metatype are added to NNCFGraph.
        Then, corresponding edges are added to the NNCFGraph with shape, type, output and input port ids.

        :param model: OpenVINO model.
        :return: NNCFGraph.
        """

        nncf_graph = NNCFGraph()

        ignore_getattr = False
        ignore_parameters_and_buffers = False

        for node in model.graph.nodes:
            if ignore_getattr and node.op == "get_attr":
                continue

            print(node.name, node.op, sep=" ")
            metatype = FX_OPERATOR_METATYPES.get_operator_metatype_by_op_name(node.op)
            nncf_node = nncf_graph.add_nncf_node(
                node.name,
                node.op,
                metatype,  # layer_attributes,
            )

            def get_module_params_or_buffers():
                for pname, ptensor in chain(leaf_module.named_parameters(), leaf_module.named_buffers()):
                    pname1 = node.name + "." + pname
                    nncf_param_node = nncf_graph.add_nncf_node(
                        pname1,
                        "parameter" if isinstance(ptensor, torch.nn.Parameter) else "buffer",
                        om.PTConstNoopMetatype,
                    )
                    nncf_graph.add_edge_between_nncf_nodes(
                        nncf_param_node, nncf_node, tensor_shape=[1, 1, 1, 1], input_port_id=0, output_port_id=0
                    )

            if node.op == "call_module":
                leaf_module = GraphConverter._get_leaf_node(model, node)

                if not ignore_parameters_and_buffers and not isinstance(leaf_module, torch.fx.GraphModule):
                    get_module_params_or_buffers()

        for node in model.graph.nodes:
            if ignore_getattr and node.op == "get_attr":
                continue

            source_node = nncf_graph.get_node_by_name(node.name)
            for user in node.users:
                dist_node = nncf_graph.get_node_by_name(user.name)
                nncf_graph.add_edge_between_nncf_nodes(
                    source_node, dist_node, tensor_shape=[1, 1, 1, 1], input_port_id=0, output_port_id=0
                )

        return nncf_graph
