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

from nncf.common.graph import NNCFGraph
from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from tests.common.quantization.metatypes import ConstantTestMetatype
from tests.common.quantization.mock_graphs import NodeWithType
from tests.common.quantization.test_filter_constant_nodes import create_mock_graph
from tests.common.quantization.test_filter_constant_nodes import get_nncf_graph_from_mock_nx_graph


# pylint: disable=protected-access
class NNCFGraphToTest:
    def __init__(
        self,
        conv_metatype,
        conv_layer_attrs=None,
        nncf_graph_cls=NNCFGraph,
        input_layer_attrs=None,
        output_layer_attrs=None,
    ):
        #       Original graph
        #          Input_1
        #             |
        #           Conv_1
        #             |
        #           Output_1
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype, layer_attributes=input_layer_attrs),
            NodeWithType("Conv_1", conv_metatype, layer_attributes=conv_layer_attrs),
            NodeWithType("Output_1", OutputNoopMetatype, layer_attributes=output_layer_attrs),
        ]
        node_edges = [("Input_1", "Conv_1"), ("Conv_1", "Output_1")]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph, nncf_graph_cls)


class NNCFGraphToTestDepthwiseConv:
    def __init__(
        self,
        depthwise_conv_metatype,
        conv_layer_attrs=None,
        input_layer_attrs=None,
        output_layer_attrs=None,
    ):
        #       Original graph
        #          Input_1
        #             |
        #        DepthwiseConv_1
        #             |
        #           Output_1
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype, layer_attributes=input_layer_attrs),
            NodeWithType("Conv_1", depthwise_conv_metatype, layer_attributes=conv_layer_attrs),
            NodeWithType("Output_1", OutputNoopMetatype, layer_attributes=output_layer_attrs),
        ]
        node_edges = [("Input_1", "Conv_1"), ("Conv_1", "Output_1")]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph)


class NNCFGraphToTestSumAggregation:
    def __init__(
        self,
        conv_metatype,
        sum_metatype,
        conv_layer_attrs=None,
        nncf_graph_cls=NNCFGraph,
        sum_layer_attrs=None,
        input_layer_attrs=None,
        output_layer_attrs=None,
    ):
        #       Original graph
        #          Input_1
        #             |
        #          Conv_1
        #             |
        #           Sum_1
        #             |
        #           Output_1
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype, layer_attributes=input_layer_attrs),
            NodeWithType("Conv_1", conv_metatype, layer_attributes=conv_layer_attrs),
            NodeWithType("Sum_1", sum_metatype, layer_attributes=sum_layer_attrs),
            NodeWithType("Output_1", OutputNoopMetatype, layer_attributes=output_layer_attrs),
        ]
        node_edges = [("Input_1", "Conv_1"), ("Conv_1", "Sum_1"), ("Sum_1", "Output_1")]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph, nncf_graph_cls)
        # Hack output size of the Sum_1 operation
        self.nncf_graph._nx_graph.out_edges[("2 /Sum_1_0", "3 /Output_1_0")][
            self.nncf_graph.ACTIVATION_SHAPE_EDGE_ATTR
        ] = [1, 1, 1]


class NNCFGraphToTestMatMul:
    def __init__(
        self,
        matmul_metatype,
        matmul_layer_attrs=None,
        nncf_graph_cls=NNCFGraph,
        input_layer_attrs=None,
        output_layer_attrs=None,
    ):
        #       Original graphs
        #          Input_1
        #             |
        #           MatMul_1
        #             |
        #           Output_1
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype, layer_attributes=input_layer_attrs),
            NodeWithType("MatMul_1", matmul_metatype, layer_attributes=matmul_layer_attrs),
            NodeWithType("Output_1", OutputNoopMetatype, layer_attributes=output_layer_attrs),
        ]
        node_edges = [("Input_1", "MatMul_1"), ("MatMul_1", "Output_1")]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph, nncf_graph_cls)


class NNCFGraphCA:
    def __init__(
        self,
        conv_metatype,
        conv_layer_attrs=None,
        conv_2_layer_attrs=None,
        use_one_layer_attrs=True,
        nncf_graph_cls=NNCFGraph,
    ):
        #       Original graph
        #          Input_1
        #             |
        #           Conv_1
        #             |
        #           Conv_2
        #             |
        #           Output_1
        if use_one_layer_attrs and not conv_layer_attrs is None and conv_2_layer_attrs is None:
            conv_2_layer_attrs = conv_layer_attrs
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Conv_1_W", ConstantTestMetatype),
            NodeWithType("Conv_1", conv_metatype, layer_attributes=conv_layer_attrs),
            NodeWithType("Conv_2_W", ConstantTestMetatype),
            NodeWithType("Conv_2", conv_metatype, layer_attributes=conv_2_layer_attrs),
            NodeWithType("Output_1", OutputNoopMetatype),
        ]
        node_edges = [
            ("Input_1", "Conv_1"),
            ("Conv_1", "Conv_2"),
            ("Conv_2", "Output_1"),
            ("Conv_1_W", "Conv_1"),
            ("Conv_2_W", "Conv_2"),
        ]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph, nncf_graph_cls)


class NNCFGraphCAWithBias:
    def __init__(
        self,
        conv_metatype,
        add_metatype,
        conv_layer_attrs=None,
        both_biases=True,
        add_layer_attrs=None,
        constant_metatype=ConstantTestMetatype,
        nncf_graph_cls=NNCFGraph,
    ):
        #       Original graph
        #          Input_1
        #             |
        #           Conv_1
        #             |
        #           Add_1
        #             |
        #           Conv_2
        #             |
        #           Add_2
        #           Output_1
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Conv_1_W", constant_metatype),
            NodeWithType("Conv_1", conv_metatype, layer_attributes=conv_layer_attrs),
            NodeWithType("Add_1_W", constant_metatype),
            NodeWithType("Add_1", add_metatype, layer_attributes=add_layer_attrs),
            NodeWithType("Conv_2_W", constant_metatype),
            NodeWithType("Conv_2", conv_metatype, layer_attributes=conv_layer_attrs),
            NodeWithType("Output_1", OutputNoopMetatype),
        ]
        if both_biases:
            nodes.extend(
                [
                    NodeWithType("Add_2_W", constant_metatype),
                    NodeWithType("Add_2", add_metatype, layer_attributes=add_layer_attrs),
                ]
            )
        node_edges = [
            ("Input_1", "Conv_1"),
            ("Conv_1", "Add_1"),
            ("Add_1", "Conv_2"),
            ("Conv_1_W", "Conv_1"),
            ("Add_1_W", "Add_1"),
            ("Conv_2_W", "Conv_2"),
        ]
        if both_biases:
            node_edges.extend([("Conv_2", "Add_2"), ("Add_2", "Output_1"), ("Add_2_W", "Add_2")])
        else:
            node_edges.extend([("Conv_2", "Output_1")])
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph, nncf_graph_cls)
