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

from nncf.common.graph import NNCFGraph
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from tests.common.quantization.metatypes import ConstantTestMetatype
from tests.common.quantization.mock_graphs import NodeWithType
from tests.common.quantization.mock_graphs import create_mock_graph
from tests.common.quantization.mock_graphs import get_nncf_graph_from_mock_nx_graph


class NNCFGraphToTest:
    def __init__(
        self,
        conv_metatype,
        conv_layer_attrs=None,
        nncf_graph_cls=NNCFGraph,
        input_layer_attrs=None,
        output_layer_attrs=None,
        const_metatype=None,
        const_layer_attrs=None,
    ):
        #       Original graph
        #          Input_1  Const_1
        #             |    /
        #           Conv_1
        #             |
        #           Output_1
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype, layer_attributes=input_layer_attrs),
            NodeWithType("Conv_1", conv_metatype, layer_attributes=conv_layer_attrs),
            NodeWithType("Const_1", const_metatype, layer_attributes=const_layer_attrs),
            NodeWithType("Output_1", OutputNoopMetatype, layer_attributes=output_layer_attrs),
        ]
        node_edges = [("Input_1", "Conv_1"), ("Const_1", "Conv_1"), ("Conv_1", "Output_1")]
        original_mock_graph = create_mock_graph(
            nodes,
            node_edges,
            (
                {NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR: (1, 3, 224, 224)},
                {NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR: (3, 10, 4, 4), NNCFGraph.INPUT_PORT_ID_EDGE_ATTR: 1},
                {NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR: (1, 10, 224, 224)},
            ),
        )
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph, nncf_graph_cls)


class NNCFGraphToTestDepthwiseConv:
    def __init__(
        self,
        depthwise_conv_metatype,
        conv_layer_attrs=None,
        input_layer_attrs=None,
        output_layer_attrs=None,
        nncf_graph_cls=NNCFGraph,
        const_metatype=None,
        const_layer_attrs=None,
    ):
        #       Original graph
        #          Input_1   Const_1
        #             |       /
        #        DepthwiseConv_1
        #             |
        #           Output_1
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype, layer_attributes=input_layer_attrs),
            NodeWithType("Conv_1", depthwise_conv_metatype, layer_attributes=conv_layer_attrs),
            NodeWithType("Const_1", const_metatype, layer_attributes=const_layer_attrs),
            NodeWithType("Output_1", OutputNoopMetatype, layer_attributes=output_layer_attrs),
        ]
        node_edges = [("Input_1", "Conv_1"), ("Conv_1", "Output_1")]
        original_mock_graph = create_mock_graph(
            nodes,
            node_edges,
            (
                {},
                {NNCFGraph.INPUT_PORT_ID_EDGE_ATTR: 1},
                {},
            ),
        )
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph, nncf_graph_cls)


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
        const_metatype=None,
        const_layer_attrs=None,
    ):
        #       Original graph
        #          Input_1  Const1
        #             |     /
        #          Conv_1
        #             |
        #           Sum_1
        #             |
        #           Output_1
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype, layer_attributes=input_layer_attrs),
            NodeWithType("Conv_1", conv_metatype, layer_attributes=conv_layer_attrs),
            NodeWithType("Const_1", const_metatype, layer_attributes=const_layer_attrs),
            NodeWithType("Sum_1", sum_metatype, layer_attributes=sum_layer_attrs),
            NodeWithType("Const_2", const_metatype, layer_attributes=const_layer_attrs),
            NodeWithType("Output_1", OutputNoopMetatype, layer_attributes=output_layer_attrs),
        ]
        node_edges = [
            ("Input_1", "Conv_1"),
            ("Const_1", "Conv_1"),
            ("Conv_1", "Sum_1"),
            ("Const_2", "Sum_1"),
            ("Sum_1", "Output_1"),
        ]
        original_mock_graph = create_mock_graph(
            nodes,
            node_edges,
            (
                {},
                {NNCFGraph.INPUT_PORT_ID_EDGE_ATTR: 1},
                {},
                {NNCFGraph.INPUT_PORT_ID_EDGE_ATTR: 1},
                {NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR: [1, 1, 1]},
            ),
        )
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph, nncf_graph_cls)


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
        if use_one_layer_attrs and conv_layer_attrs is not None and conv_2_layer_attrs is None:
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
        conv_1_layer_attrs=None,
        conv_2_layer_attrs=None,
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
        if conv_2_layer_attrs is None:
            conv_2_layer_attrs = conv_1_layer_attrs
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Conv_1_W", constant_metatype),
            NodeWithType("Conv_1", conv_metatype, layer_attributes=conv_1_layer_attrs),
            NodeWithType("Add_1_W", constant_metatype),
            NodeWithType("Add_1", add_metatype, layer_attributes=add_layer_attrs),
            NodeWithType("Conv_2_W", constant_metatype),
            NodeWithType("Conv_2", conv_metatype, layer_attributes=conv_2_layer_attrs),
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


class NNCFGraphDropoutRemovingCase:
    def __init__(
        self,
        dropout_metatype,
        wrong_dropout_node: bool = False,
        wrong_parallel_edges: bool = False,
        nncf_graph_cls=NNCFGraph,
    ):
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Split_1", None),
            NodeWithType(
                "Dropout_1",
                dropout_metatype,
            ),
            NodeWithType("Output_1", OutputNoopMetatype),
            NodeWithType(
                "Dropout_2",
                dropout_metatype,
            ),
            NodeWithType("Output_2_1", OutputNoopMetatype),
            NodeWithType("Output_2_2", OutputNoopMetatype),
            NodeWithType("Output_2_3", OutputNoopMetatype),
            NodeWithType(
                "Dropout_3",
                dropout_metatype,
            ),
            NodeWithType("Output_3", OutputNoopMetatype),
        ]
        node_edges = [
            ("Input_1", "Split_1"),
            ("Split_1", "Dropout_1"),
            ("Dropout_1", "Output_1"),
            ("Split_1", "Dropout_2"),
            ("Dropout_2", "Output_2_1"),
            ("Dropout_2", "Output_2_2"),
            ("Dropout_2", "Output_2_3"),
            ("Split_1", "Dropout_3"),
            ("Dropout_3", "Output_3"),
        ]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph, nncf_graph_cls)

        dropout_2 = self.nncf_graph.get_node_by_key("3 /Dropout_2_0")
        output = self.nncf_graph.add_nncf_node("/Output_2_4_0", "output", OutputNoopMetatype)
        tensor_shape = [1, 2, 1, 1] if wrong_dropout_node else [1, 1, 1, 1]
        self.nncf_graph.add_edge_between_nncf_nodes(
            dropout_2.node_id,
            output.node_id,
            tensor_shape=tensor_shape,
            input_port_id=15,
            output_port_id=1,
            dtype=Dtype.FLOAT,
        )

        dropout_2 = self.nncf_graph.get_node_by_key("4 /Dropout_3_0")
        output = self.nncf_graph.add_nncf_node("/Output_3_1_0", "output", OutputNoopMetatype)
        self.nncf_graph.add_edge_between_nncf_nodes(
            dropout_2.node_id,
            output.node_id,
            tensor_shape=tensor_shape,
            input_port_id=1,
            output_port_id=1,
            dtype=Dtype.FLOAT,
            parallel_input_port_ids=list(range(2, 10)),
        )
        if wrong_parallel_edges:
            dropout_4 = self.nncf_graph.add_nncf_node("100 /dropout", "dropout", dropout_metatype)
            self.nncf_graph.add_edge_between_nncf_nodes(
                self.nncf_graph.get_node_by_key("0 /Input_1_0").node_id,
                dropout_4.node_id,
                tensor_shape=[1, 1, 1, 1],
                input_port_id=0,
                output_port_id=0,
                dtype=Dtype.FLOAT,
                parallel_input_port_ids=list(range(1, 10)),
            )


class NNCFGraphToTestConstantFiltering:
    def __init__(
        self,
        constant_metatype,
        node_with_weights_metatype,
        concat_layer_attr,
        add_node_between_const_and_weight_node,
        nncf_graph_cls=NNCFGraph,
    ) -> None:
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Conv", node_with_weights_metatype),
            NodeWithType("Weights", constant_metatype),
            NodeWithType("Weights2", constant_metatype),
            NodeWithType("Conv2", node_with_weights_metatype),
            NodeWithType("ReadVariable", InputNoopMetatype),
            NodeWithType("Add", None),
            NodeWithType("Weights3", constant_metatype),
            NodeWithType("Weights4", constant_metatype),
            NodeWithType("Conv3", node_with_weights_metatype),
            NodeWithType("NodeAfterConstantConv", None),
            NodeWithType("Final_node", None),
            NodeWithType("Input_2", InputNoopMetatype),
            NodeWithType("Const0", constant_metatype),
            NodeWithType("Const1", constant_metatype),
            NodeWithType("Concat_with_input", None, layer_attributes=concat_layer_attr),
            NodeWithType("Const2", constant_metatype),
            NodeWithType("Const3", constant_metatype),
            NodeWithType("Const4", constant_metatype),
            NodeWithType("Concat_with_constant", None, layer_attributes=concat_layer_attr),
            NodeWithType("Const5", constant_metatype),
            NodeWithType("Const6", constant_metatype),
            NodeWithType("Concat_with_missed_input", None, layer_attributes=concat_layer_attr),
        ]

        edges = [
            ("Input_1", "Conv"),
            ("Weights", "Conv"),
            ("Weights2", "Conv2"),
            ("Conv2", "Add"),
            ("ReadVariable", "Add"),
            ("Add", "Final_node"),
            ("Weights3", "Conv3"),
            ("Weights4", "Conv3"),
            ("Conv3", "NodeAfterConstantConv"),
            ("Input_2", "Concat_with_input"),
            ("Const0", "Concat_with_input"),
            ("Const1", "Concat_with_input"),
            ("Const2", "Concat_with_constant"),
            ("Const3", "Concat_with_constant"),
            ("Const4", "Concat_with_constant"),
            ("Const5", "Concat_with_missed_input"),
            ("Const6", "Concat_with_missed_input"),
        ]
        if add_node_between_const_and_weight_node:
            constant_nodes = [node for node in nodes if node.node_op_metatype is constant_metatype]
            const_node_to_edge = {}
            for node in constant_nodes:
                for i, edge in enumerate(edges):
                    if node.node_name == edge[0]:
                        const_node_to_edge[node] = edge
                        break
                del edges[i]
            for node, edge in const_node_to_edge.items():
                any_after_node_name = f"AnyAfter{node.node_name}"
                nodes.append(NodeWithType(any_after_node_name, None))
                edges.append((edge[0], any_after_node_name))
                edges.append((any_after_node_name, edge[1]))

        original_mock_graph = create_mock_graph(nodes, edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph, nncf_graph_cls)


class NNCFGraphTransformer:
    def __init__(
        self,
        matmul_metatype,
        softmax_metatype,
        transpose_metatype,
        const_metatype,
        mul_metatype,
        matmul_layer_weighted_attrs=None,
        matmul_layer_non_weighted_attrs=None,
        default_layer_attrs=None,
        nncf_graph_cls=NNCFGraph,
    ):
        # softmax((K x Q) * scale) x V.T
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype, layer_attributes=default_layer_attrs),
            NodeWithType("W_K", const_metatype, layer_attributes=default_layer_attrs),
            NodeWithType("W_Q", const_metatype, layer_attributes=default_layer_attrs),
            NodeWithType("W_V", const_metatype, layer_attributes=default_layer_attrs),
            NodeWithType("K", matmul_metatype, layer_attributes=matmul_layer_weighted_attrs),
            NodeWithType("Q", matmul_metatype, layer_attributes=matmul_layer_weighted_attrs),
            NodeWithType("V", matmul_metatype, layer_attributes=matmul_layer_weighted_attrs),
            NodeWithType("K_Q", matmul_metatype, layer_attributes=matmul_layer_non_weighted_attrs),
            NodeWithType("div", mul_metatype, layer_attributes=default_layer_attrs),
            NodeWithType("softmax", softmax_metatype, layer_attributes=default_layer_attrs),
            NodeWithType("transpose", transpose_metatype, layer_attributes=default_layer_attrs),
            NodeWithType("SA_V", matmul_metatype, layer_attributes=matmul_layer_non_weighted_attrs),
            NodeWithType("Output_1", OutputNoopMetatype, layer_attributes=default_layer_attrs),
        ]
        node_edges = [
            ("Input_1", "K"),
            ("W_K", "K"),
            ("Input_1", "Q"),
            ("W_Q", "Q"),
            ("Input_1", "V"),
            ("W_V", "V"),
            ("K", "K_Q"),
            ("Q", "K_Q"),
            ("K_Q", "div"),
            ("div", "softmax"),
            ("softmax", "SA_V"),
            ("V", "transpose"),
            ("transpose", "SA_V"),
            ("SA_V", "Output_1"),
        ]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph, nncf_graph_cls)
