
from nncf.common.graph import NNCFGraph
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from nncf.common.graph.operator_metatypes import InputNoopMetatype

from tests.common.quantization.test_filter_constant_nodes import create_mock_graph
from tests.common.quantization.test_filter_constant_nodes import get_nncf_graph_from_mock_nx_graph
from tests.common.quantization.mock_graphs import NodeWithType


# pylint: disable=protected-access
class NNCFGraphToTest:
    def __init__(self, conv_metatype,
                 conv_layer_attrs = None,
                 nncf_graph_cls = NNCFGraph):
        #       Original graph
        #          Input_1
        #             |
        #           Conv_1
        #             |
        #           Output_1
        nodes = [NodeWithType('Input_1', InputNoopMetatype),
                 NodeWithType('Conv_1', conv_metatype,
                              conv_layer_attrs),
                 NodeWithType('Output_1', OutputNoopMetatype),
                 ]
        node_edges = [('Input_1', 'Conv_1'), ('Conv_1', 'Output_1')]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph, nncf_graph_cls)


class NNCFGraphToTestDepthwiseConv:
    def __init__(self, depthwise_conv_metatype):
        #       Original graph
        #          Input_1
        #             |
        #        DepthwiseConv_1
        #             |
        #           Output_1
        nodes = [NodeWithType('Input_1', InputNoopMetatype),
                 NodeWithType('Conv_1', depthwise_conv_metatype),
                 NodeWithType('Output_1', OutputNoopMetatype),
                 ]
        node_edges = [('Input_1', 'Conv_1'), ('Conv_1', 'Output_1')]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph)
