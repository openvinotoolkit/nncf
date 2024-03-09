import xml.etree.ElementTree as ET  # nosec
from dataclasses import dataclass
from typing import Optional

from nncf.experimental.common.graph.netron import get_graph_desc, convert_dummy_precision, EdgeDesc, NodeDesc, PortDesc, Tags, GET_ATTRIBUTES_FN_TYPE
from tests.common.quantization.mock_graphs import get_two_branch_mock_model_graph

import pytest

@dataclass
class GraphDescTestCase:
    include_fq_params: Optional[bool]
    get_attributes_fn: GET_ATTRIBUTES_FN_TYPE

GRAPH_DESC_TEST_CASES = [
    GraphDescTestCase(
        include_fq_params = False,
        get_attributes_fn = None
    ),

    GraphDescTestCase(
        include_fq_params = True,
        get_attributes_fn = None
    ),

    GraphDescTestCase(
        include_fq_params = True,
        get_attributes_fn = lambda x: {
            "name": x.node_name,
            "type": x.node_type
        } 
    )
]

@pytest.mark.parametrize(
    "graph_desc_test_case",
    GRAPH_DESC_TEST_CASES,
)
def test_get_graph_desc(graph_desc_test_case: GraphDescTestCase):
    include_fq_params = graph_desc_test_case.include_fq_params
    get_attributes_fn = graph_desc_test_case.get_attributes_fn
    
    nncf_graph = get_two_branch_mock_model_graph()

    edges = list(nncf_graph.get_all_edges())
    nodes = list(nncf_graph.get_all_nodes())

    node_desc_list, edges_desc_list = get_graph_desc(nncf_graph, include_fq_params, get_attributes_fn)

    assert all(isinstance(node_desc, NodeDesc) for node_desc in node_desc_list)
    assert all(isinstance(edge_desc, EdgeDesc) for edge_desc in edges_desc_list)

    assert len(node_desc_list) == len(nodes)
    assert len(edges_desc_list) == len(edges)

    if get_attributes_fn is not None:
        assert all([node_desc.attrs == get_attributes_fn(node) for node, node_desc in zip(nodes, node_desc_list)])


def test_edge_desc():
    nncf_graph = get_two_branch_mock_model_graph()

    for edge in nncf_graph.get_all_edges():

        edgeDesc = EdgeDesc(
            from_node=str(edge.from_node.node_id),
            from_port=str(edge.output_port_id),
            to_node=str(edge.to_node.node_id),
            to_port=str(edge.input_port_id),
        )

        xmlElement = edgeDesc.as_xml_element()

        assert isinstance(xmlElement, ET.Element)
        assert xmlElement.tag == Tags.EDGE
        assert xmlElement.attrib["from-layer"] == str(edge.from_node.node_id)
        assert xmlElement.attrib["from-port"] == str(edge.output_port_id)
        assert xmlElement.attrib["to-layer"] == str(edge.to_node.node_id)
        assert xmlElement.attrib["to-port"] == str(edge.input_port_id)


def test_node_desc():
    nncf_graph = get_two_branch_mock_model_graph()

    for node in nncf_graph.get_all_nodes():
        nodeDesc = NodeDesc(
            node_id=str(node.node_id),
            name=node.node_name,
            node_type=node.node_type.title(),
        )

        xmlElement = nodeDesc.as_xml_element()

        assert isinstance(xmlElement, ET.Element)
        assert xmlElement.tag == Tags.NODE
        assert xmlElement.attrib["id"] == str(node.node_id)
        assert xmlElement.attrib["name"] == node.node_name
        assert xmlElement.attrib["type"] == node.node_type.title()
        assert all([child.tag == Tags.DATA for child in xmlElement])


def test_port_desc():
    nncf_graph = get_two_branch_mock_model_graph()

    for edge in nncf_graph.get_all_edges():
        portDesc = PortDesc(
            port_id=str(edge.input_port_id),
            precision=convert_dummy_precision(edge.dtype),
            shape=edge.tensor_shape
        )

        xmlElement = portDesc.as_xml_element()

        assert xmlElement.tag == Tags.PORT
        assert xmlElement.attrib["id"] == str(edge.input_port_id)
        assert xmlElement.attrib["precision"] == convert_dummy_precision(edge.dtype)
        assert all([child.tag == Tags.DIM for child in xmlElement])
        assert all([str(edge_shape) == port_shape.text for edge_shape, port_shape in zip(edge.tensor_shape, xmlElement)])
        
