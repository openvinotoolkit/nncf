import os
import pytest

from nncf.experimental.onnx.graph.nncf_graph_builder import GraphConverter

from tests.onnx.models import LinearModel
from tests.onnx.models import MultiInputOutputModel
from tests.onnx.models import ModelWithIntEdges

import networkx as nx

TEST_MODELS = [LinearModel, MultiInputOutputModel, ModelWithIntEdges]
PROJECT_ROOT = os.path.dirname(__file__)
REFERENCE_GRAPHS_TEST_ROOT = 'data/reference_graphs'


def check_nx_graph(nx_graph: nx.DiGraph, graph_path: str, generate_ref_graphs: bool = False):
    if generate_ref_graphs:
        nx.drawing.nx_pydot.write_dot(nx_graph, graph_path)

    expected_graph = nx.drawing.nx_pydot.read_dot(graph_path)

    for nx_graph_node, expected_graph_node in zip(sorted(nx_graph.nodes.keys()), sorted(expected_graph.nodes.keys())):
        assert nx_graph_node == expected_graph_node

    # Check nodes attrs
    for node_name, node_attrs in nx_graph.nodes.items():
        expected_attrs = expected_graph.nodes[node_name]
        attrs = {k: str(v) for k, v in node_attrs.items()}
        assert expected_attrs == attrs

    assert nx.DiGraph(expected_graph).edges == nx_graph.edges

    # Check edges attrs
    for nx_graph_edges, expected_graph_edges in zip(nx_graph.edges.data(), expected_graph.edges.data()):
        for nx_edge_attrs, expected_graph_edge_attrs in zip(nx_graph_edges, expected_graph_edges):
            if isinstance(nx_edge_attrs, dict):
                nx_edge_attrs['label'] = str(nx_edge_attrs['label'])
                expected_graph_edge_attrs['label'] = expected_graph_edge_attrs['label'].replace('"', '')
            assert nx_edge_attrs == expected_graph_edge_attrs


@pytest.mark.parametrize("model_creator_func", TEST_MODELS)
def test_built_nncf_graphs(model_creator_func):
    model = model_creator_func()
    nncf_graph = GraphConverter.create_nncf_graph(model.onnx_model)

    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    data_dir = os.path.join(PROJECT_ROOT, REFERENCE_GRAPHS_TEST_ROOT)
    path_to_dot = os.path.abspath(os.path.join(data_dir, model.path_ref_graph))

    check_nx_graph(nx_graph, path_to_dot)
