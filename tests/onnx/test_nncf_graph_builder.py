import os
import pytest

from nncf.experimental.onnx.graph.nncf_graph_builder import GraphConverter

from tests.onnx.models import LinearModel
from tests.onnx.models import MultiInputOutputModel

import networkx as nx

TEST_MODELS = [LinearModel(), MultiInputOutputModel()]


@pytest.mark.parametrize("model", TEST_MODELS)
def test_built_nncf_graphs(model):
    nncf_graph = GraphConverter.create_nncf_graph(model.onnx_model)

    nx_graph = nncf_graph.get_graph_for_structure_analysis()

    data_dir = os.path.join(os.path.dirname(__file__), 'data/reference_graphs')
    path_to_dot = os.path.abspath(os.path.join(data_dir, model.path_ref_graph))

    load_graph = nx.drawing.nx_pydot.read_dot(path_to_dot)

    for k, attrs in nx_graph.nodes.items():
        attrs = {k: str(v) for k, v in attrs.items()}
        load_attrs = {k: str(v).strip('"') for k, v in load_graph.nodes[k].items()}
        if attrs != load_attrs:
            assert attrs == load_attrs

    assert load_graph.nodes.keys() == nx_graph.nodes.keys()
    assert nx.DiGraph(load_graph).edges == nx_graph.edges
