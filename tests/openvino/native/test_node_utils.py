
from nncf.experimental.openvino_native.graph.node_utils import is_node_with_bias
from nncf.experimental.openvino_native.graph.nncf_graph_builder import GraphConverter

from tests.openvino.native.models import ConvNotBiasModel
from tests.openvino.native.models import ConvModel


def test_is_node_with_bias():
    model = ConvNotBiasModel().ov_model
    nncf_graph = GraphConverter.create_nncf_graph(model)
    node = nncf_graph.get_node_by_name('Conv')
    assert not is_node_with_bias(node, nncf_graph)
    
    model = ConvModel().ov_model
    nncf_graph = GraphConverter.create_nncf_graph(model)
    node = nncf_graph.get_node_by_name('Conv')
    assert is_node_with_bias(node, nncf_graph)