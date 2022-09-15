import pytest

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.pruning.utils import is_batched_linear


@pytest.mark.parametrize('batched,has_output_edges,res', [(False, True, False),
                                                          (True, True, True),
                                                          (True, False, False)])
def test_is_batched_linear(batched, has_output_edges, res):
    graph = NNCFGraph()
    linear = graph.add_nncf_node('linear', 'linear', 'linear', LinearLayerAttributes(True, 5, 5))
    if has_output_edges:
        last_linear = graph.add_nncf_node('last_linear', 'linear', 'linear', LinearLayerAttributes(True, 5, 5))
        tensor_shape = [5, 5] if not batched else [5, 5, 5]
        graph.add_edge_between_nncf_nodes(linear.node_id, last_linear.node_id, tensor_shape, 0, 0, Dtype.FLOAT)
    assert is_batched_linear(linear, graph) == res
