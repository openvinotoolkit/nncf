import pytest
from nncf.onnx.graph.onnx_graph import ONNXGraph
from tests.onnx.models import MatMulWeightModel
from tests.onnx.models import MatMulWeightModel_2


@pytest.mark.parametrize('model', [MatMulWeightModel(), MatMulWeightModel_2()])
def test_is_node_with_weight(model):
    onnx_graph = ONNXGraph(model.onnx_model)
    node = onnx_graph.get_node_by_name("MatMul")
    onnx_graph.is_node_with_weight(node, 0)
    onnx_graph.is_node_with_weight(node, 1)
