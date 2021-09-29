from nncf.common.graph import NNCFGraph
from nncf.experimental.onnx.graph.graph_builder import GraphConverter


class NNCFNetwork:
    def __init__(self, onnx_model):
        self.onnx_model = onnx_model
        self.nncf_graph = None # type: NNCFGraph
        self._original_graph = GraphConverter.create_nncf_graph(onnx_model)
