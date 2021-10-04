from nncf.common.graph import NNCFGraph
from nncf.experimental.onnx.graph.nncf_graph_builder import GraphConverter

from copy import deepcopy


class NNCFNetwork:
    def __init__(self, onnx_model):
        self.onnx_model = onnx_model
        self.onnx_nncf_model = deepcopy(onnx_model)
        self.nncf_graph = None # type: NNCFGraph
        self._original_graph = GraphConverter.create_nncf_graph(onnx_model)
