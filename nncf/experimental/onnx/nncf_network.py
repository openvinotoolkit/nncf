from copy import deepcopy
from nncf.experimental.onnx.graph.nncf_graph_builder import GraphConverter


class NNCFNetwork:
    def __init__(self, onnx_model):
        self.onnx_model = onnx_model
        self.onnx_compressed_model = deepcopy(onnx_model)
        self.nncf_graph = GraphConverter.create_nncf_graph(onnx_model)
