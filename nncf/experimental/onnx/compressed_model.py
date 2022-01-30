import onnx

from nncf.experimental.post_training_api.compressed_model import CompressedModel
from nncf.experimental.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.common.graph.graph import NNCFGraph


class ONNXCompressedModel(CompressedModel):
    def __init__(self, model: str):
        super().__init__(model)

    def build_nncf_graph(self) -> NNCFGraph:
        model = onnx.load(self.original_model)
        return GraphConverter.create_nncf_graph(model)
