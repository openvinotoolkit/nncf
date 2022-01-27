from onnx import ModelProto

from nncf.common.graph.graph import NNCFGraph
from nncf.experimental.post_training_api.compressed_model import CompressedModel
from nncf.experimental.onnx.graph.nncf_graph_builder import GraphConverter


class ONNXCompressedModel(CompressedModel):
    def __init__(self, model: ModelProto):
        super().__init__(model)

    def build_nncf_graph(self) -> NNCFGraph:
        return GraphConverter.create_nncf_graph(self.original_model)
