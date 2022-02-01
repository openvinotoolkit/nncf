import onnx

from nncf.experimental.post_training.compressed_model import CompressedModel
from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.api.dataloader import DataLoader
from nncf.experimental.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.common.graph.graph import NNCFGraph


class ONNXCompressedModel(CompressedModel):
    def __init__(self, model: str, dataloader: DataLoader, engine: Engine):
        super().__init__(model, dataloader, engine)

    def build_nncf_graph(self, dataloader: DataLoader, engine: Engine) -> NNCFGraph:
        self.original_model = onnx.load(self.original_model)
        return GraphConverter.create_nncf_graph(self.original_model)

    def export(self, path: str):
        onnx.save_model(self.transformed_model, path)
