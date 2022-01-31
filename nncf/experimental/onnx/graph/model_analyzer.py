from nncf.experimental.post_training.graph.model_analyzer import ModelAnalyzer
from nncf.experimental.post_training.compressed_model import CompressedModel


class ONNXModelAnalyzer(ModelAnalyzer):

    def get_quantization_transformations(self, compressed_model: CompressedModel):
        ...

    def get_sparsity_transformations(self, compressed_model: CompressedModel):
        ...
