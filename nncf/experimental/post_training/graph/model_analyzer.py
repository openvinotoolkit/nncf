from abc import ABC
from abc import abstractmethod

from nncf.experimental.post_training.compressed_model import CompressedModel


class ModelAnalyzer(ABC):

    @abstractmethod
    def get_quantization_transformations(self, compressed_model: CompressedModel):
        ...

    @abstractmethod
    def get_sparsity_transformations(self, compressed_model: CompressedModel):
        ...
