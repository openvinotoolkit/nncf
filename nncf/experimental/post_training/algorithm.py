from abc import ABC
from abc import abstractmethod

from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.compressed_model import CompressedModel


class PostTrainingAlgorithm(ABC):
    @abstractmethod
    def apply(self, compressed_model: CompressedModel, engine: Engine) -> CompressedModel:
        """
        Applies the algorithm to the 'compressed_model'.
        """
