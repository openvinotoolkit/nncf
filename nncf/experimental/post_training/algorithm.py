from abc import ABC
from abc import abstractmethod

from nncf.experimental.post_training.api.dataloader import DataLoader
from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.compressed_model import CompressedModel


class PostTrainingAlgorithm(ABC):
    def __init__(self, **kwargs):
        self.priority = None  # Priority of algorithms application set by CompressionBuilder

    @abstractmethod
    def apply(self, compressed_model: CompressedModel, dataloader: DataLoader, engine: Engine) -> CompressedModel:
        """
        Applies the algorithm to the 'compressed_model'.
        """

    def __lt__(self, other):
        return self.priority < other.priority

    def __gt__(self, other):
        return self.priority > other.priority

    def __eq__(self, other):
        return self.priority == other.priority
