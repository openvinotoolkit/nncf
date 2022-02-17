from abc import ABC
from abc import abstractmethod

from enum import Enum

from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.compressed_model import CompressedModel


class PostTrainingAlgorithms(Enum):
    QuantizerRangeFinder = 'quantizer_range_finder'
    BiasCorrection = 'bias_correction'
    PostTrainingQuantization = 'post_training_quantization'


class AlgorithmParameters(ABC):
    def __init__(self):
        """

        """


class Algorithm(ABC):
    @abstractmethod
    def apply(self, compressed_model: CompressedModel, engine: Engine) -> CompressedModel:
        """
        Applies the algorithm to the 'compressed_model'.
        """
