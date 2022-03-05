from abc import ABC
from abc import abstractmethod

from typing import TypeVar

from enum import Enum

from nncf.experimental.post_training.api.engine import Engine

ModelType = TypeVar('ModelType')


class PostTrainingAlgorithms(Enum):
    MinMaxQuantization = 'min_max_quantization'
    BiasCorrection = 'bias_correction'
    PostTrainingQuantization = 'post_training_quantization'


class AlgorithmParameters(ABC):
    """
    """


class Algorithm(ABC):
    @abstractmethod
    def apply(self, model: ModelType, engine: Engine) -> ModelType:
        """
        Applies the algorithm to the 'compressed_model'.
        """
