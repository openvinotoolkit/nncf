from abc import ABC
from abc import abstractmethod

from typing import TypeVar
from typing import Dict
from typing import Union

from enum import Enum

from nncf.experimental.post_training.api.engine import Engine

ModelType = TypeVar('ModelType')


class PostTrainingAlgorithms(Enum):
    MinMaxQuantization = 'min_max_quantization'
    BiasCorrection = 'bias_correction'
    PostTrainingQuantization = 'post_training_quantization'


class AlgorithmParameters(ABC):
    """
    Base class for Post-Training algorithms parameters.
    """

    @abstractmethod
    def to_json(self) -> Dict[str, Union[str, float, int]]:
        """
        Serializes algorithms parameters to JSON format.
        """


class Algorithm(ABC):
    """
    Base class for all Post-Training algorithms.
    """

    @abstractmethod
    def apply(self, model: ModelType, engine: Engine) -> ModelType:
        """
        Applies the algorithm to the 'compressed_model'.
        """
