from abc import ABC
from abc import abstractmethod

from typing import TypeVar
from typing import Dict
from typing import Union

from enum import Enum
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
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
    def _apply(self, model: ModelType, engine: Engine, layer_statistics) -> ModelType:
        pass

    @abstractmethod
    def get_layers_for_statistics(self, model: ModelType) -> Dict[str, TensorStatisticCollectorBase]:
        """
        Returns activations layers, for which StatisticsCollector should collect statistics.
        """

    def apply(self, model: ModelType, engine: Engine, layer_statistics: Dict[str, TensorStatisticCollectorBase]) -> ModelType:
        """
        Applies the algorithm to the 'compressed_model'.
        """
        layers = self.get_layers_for_statistics(model)
        for layer in layers.keys():
            if layer_statistics.get(layer) is None:
                raise RuntimeError('No statistics collected for the layer {layer}')
        return self._apply(model, engine, layer_statistics)
