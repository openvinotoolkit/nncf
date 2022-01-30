from abc import ABC
from abc import abstractmethod

from typing import List
from typing import TypeVar

ModelType = TypeVar('ModelType')
TensorType = TypeVar('TensorType')


class Engine(ABC):
    """
    The basic class aims to provide the interface to infer the model.
    """

    def set_model(self, model: ModelType) -> None:
        self.model = model

    @abstractmethod
    def infer(self, input_data: List[TensorType]) -> List[TensorType]:
        """
        Infer the model on the provided input_data.
        """
