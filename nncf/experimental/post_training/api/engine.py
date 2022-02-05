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
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def set_model(self, model: ModelType) -> None:
        self.model = model

    @abstractmethod
    def infer_model(self, i: int) -> List[TensorType]:
        """
        Infer the model on the i-th sample of the dataset
        """
