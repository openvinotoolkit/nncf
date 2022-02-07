from abc import ABC
from abc import abstractmethod

from typing import Tuple
from typing import Dict
from typing import TypeVar

from nncf.experimental.post_training.api.dataloader import DataLoader

ModelType = TypeVar('ModelType')
TensorType = TypeVar('TensorType')


class Engine(ABC):
    """
    The basic class aims to provide the interface to infer the model.
    """

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader

    def set_model(self, model: ModelType) -> None:
        self.model = model

    @abstractmethod
    def infer(self, i: int) -> Tuple[Dict[str, TensorType], TensorType]:
        """
        Infer the model on the i-th sample of the dataset
        """
