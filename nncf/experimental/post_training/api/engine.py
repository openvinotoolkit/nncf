from abc import ABC
from abc import abstractmethod

from typing import Dict
from typing import TypeVar

from nncf.experimental.post_training.api.dataloader import DataLoader

ModelType = TypeVar('ModelType')
TensorType = TypeVar('TensorType')


class Engine(ABC):
    """
    The basic class aims to provide the interface to infer the model.
    """

    def __init__(self, dataloader: DataLoader = None):
        self.dataloader = dataloader
        self.model = None

    def set_model(self, model: ModelType) -> None:
        self.model = model

    def is_model_set(self) -> bool:
        return self.model is not None

    def infer(self, _input: TensorType) -> Dict[str, TensorType]:
        if not self.is_model_set():
            raise RuntimeError('The {} tried to infer the model, while the model was not set.'.format(self.__class__))
        return self._infer(_input)

    @abstractmethod
    def _infer(self, _input: TensorType) -> Dict[str, TensorType]:
        """
        Infer the model on the provided input.
        Returns the model outputs and corresponding node names in the model.
        """
