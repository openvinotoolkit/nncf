from abc import ABC
from abc import abstractmethod

from typing import TypeVar

from nncf.experimental.post_training_api.dataloader import DataLoader

ModelType = TypeVar('ModelType')


class Engine(ABC):
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.statistics = None

    def set_model(self, model: ModelType) -> None:
        self.model = model

    @abstractmethod
    def infer(self):
        ...
