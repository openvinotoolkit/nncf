from abc import ABC
from abc import abstractmethod


class DataLoader(ABC):
    @abstractmethod
    def __getitem__(self, i):
        ...

    @abstractmethod
    def __len__(self):
        ...
