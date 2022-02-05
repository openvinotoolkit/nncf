from abc import ABC
from abc import abstractmethod


class Sampler(ABC):
    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass


class RandomSampler(Sampler):
    def __iter__(self):
        pass
