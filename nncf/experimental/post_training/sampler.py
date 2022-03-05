from abc import ABC
from abc import abstractmethod

import random

from nncf.experimental.post_training.api.dataloader import DataLoader


class Sampler(ABC):
    """
    Base class for dataset sampler.
    """

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass


class BatchSampler(Sampler):
    """
    Base class for dataset sampler forms a batch from samples with batch_size determined in dataloader.
    """

    def __init__(self, dataloader: DataLoader):
        super().__init__(dataloader)
        self.indices = list(range(len(self.dataloader)))
        self.batch_size = dataloader.batch_size

    def __iter__(self):
        batch_indices = list(range(0, len(self.dataloader), self.batch_size))
        for i in range(len(batch_indices) - 1):
            batch = self.form_batch(batch_indices[i], batch_indices[i + 1])
            yield batch

    def __len__(self):
        return len(self.dataloader)

    @abstractmethod
    def form_batch(self, start_i: int, end_i: int):
        pass


class RandomBatchSampler(BatchSampler):
    """
    Base class for dataset sampler forms a batch from randomly shuffled samples
    with batch_size determined in dataloader.
    """

    def __init__(self, dataloader: DataLoader, seed: int = 0):
        super().__init__(dataloader)
        random.seed(seed)
        self.random_permutated_indices = list(range(len(self.dataloader)))
        random.shuffle(self.random_permutated_indices)

    def __iter__(self):
        batch_indices = list(range(0, len(self.dataloader), self.batch_size))
        for i in range(len(batch_indices) - 1):
            batch = self.form_batch(batch_indices[i], batch_indices[i + 1])
            yield batch
