"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

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
        self.batch_size = dataloader.batch_size
        self.batch_indices = list(range(0, len(self.dataloader) + 1, self.batch_size))

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

    def __iter__(self):
        for i in range(len(self.batch_indices) - 1):
            batch = self.form_batch(self.batch_indices[i], self.batch_indices[i + 1])
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
        for i in range(len(self.batch_indices) - 1):
            batch = self.form_batch(self.batch_indices[i], self.batch_indices[i + 1])
            yield batch
