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

from typing import List, Tuple, Union
import torch
import numpy as np

from abc import abstractmethod
from nncf.experimental.post_training.api.sampler import Sampler
from nncf.experimental.post_training.api.dataset import Dataset

import random

SAMPLER_OUTPUT_TYPE = Union[torch.Tensor, np.ndarray]

# TODO (Nikita Malinin): Replace or rename this file
class BatchSampler(Sampler):
    """
    Base class for dataset sampler forms a batch from samples
    with batch_size determined in dataset instance.
    """

    def __iter__(self):
        for i in range(len(self.batch_indices) - 1):
            batch = self.form_batch(
                self.batch_indices[i], self.batch_indices[i + 1])
            yield batch

    def __len__(self):
        return len(self.dataset)

    @abstractmethod
    def form_batch(self, start_i: int, end_i: int):
        pass

    def _post_process(self, tensors: List, targets: List) -> Tuple[SAMPLER_OUTPUT_TYPE, SAMPLER_OUTPUT_TYPE]:
        if isinstance(tensors[0], torch.Tensor):
            return torch.stack(tensors), torch.LongTensor(targets)
        if isinstance(tensors[0], np.ndarray):
            return np.stack(tensors), np.array(targets)
        raise RuntimeError(
            'Unexpected input data type {tensors[0]}. Should be one of torch.Tensor or np.ndarray')


class RandomBatchSampler(BatchSampler):
    """
    Base class for dataset sampler forms a batch from randomly shuffled samples
    with batch_size determined in dataset instance.
    """

    def __init__(self, dataset: Dataset, seed: int = 0, sample_indices=None):
        super().__init__(dataset, sample_indices)
        random.seed(seed)
        self.random_permutated_indices = list(range(len(self.dataset)))
        random.shuffle(self.random_permutated_indices)

    def __iter__(self):
        for i in range(len(self.batch_indices) - 1):
            batch = self.form_batch(
                self.batch_indices[i], self.batch_indices[i + 1])
            yield batch
