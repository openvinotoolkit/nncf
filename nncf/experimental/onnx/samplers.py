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

from typing import Union

import torch
import numpy as np

from nncf.common.utils.logger import logger as nncf_logger

from nncf.experimental.post_training.api.dataloader import DataLoader

from nncf.experimental.post_training.sampler import BatchSampler
from nncf.experimental.post_training.sampler import RandomBatchSampler


class ONNXBatchSampler(BatchSampler):
    def form_batch(self, start_i: int, end_i: int):
        tensors = []  # type: List[torch.tensor]
        targets = []  # type: List[int]
        for i in range(start_i, end_i):
            tensors.append(self.dataloader[i][0])
            targets.append(self.dataloader[i][1])

        if isinstance(tensors[0], torch.Tensor):
            return torch.stack(tensors), targets
        if isinstance(tensors[0], np.ndarray):
            return np.stack(tensors), targets
        raise RuntimeError('Unexpected input data type {tensors[0]}. Should be one of torch.Tensor or np.ndarray')


class ONNXRandomBatchSampler(RandomBatchSampler):
    def form_batch(self, start_i: int, end_i: int):
        tensors = []  # type: List[torch.tensor]
        targets = []  # type: List[int]
        for i in range(start_i, end_i):
            tensors.append(self.dataloader[self.random_permutated_indices[i]][0])
            targets.append(self.dataloader[self.random_permutated_indices[i]][1])

        if isinstance(tensors[0], torch.Tensor):
            return torch.stack(tensors), targets
        if isinstance(tensors[0], np.ndarray):
            return np.stack(tensors), targets
        raise RuntimeError('Unexpected input data type {tensors[0]}. Should be one of torch.Tensor or np.ndarray')


def create_onnx_sampler(dataloader: DataLoader) -> Union[ONNXBatchSampler, ONNXRandomBatchSampler]:
    if dataloader.shuffle:
        nncf_logger.info('Using Shuffled dataset')
        return ONNXRandomBatchSampler(dataloader)
    nncf_logger.info('Using Non-Shuffled dataset')
    return ONNXBatchSampler(dataloader)
