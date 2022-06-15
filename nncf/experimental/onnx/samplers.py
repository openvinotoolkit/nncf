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

from typing import List
from typing import Union

import torch

from nncf.common.utils.logger import logger as nncf_logger

from nncf.experimental.post_training.api.dataset import Dataset

from nncf.experimental.post_training.samplers import BatchSampler
from nncf.experimental.post_training.samplers import RandomBatchSampler


class ONNXBatchSampler(BatchSampler):
    def form_batch(self, start_i: int, end_i: int):
        tensors = []  # type: List[torch.tensor]
        targets = []  # type: List[int]
        for i in range(start_i, end_i):
            tensors.append(self.dataset[i][0])
            targets.append(self.dataset[i][1])

        return self._post_process(tensors, targets)


class ONNXRandomBatchSampler(RandomBatchSampler):
    def form_batch(self, start_i: int, end_i: int):
        tensors = []  # type: List[torch.tensor]
        targets = []  # type: List[int]
        for i in range(start_i, end_i):
            tensors.append(self.dataset[self.random_permutated_indices[i]][0])
            targets.append(self.dataset[self.random_permutated_indices[i]][1])

        return self._post_process(tensors, targets)


def create_onnx_sampler(dataset: Dataset,
                        sample_indices: List) -> Union[ONNXBatchSampler, ONNXRandomBatchSampler]:
    if dataset.shuffle:
        nncf_logger.info('Using Shuffled dataset')
        return ONNXRandomBatchSampler(dataset, sample_indices=sample_indices)
    nncf_logger.info('Using Non-Shuffled dataset')
    return ONNXBatchSampler(dataset, sample_indices=sample_indices)
