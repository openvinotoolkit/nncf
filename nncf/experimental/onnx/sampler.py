from typing import Union

import torch
import numpy as np

from nncf.experimental.onnx.engine import ONNXEngine

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


def create_onnx_sampler(engine: ONNXEngine) -> Union[ONNXBatchSampler, ONNXRandomBatchSampler]:
    if engine.dataloader.shuffle:
        print('Using Shuffled dataset')
        return ONNXRandomBatchSampler(engine.dataloader)
    else:
        print('Using Non-Shuffled dataset')
        return ONNXBatchSampler(engine.dataloader)
