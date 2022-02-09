import torch

from nncf.experimental.post_training.sampler import BatchSampler


class ONNXBatchSampler(BatchSampler):
    def form_batch(self, start_i: int, end_i: int):
        tensors = []  # type: List[torch.tensor]
        targets = []  # type: List[int]
        for i in range(start_i, end_i):
            tensors.append(self.dataloader[i][0])
            targets.append(self.dataloader[i][1])

        return torch.stack(tensors), targets
