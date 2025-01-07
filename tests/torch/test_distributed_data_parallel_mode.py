# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import Tuple

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from nncf import NNCFConfig
from nncf.torch import create_compressed_model
from nncf.torch import register_default_init_args
from tests.torch.helpers import create_random_mock_dataloader


class ModelWithChangedTrain(nn.Module):
    def __init__(
        self, in_out_channels: Tuple[Tuple[int, int]] = ((1, 3), (3, 5), (5, 7), (7, 10)), freezing_stages: int = -1
    ):
        super().__init__()
        self.freezing_stages = freezing_stages
        self.features = nn.ModuleList()
        for in_out_ch in in_out_channels:
            block = nn.ModuleList()
            block.append(nn.Conv2d(*in_out_ch, 3))
            block.append(nn.BatchNorm2d(in_out_ch[1]))
            block.append(nn.ReLU())
            self.features.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blocks in self.features:
            for module in blocks:
                x = module(x)
        return x

    def train(self: nn.Module, mode: bool = True) -> nn.Module:
        super().train(mode)
        for i in range(self.freezing_stages):
            for module in self.features[i]:
                for p in module.parameters():
                    p.requires_grad = False


def worker(rank: int, world_size: int) -> None:
    torch.distributed.init_process_group(
        backend="nccl", init_method="tcp://127.0.0.1:8999", world_size=world_size, rank=rank
    )
    model = ModelWithChangedTrain(freezing_stages=1)
    model.cuda()
    model.to(rank)

    nncf_config = NNCFConfig()
    nncf_config.update(
        {
            "input_info": {"sample_size": [1, 1, 30, 30]},
            "compression": {
                "algorithm": "quantization",
                "initializer": {
                    "range": {"num_init_samples": 10},
                    "batchnorm_adaptation": {"num_bn_adaptation_samples": 10},
                },
            },
        }
    )
    dataloader = create_random_mock_dataloader(nncf_config, num_samples=10)
    register_default_init_args(nncf_config, dataloader)

    _, compressed_model = create_compressed_model(model, nncf_config)

    # At this part the additional processes may be freezing

    _ = torch.nn.parallel.DistributedDataParallel(compressed_model, device_ids=[rank])


@pytest.mark.cuda
@pytest.mark.parametrize("waiting_time", [20.0])
def test_is_ddp_freezing(waiting_time: float) -> None:
    # Number of processes the same as GPU count
    n_procs = torch.cuda.device_count()
    ctx = mp.spawn(fn=worker, args=(n_procs,), nprocs=n_procs, join=False)

    start_time = time.monotonic()
    while not ctx.join(waiting_time):
        current_time = time.monotonic()
        if current_time - start_time >= waiting_time:
            for process in ctx.processes:
                if process.is_alive():
                    process.terminate()
            raise TimeoutError("DDP wrapper may be freezing")
