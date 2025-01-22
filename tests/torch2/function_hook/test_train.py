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
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from tests.torch2.function_hook.helpers import get_wrapped_simple_model_with_hook


def run_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, use_cuda: bool):
    tensor = torch.ones((10, 1, 3, 3))
    if use_cuda:
        tensor = tensor.cuda()
    pred: torch.Tensor = model(tensor)
    loss = torch.tensor(100) - pred.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_train_simple(use_cuda):
    wrapped_model = get_wrapped_simple_model_with_hook()
    if use_cuda:
        wrapped_model = wrapped_model.cuda()
    optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=0.1)
    run_one_epoch(wrapped_model, optimizer, use_cuda)
    assert all(p.grad is not None for p in wrapped_model.parameters())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2xGPUs")
@pytest.mark.cuda
def test_train_data_parallel():
    wrapped_model = get_wrapped_simple_model_with_hook().cuda()
    optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=0.1)
    parallel_model = torch.nn.DataParallel(wrapped_model)
    parallel_model.to("cuda")
    run_one_epoch(parallel_model, optimizer, use_cuda=True)
    assert all(p.grad is not None for p in wrapped_model.parameters())


def _train_ddp(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}")
    model = get_wrapped_simple_model_with_hook().to(device)
    model = DDP(model, device_ids=[rank])
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    run_one_epoch(model, optimizer, use_cuda=True)
    dist.destroy_process_group()
    assert all(p.grad is not None for p in model.parameters())


@pytest.fixture
def _ddp_env():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    yield
    os.environ.pop("MASTER_ADDR")
    os.environ.pop("MASTER_PORT")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2xGPUs")
@pytest.mark.cuda
def test_train_distributed_data_parallel(_ddp_env):
    world_size = torch.cuda.device_count()
    mp.spawn(_train_ddp, args=(world_size,), nprocs=world_size, join=True)
