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

from math import ceil

import pytest

from nncf.torch.initialization import PartialDataLoader
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.helpers import get_empty_config

N_SAMPLE = 10
VALID_RATIO = [0.0, 0.05, 0.2, 0.51, 0.89, 1.0]
INVALID_RATIO = ["string", "0.5", -17, -0.01, 1.01, 5, 100]


def create_regular_dataloader():
    return create_ones_mock_dataloader(config=get_empty_config(), num_samples=N_SAMPLE)


def test_can_create_partial_dataloader__with_defaults():
    dataloader = create_regular_dataloader()
    partial_dataloader = PartialDataLoader(dataloader)
    assert partial_dataloader.data_loader is dataloader
    assert partial_dataloader.batch_size == 1
    assert len(partial_dataloader) == N_SAMPLE


@pytest.mark.parametrize("invalid_ratio", INVALID_RATIO, ids=[str(r) for r in INVALID_RATIO])
def test_partial_dataloader__with_invalid_ratio(invalid_ratio):
    dataloader = create_regular_dataloader()
    with pytest.raises((ValueError, TypeError)):
        _ = PartialDataLoader(dataloader, iter_ratio=invalid_ratio)


@pytest.mark.parametrize("valid_ratio", VALID_RATIO, ids=[str(r) for r in VALID_RATIO])
def test_partial_dataloader__with_valid_ratio(valid_ratio):
    dataloader = create_regular_dataloader()
    partial_dataloader = PartialDataLoader(dataloader, iter_ratio=valid_ratio)
    truth_num_batch = ceil(valid_ratio * ceil(N_SAMPLE / dataloader.batch_size))
    assert len(partial_dataloader) == truth_num_batch

    i = 0
    for i, _ in enumerate(partial_dataloader):
        pass

    if valid_ratio == 0:
        assert truth_num_batch == 0
    else:
        assert (i + 1) == truth_num_batch
