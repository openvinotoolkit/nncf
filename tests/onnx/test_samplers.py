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
from typing import Tuple

import pytest

import numpy as np

from nncf.experimental.onnx.samplers import ONNXBatchSampler
from nncf.experimental.onnx.samplers import ONNXRandomBatchSampler
from nncf.experimental.post_training.api.dataset import Dataset

INPUT_SHAPE = [3, 10, 10]

DATASET_SAMPLES = [(np.zeros(INPUT_SHAPE), 0),
                   (np.ones(INPUT_SHAPE), 1),
                   (100 * np.ones(INPUT_SHAPE), 2)]


class TestDataset(Dataset):
    def __init__(self, samples: List[Tuple[np.ndarray, int]]):
        super().__init__(shuffle=False)
        self.samples = samples

    def __getitem__(self, item):
        return self.samples[item]

    def __len__(self):
        return 3


@pytest.mark.parametrize("batch_size", (1, 2, 3))
def test_batch_sampler(batch_size):
    dataset = TestDataset(DATASET_SAMPLES)
    dataset.batch_size = batch_size
    sampler = ONNXBatchSampler(dataset)
    for i, sample in enumerate(sampler):
        ref_sample = []
        ref_target = []
        for j in range(i * batch_size, i * batch_size + batch_size):
            ref_sample.extend([DATASET_SAMPLES[j][0]])
            ref_target.extend([DATASET_SAMPLES[j][1]])
        ref_sample = np.stack(ref_sample)
        ref_target = np.stack(ref_target)
        assert np.array_equal(sample[0], ref_sample)
        assert np.array_equal(sample[1], ref_target)


@pytest.mark.parametrize("batch_size", (1, 2, 3))
def test_random_batch_sampler(batch_size):
    np.random.seed(0)
    dataset = TestDataset(DATASET_SAMPLES)
    dataset.batch_size = batch_size
    sampler = ONNXRandomBatchSampler(dataset)
    random_permuated_indices = [0, 2, 1]
    for i, sample in enumerate(sampler):
        ref_sample = []
        ref_target = []
        for j in range(i * batch_size, i * batch_size + batch_size):
            ref_sample.extend([DATASET_SAMPLES[random_permuated_indices[j]][0]])
            ref_target.extend([DATASET_SAMPLES[random_permuated_indices[j]][1]])
        ref_sample = np.stack(ref_sample)
        ref_target = np.stack(ref_target)
        assert np.array_equal(sample[0], ref_sample)
        assert np.array_equal(sample[1], ref_target)
