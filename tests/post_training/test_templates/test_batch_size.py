# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import fields

import pytest

from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from tests.post_training.test_templates.helpers import get_static_dataset


@dataclass
class DataForTest:
    batch_size: int
    dataset_len: int
    subset_size: int
    ref_calibration_samples_num: int
    ref_iterations_num: int


class TemplateTestBatchSize(ABC):
    @abstractmethod
    def create_statistics_aggregator(self, dataset) -> StatisticsAggregator:
        ...

    def create_dataset(self, lenght, batch_size):
        dataset = get_static_dataset(None, None, None, lenght)
        dataset._data_source.batch_size = batch_size
        # print(dataset.get_batch_size())
        return dataset

    @pytest.mark.parametrize(
        ("test_data"),
        (
            [
                # DataForTest(None, 1000, None, None, None),
                # DataForTest(1, 1000, 300, 300, 300),
                # DataForTest(10, 1000, 300, 300, 30),
                # DataForTest(300, 1000, 300, 300, 1),
                # DataForTest(301, 1000, 300, 300, 0),
                # DataForTest(301, 1000, 300, 300, 0),  # batch_size > subset_size
                DataForTest(300, 10, 300, 10, 0),  # batch_size > len(dataset)
                # DataForTest(300, 10, 300, 10, 0),  # batch_size > len(dataset)
            ]
        ),
    )
    def test_batch_size_subset(self, test_data):
        batch_size, dataset_length, subset_size, ref_calibration_samples_num, ref_iterations_num = (
            getattr(test_data, field.name) for field in fields(test_data)
        )
        dataset = self.create_dataset(dataset_length, batch_size)
        statistics_aggregator = self.create_statistics_aggregator(dataset)
        statistics_aggregator.stat_subset_size = subset_size
        print(statistics_aggregator.dataset_size)
        calibration_samples_num = statistics_aggregator._get_total_calibration_samples()
        assert calibration_samples_num == ref_calibration_samples_num
        iterataions_num = statistics_aggregator._get_iterations_num(calibration_samples_num)
        assert iterataions_num == ref_iterations_num
