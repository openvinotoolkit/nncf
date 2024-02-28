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
    stat_subset_size: int
    ref_calibration_samples_num: int
    ref_iterations_num: int


class TemplateTestBatchSize(ABC):
    @abstractmethod
    def create_statistics_aggregator(self, dataset) -> StatisticsAggregator:
        pass

    def create_dataset(self, lenght, batch_size):
        dataset = get_static_dataset(None, None, None, lenght)
        dataset._data_source.batch_size = batch_size
        return dataset

    @pytest.mark.parametrize(
        ("test_data"),
        (
            [  # batch_size | dataset_len | stat_subset_size | ref_calibration_samples_num | ref_iterations_num
                # DataForTest(None, None, None, None, None),  # None is None
                DataForTest(1, 1000, 300, 300, 300),
                DataForTest(10, 1000, 300, 300, 30),
                DataForTest(300, 1000, 300, 300, 1),
                DataForTest(301, 1000, 300, 300, 0),  # batch_size > stat_subset_size
                DataForTest(10, 10, 300, 100, 10),  # len(dataset) * batch_size < subset_size
                DataForTest(11, 300, 300, 300, 27),  # stat_subset_size % batch_size != 0
            ]
        ),
    )
    def test_batch_size_subset_size_dataset_len(self, test_data):
        # Checks correct iterations number depending on batch_size, dataset length, subset_size
        batch_size, dataset_length, stat_subset_size, ref_calibration_samples_num, ref_iterations_num = (
            getattr(test_data, field.name) for field in fields(test_data)
        )
        dataset = self.create_dataset(dataset_length, batch_size)
        statistics_aggregator = self.create_statistics_aggregator(dataset)
        statistics_aggregator.stat_subset_size = stat_subset_size
        total_calibration_samples = statistics_aggregator._get_number_samples_for_statistics()
        assert total_calibration_samples == ref_calibration_samples_num
        iterataions_num = statistics_aggregator._get_iterations_num(total_calibration_samples)
        assert iterataions_num == ref_iterations_num
