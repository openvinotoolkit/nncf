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

from abc import abstractmethod

import pytest

from nncf.experimental.common.tensor_statistics.collectors import HAWQAggregator
from nncf.experimental.common.tensor_statistics.collectors import NoopReducer
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector


class TemplateTestMixedPrecisionAlgoBackend:
    @abstractmethod
    def get_hawq_with_backend(self, subset_size: int):
        """
        Returns an instance of the algorithm with the specified subset size.

        :param subset_size: The size of the subset to be used by the algorithm.
        :return: An instance of the algorithm.
        """

    @pytest.mark.parametrize("subset_size", (10, 1, None))
    def test_hawq_statistic_collector(self, subset_size: int):
        algo = self.get_hawq_with_backend(subset_size)
        collector = algo._get_statistic_collector()

        # Check if the collector is an instance of TensorCollector
        assert isinstance(collector, TensorCollector)

        # Test the aggregator
        assert len(collector.aggregators) == 1
        _, aggregator = collector.aggregators.popitem()
        assert isinstance(aggregator, HAWQAggregator)
        assert aggregator.num_samples == subset_size

        # Test the reducer
        assert len(collector.reducers) == 1
        reducer = collector.reducers.pop()
        assert isinstance(reducer, NoopReducer)
