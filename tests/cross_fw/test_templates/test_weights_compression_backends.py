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

from abc import abstractmethod

import pytest

from nncf.experimental.common.tensor_statistics.collectors import HAWQAggregator
from nncf.experimental.common.tensor_statistics.collectors import MaxVarianceReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanAbsMaxReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanAggregator
from nncf.experimental.common.tensor_statistics.collectors import MeanVarianceReducer
from nncf.experimental.common.tensor_statistics.collectors import RawReducer
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector


class TemplateTestMixedPrecisionAlgoBackend:
    @abstractmethod
    def get_hawq_with_backend(self, subset_size: int):
        """Returns a HAWQ instance of the algorithm."""

    @abstractmethod
    def get_mean_variance_with_backend(self, subset_size: int):
        """Returns a Mean Variance instance of the algorithm."""

    @abstractmethod
    def get_max_variance_with_backend(self, subset_size: int):
        """Returns a Max Variance instance of the algorithm."""

    @abstractmethod
    def get_mean_max_with_backend(self, subset_size: int):
        """Returns a Mean Max instance of the algorithm."""

    def check_aggregator(self, collector: TensorCollector, expected_aggregator_type, subset_size: int):
        assert len(collector.aggregators) == 1, "Collector should have exactly one aggregator."
        _, aggregator = collector.aggregators.popitem()
        assert isinstance(
            aggregator, expected_aggregator_type
        ), f"Expected aggregator of type {expected_aggregator_type.__name__}, got {type(aggregator).__name__}."
        assert aggregator.num_samples == subset_size, "Aggregator num_samples does not match the provided subset size."

    def check_reducer(self, collector: TensorCollector, expected_reducer_type):
        assert len(collector.reducers) == 1
        reducer = collector.reducers.pop()
        assert isinstance(
            reducer, expected_reducer_type
        ), f"Expected reducer of type {expected_reducer_type.__name__}, got {type(reducer).__name__}."

    @pytest.mark.parametrize("subset_size", [1, 10, None])
    @pytest.mark.parametrize(
        "algo_func, aggregator_type, reducer_type",
        [
            ("get_hawq_with_backend", HAWQAggregator, RawReducer),
            ("get_mean_variance_with_backend", MeanAggregator, MeanVarianceReducer),
            ("get_max_variance_with_backend", MeanAggregator, MaxVarianceReducer),
            ("get_mean_max_with_backend", MeanAggregator, MeanAbsMaxReducer),
        ],
    )
    def test_statistic_collector(self, subset_size, algo_func, aggregator_type, reducer_type):
        """Test function to validate statistic collectors."""
        algo = getattr(self, algo_func)(subset_size)
        collector = algo._get_statistic_collector()

        # Verify the collector instance and properties
        assert isinstance(collector, TensorCollector), "Collector is not an instance of TensorCollector."

        # Validate the aggregator and reducer types
        self.check_aggregator(collector, aggregator_type, subset_size)
        self.check_reducer(collector, reducer_type)
