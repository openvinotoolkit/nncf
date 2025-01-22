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

from functools import partial
from typing import Dict, Tuple, Type

import pytest
import tensorflow as tf

from nncf.common.tensor_statistics.collectors import OfflineTensorStatisticCollector
from nncf.common.tensor_statistics.collectors import ReductionAxes
from nncf.common.tensor_statistics.collectors import StatisticsNotCollectedError
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.tensor_statistics.statistics import TensorStatistic
from nncf.tensorflow.tensor_statistics.collectors import TFMeanMinMaxStatisticCollector
from nncf.tensorflow.tensor_statistics.collectors import TFMeanPercentileStatisticCollector
from nncf.tensorflow.tensor_statistics.collectors import TFMedianMADStatisticCollector
from nncf.tensorflow.tensor_statistics.collectors import TFMinMaxStatisticCollector
from nncf.tensorflow.tensor_statistics.collectors import TFMixedMinMaxStatisticCollector
from nncf.tensorflow.tensor_statistics.collectors import TFPercentileStatisticCollector
from nncf.tensorflow.tensor_statistics.statistics import TFMedianMADTensorStatistic
from nncf.tensorflow.tensor_statistics.statistics import TFMinMaxTensorStatistic
from nncf.tensorflow.tensor_statistics.statistics import TFPercentileTensorStatistic


class TestCollectedStatistics:
    REF_INPUTS = [
        tf.constant([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]]),
        tf.constant([[4.5, 2.6, 3.7], [-1.3, -4, -3.5], [4.3, 5.8, 6.1]]),
    ]

    @pytest.mark.parametrize(
        ("collector", "reduction_shapes_vs_ref_statistic"),
        [
            (
                TFMinMaxStatisticCollector,
                {
                    (0, 1): TFMinMaxTensorStatistic(min_values=tf.constant(-4.0), max_values=tf.constant(6.1)),
                    (1,): TFMinMaxTensorStatistic(
                        min_values=tf.constant([1.0, -4.0, 4.0]), max_values=tf.constant([4.5, 4.0, 6.1])
                    ),
                    (0,): TFMinMaxTensorStatistic(
                        min_values=tf.constant([[-1.3, -4.0, -3.5]]), max_values=tf.constant([[4.5, 5.8, 6.1]])
                    ),
                    # Not supported for now:
                    # ((3, 3), ): PTTFMinMaxTensorStatistic(
                    #     min_values=tf.constant([
                    #         [1.0, 2.0, 3.0],
                    #         [-1.3, -4, -3.5],
                    #         [4.0, 5.0, 6.0]
                    #     ]),
                    #     max_values=tf.constant([
                    #         [4.5, 2.6, 3.7],
                    #         [1.3, 4.0, 3.5],
                    #         [4.3, 5.8, 6.1]
                    #     ]),
                    # ),
                },
            ),
            (
                partial(TFMeanMinMaxStatisticCollector, use_per_sample_stats=False),
                {
                    (0, 1): TFMinMaxTensorStatistic(min_values=tf.constant(-3.5), max_values=tf.constant(6.05)),
                    (1,): TFMinMaxTensorStatistic(
                        min_values=tf.constant([1.8, -3.5, 4.15]), max_values=tf.constant([3.75, 3.5, 6.05])
                    ),
                    (0,): TFMinMaxTensorStatistic(
                        min_values=tf.constant([[-1.15, -3, -3.25]]), max_values=tf.constant([[4.25, 5.4, 6.05]])
                    ),
                },
            ),
            (
                partial(
                    TFMixedMinMaxStatisticCollector,
                    use_per_sample_stats=False,
                    use_means_of_mins=False,
                    use_means_of_maxs=True,
                ),
                {
                    (0, 1): TFMinMaxTensorStatistic(min_values=tf.constant(-4.0), max_values=tf.constant(6.05)),
                    (1,): TFMinMaxTensorStatistic(
                        min_values=tf.constant([1.0, -4.0, 4.0]), max_values=tf.constant([3.75, 3.5, 6.05])
                    ),
                    (0,): TFMinMaxTensorStatistic(
                        min_values=tf.constant([[-1.3, -4.0, -3.5]]), max_values=tf.constant([[4.25, 5.4, 6.05]])
                    ),
                },
            ),
        ],
    )
    def test_collected_statistics_with_shape_convert(
        self,
        collector: Type[TensorStatisticCollectorBase],
        reduction_shapes_vs_ref_statistic: Dict[Tuple[ReductionAxes, ReductionAxes], TensorStatistic],
    ):
        for reduction_shape in reduction_shapes_vs_ref_statistic:
            collector_obj = collector(use_abs_max=True, reduction_shape=reduction_shape)
            for input_ in TestCollectedStatistics.REF_INPUTS:
                collector_obj.register_input(input_)
            test_stats = collector_obj.get_statistics()
            assert reduction_shapes_vs_ref_statistic[reduction_shape] == test_stats

    @pytest.mark.parametrize(
        ("collector", "reduction_shapes_vs_ref_statistic"),
        [
            (
                TFMedianMADStatisticCollector,
                {
                    (0, 1): TFMedianMADTensorStatistic(median_values=tf.constant([2.8]), mad_values=tf.constant([2.6])),
                    (1,): TFMedianMADTensorStatistic(
                        median_values=tf.constant([2.8, -2.5, 5.4]), mad_values=tf.constant([0.85, 1.1, 0.65])
                    ),
                    (0,): TFMedianMADTensorStatistic(
                        median_values=tf.constant([[2.5, 2.3, 3.35]]), mad_values=tf.constant([[1.9, 3.1, 2.7]])
                    ),
                    # Not supported for now:
                    # (3, 3): TFMedianMADTensorStatistic(
                    #     median_values=tf.constant([
                    #         [2.75, 2.3, 3.35],
                    #         [-1.15, -3, -3.25],
                    #         [4.15, 5.4, 6.05]
                    #     ]),
                    #     mad_values=tf.constant([
                    #         [1.75, 0.3, 0.35],
                    #         [0.15, 1, 0.25],
                    #         [0.15, 0.4, 0.05]
                    #     ]),
                    # ),
                },
            ),
            (
                partial(TFPercentileStatisticCollector, percentiles_to_collect=[10.0]),
                {
                    (0, 1): TFPercentileTensorStatistic({10.0: tf.constant([-3.15])}),
                    (1,): TFPercentileTensorStatistic({10.0: tf.constant([1.5, -3.75, 4.15])}),
                    (0,): TFPercentileTensorStatistic({10.0: tf.constant([[-1.15, -3, -3.25]])}),
                    # Not supported for now:
                    # (3, 3): TFPercentileTensorStatistic(
                    #     {
                    #         10.0: tf.constant([
                    #             [1.35, 2.06, 3.07],
                    #             [-1.27, -3.8, -3.45],
                    #             [4.03, 5.08, 6.01]
                    #         ])
                    #     }
                    # ),
                },
            ),
            (
                partial(TFMeanPercentileStatisticCollector, percentiles_to_collect=[10.0]),
                {
                    (0, 1): TFPercentileTensorStatistic({10.0: tf.constant([-2.9])}),
                    (1,): TFPercentileTensorStatistic({10.0: tf.constant([[2.0100], [-3.3500], [4.4000]])}),
                    (0,): TFPercentileTensorStatistic({10.0: tf.constant([[-0.3900, -1.9400, -1.9300]])}),
                    # Not supported for now:
                    # (3, 3): TFPercentileTensorStatistic(
                    #     {
                    #         10.0: tf.constant([
                    #             [ 2.7500,  2.3000,  3.3500],
                    #             [-1.1500, -3.0000, -3.2500],
                    #             [ 4.1500,  5.4000,  6.0500]
                    #         ])
                    #     }
                    # ),
                },
            ),
        ],
    )
    def test_collected_statistics(
        self,
        collector: Type[TensorStatisticCollectorBase],
        reduction_shapes_vs_ref_statistic: Dict[ReductionAxes, TensorStatistic],
    ):
        for reduction_shape in reduction_shapes_vs_ref_statistic:
            collector_obj = collector(reduction_shape=reduction_shape)
            for input_ in TestCollectedStatistics.REF_INPUTS:
                collector_obj.register_input(input_)
            test_stats = collector_obj.get_statistics()
            assert reduction_shapes_vs_ref_statistic[reduction_shape] == test_stats

    COLLECTORS = [
        partial(TFMinMaxStatisticCollector, use_abs_max=False),
        partial(
            TFMixedMinMaxStatisticCollector,
            use_per_sample_stats=False,
            use_abs_max=False,
            use_means_of_mins=False,
            use_means_of_maxs=False,
        ),
        partial(TFMeanMinMaxStatisticCollector, use_per_sample_stats=False, use_abs_max=False),
        TFMedianMADStatisticCollector,
        partial(TFPercentileStatisticCollector, percentiles_to_collect=[10.0]),
        partial(TFMeanPercentileStatisticCollector, percentiles_to_collect=[10.0]),
    ]

    @pytest.fixture(params=COLLECTORS)
    def collector_for_interface_test(self, request):
        collector_type = request.param
        return collector_type(reduction_shape=(1,))

    def test_collected_samples(self, collector_for_interface_test: TensorStatisticCollectorBase):
        for input_ in TestCollectedStatistics.REF_INPUTS:
            collector_for_interface_test.register_input(input_)
        assert collector_for_interface_test.collected_samples() == len(TestCollectedStatistics.REF_INPUTS)

    def test_reset(self, collector_for_interface_test: TensorStatisticCollectorBase):
        for input_ in TestCollectedStatistics.REF_INPUTS:
            collector_for_interface_test.register_input(input_)
        collector_for_interface_test.reset()
        assert collector_for_interface_test.collected_samples() == 0
        with pytest.raises(StatisticsNotCollectedError):
            collector_for_interface_test.get_statistics()

    def test_enable_disable(self, collector_for_interface_test: TensorStatisticCollectorBase):
        for input_ in TestCollectedStatistics.REF_INPUTS:
            collector_for_interface_test.register_input(input_)

        collector_for_interface_test.disable()
        for input_ in TestCollectedStatistics.REF_INPUTS:
            collector_for_interface_test.register_input(input_)
        assert collector_for_interface_test.collected_samples() == len(TestCollectedStatistics.REF_INPUTS)

        collector_for_interface_test.enable()
        for input_ in TestCollectedStatistics.REF_INPUTS:
            collector_for_interface_test.register_input(input_)
        assert collector_for_interface_test.collected_samples() == 2 * len(TestCollectedStatistics.REF_INPUTS)

    OFFLINE_COLLECTORS = [
        partial(
            TFMixedMinMaxStatisticCollector,
            use_per_sample_stats=False,
            use_abs_max=False,
            use_means_of_mins=False,
            use_means_of_maxs=False,
        ),
        partial(TFMeanMinMaxStatisticCollector, use_per_sample_stats=False, use_abs_max=False),
        TFMedianMADStatisticCollector,
        partial(TFPercentileStatisticCollector, percentiles_to_collect=[10.0]),
        partial(TFMeanPercentileStatisticCollector, percentiles_to_collect=[10.0]),
    ]

    REF_NUM_SAMPLES = 3

    @pytest.fixture(params=OFFLINE_COLLECTORS)
    def collector_for_num_samples_test(self, request):
        collector_type = request.param
        return collector_type(reduction_shape=(1,), num_samples=TestCollectedStatistics.REF_NUM_SAMPLES)

    def test_num_samples(self, collector_for_num_samples_test: OfflineTensorStatisticCollector):
        for input_ in TestCollectedStatistics.REF_INPUTS * 10:
            collector_for_num_samples_test.register_input(input_)
        assert collector_for_num_samples_test.collected_samples() == TestCollectedStatistics.REF_NUM_SAMPLES
