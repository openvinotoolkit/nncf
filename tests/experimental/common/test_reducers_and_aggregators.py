# Copyright (c) 2023 Intel Corporation
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
from itertools import product

import numpy as np
import pytest

from nncf.experimental.common.tensor_statistics.collectors import MaxAggregator
from nncf.experimental.common.tensor_statistics.collectors import MeanAggregator
from nncf.experimental.common.tensor_statistics.collectors import MeanNoOutliersAggregator
from nncf.experimental.common.tensor_statistics.collectors import MedianAggregator
from nncf.experimental.common.tensor_statistics.collectors import MedianNoOutliersAggregator
from nncf.experimental.common.tensor_statistics.collectors import MinAggregator
from nncf.experimental.common.tensor_statistics.collectors import NoopAggregator
from nncf.experimental.common.tensor_statistics.collectors import ShapeAggregator

DEFALUT_3D_MEAN_VALUE = [[2503.125, -2493.75, 5009.375], [-4987.5, 7515.625, -7481.25], [10021.875, -9975.0, 12528.125]]


DEFALUT_3D_MEDIAN_VALUE = [[4.5, 5.0, 13.5], [10.0, 22.5, 15.0], [31.5, 20.0, 40.5]]


NO_OUTLIERS_DEFAULT_3D_MEAN_VALUE = [
    [4.16666667, 8.33333333, 12.5],
    [16.66666667, 20.83333333, 25.0],
    [29.16666667, 33.33333333, 37.5],
]


NO_OUTLIERS_DEFAULT_3D_MEDIAN_VALUE = [[5.0, 4.0, 15.0], [8.0, 25.0, 12.0], [35.0, 16.0, 45.0]]


default_test_quantile = 0.1


def default_test_mean_no_outlier(tp, ps):
    return MeanNoOutliersAggregator(tp, ps, quantile=default_test_quantile)


def default_test_median_no_outlier(tp, ps):
    return MedianNoOutliersAggregator(tp, ps, quantile=default_test_quantile)


class TemplateTestReducersAggreagtors:
    @abstractmethod
    def get_nncf_tensor(self, x: np.array):
        pass

    @pytest.fixture
    @abstractmethod
    def tensor_processor(self):
        pass

    @pytest.fixture(scope="module")
    @abstractmethod
    def reducers(self):
        pass

    @abstractmethod
    def all_close(self, val, ref) -> bool:
        pass

    def test_noop_reducer(self, reducers):
        reducer = reducers["noop"]()
        input_ = np.arange(24).reshape((1, 2, 3, 4))
        reduced_input = reducer([self.get_nncf_tensor(input_)])
        assert len(reduced_input) == 1
        assert self.all_close(reduced_input[0].tensor, input_)

    @pytest.mark.parametrize(
        "reducer_name,ref",
        [
            ("min", ([[[-26]], [[-17]], [[-8]], [[1]]], [[[-26]]])),
            ("max", ([[[-18]], [[-9]], [[0]], [[9]]], [[[9]]])),
            ("abs_max", ([[[26]], [[17]], [[8]], [[9]]], [[[26]]])),
            ("mean", ([[[-22.0]], [[-13.0]], [[-4.0]], [[5.0]]], [[[-8.5]]])),
        ],
    )
    def test_min_max_mean_reducers(self, reducer_name, ref, reducers):
        reduction_shape = (1, 2)
        input_ = np.arange(-26, 10).reshape((4, 3, 3))
        for i, red_shape in enumerate([reduction_shape, None]):
            reducer = reducers[reducer_name](red_shape, False)
            val = reducer([self.get_nncf_tensor(input_)])
            assert len(val) == 1
            assert self.all_close(val[0].tensor, ref[i])

    @pytest.mark.parametrize(
        "reducer_name,ref", [("quantile", ([[[[-20000]]]], [[[[10000]]]])), ("abs_quantile", ([[[[20000]]]],))]
    )
    def test_quantile_reducers(self, reducer_name, ref, reducers):
        reduction_shape = (1, 2, 3)
        input_ = np.arange(-26, 10).reshape((1, 4, 3, 3))
        input_[0][0][0] = -20000
        input_[0][0][1] = 10000
        reducer = reducers[reducer_name](reduction_shape, inplace=False)
        val = reducer([self.get_nncf_tensor(input_)])
        assert len(val) == len(ref)
        for i, ref_ in enumerate(ref):
            assert self.all_close(val[i].tensor, ref_)

    @pytest.mark.parametrize(
        "reducer_name,ref",
        [("batch_mean", [[[[-12.5, -11.5, -10.5], [-9.5, -8.5, -7.5], [-6.5, -5.5, -4.5]]]]), ("mean_per_ch", [-8.5])],
    )
    def test_batch_mean_mean_per_ch_reducers(self, reducer_name, ref, reducers):
        input_ = np.arange(-26, 10).reshape((4, 1, 3, 3))
        reducer = reducers[reducer_name](inplace=False)
        val = reducer([self.get_nncf_tensor(input_)])
        assert len(val) == 1
        assert self.all_close(val[0].tensor, ref)

    def test_noop_aggregator(self):
        aggregator = NoopAggregator(None)

        ref_shape = (1, 3, 5, 7, 9)
        input_ = np.arange(np.prod(ref_shape)).reshape(ref_shape)
        for _ in range(3):
            aggregator.register_reduced_input(self.get_nncf_tensor(input_))

        # pylint: disable=protected-access
        assert aggregator._collected_samples == 3
        aggregated = aggregator.aggregate()
        assert len(aggregated) == 3
        for val in aggregated:
            assert self.all_close(val, input_)

    def test_shape_aggregator(self):
        aggregator = ShapeAggregator()
        ref_shape = (1, 3, 5, 7, 9)
        input_ = np.empty(ref_shape)
        for _ in range(3):
            aggregator.register_reduced_input(self.get_nncf_tensor(input_))

        # pylint: disable=protected-access
        assert aggregator._collected_samples == 1
        assert ref_shape == aggregator.aggregate()

    def test_min_max_aggregators(self, tensor_processor):
        min_aggregator = MinAggregator(tensor_processor)
        max_aggregator = MaxAggregator(tensor_processor)
        input_ = np.arange(3 * 3).reshape((1, 3, 3))
        input_[0, 0, 0] = -10000
        for i in range(-5, 5):
            min_aggregator.register_reduced_input(self.get_nncf_tensor(input_ * (-i)))
            max_aggregator.register_reduced_input(self.get_nncf_tensor(input_ * i))

        min_ref = [[[-50000, -4, -8], [-12, -16, -20], [-24, -28, -32]]]
        assert self.all_close(min_ref, min_aggregator.aggregate())

        max_ref = [[[50000, 4, 8], [12, 16, 20], [24, 28, 32]]]
        assert self.all_close(max_ref, max_aggregator.aggregate())

    NO_OUTLIERS_TEST_PARAMS = [
        (MeanAggregator, True, 1, 1404.5138888888905),
        (MedianAggregator, True, 1, 15.5),
        (
            MeanAggregator,
            False,
            1,
            [2503.125, -2493.75, 5009.375, -4987.5, 7515.625, -7481.25, 10021.875, -9975.0, 12528.125],
        ),
        (MedianAggregator, False, 1, [4.5, 5.0, 13.5, 10.0, 22.5, 15.0, 31.5, 20.0, 40.5]),
        (MeanAggregator, True, 2, [2512.5, -1651.04166667, 3352.08333333]),
        (MedianAggregator, True, 2, [13.0, 12.5, 21.0]),
        (MeanAggregator, False, 2, DEFALUT_3D_MEAN_VALUE),
        (MedianAggregator, False, 2, DEFALUT_3D_MEDIAN_VALUE),
        (MeanAggregator, True, 3, DEFALUT_3D_MEAN_VALUE),
        (MedianAggregator, True, 3, DEFALUT_3D_MEDIAN_VALUE),
        (MeanAggregator, False, 3, [DEFALUT_3D_MEAN_VALUE]),
        (MedianAggregator, False, 3, [DEFALUT_3D_MEDIAN_VALUE]),
        (default_test_mean_no_outlier, True, 1, 1404.5138888888905),
        (default_test_median_no_outlier, True, 1, 15.5),
        (
            default_test_mean_no_outlier,
            False,
            1,
            [4.16666667, 8.33333333, 12.5, 16.66666667, 20.83333333, 25.0, 29.16666667, 33.33333333, 37.5],
        ),
        (default_test_median_no_outlier, False, 1, [5.0, 4.0, 15.0, 8.0, 25.0, 12.0, 35.0, 16.0, 45.0]),
        (default_test_mean_no_outlier, True, 2, [16.66666667, 20.83333333, 25.0]),
        (default_test_median_no_outlier, True, 2, [14.0, 10.0, 24.0]),
        (default_test_mean_no_outlier, False, 2, NO_OUTLIERS_DEFAULT_3D_MEAN_VALUE),
        (default_test_median_no_outlier, False, 2, NO_OUTLIERS_DEFAULT_3D_MEDIAN_VALUE),
        (default_test_mean_no_outlier, True, 3, NO_OUTLIERS_DEFAULT_3D_MEAN_VALUE),
        (default_test_median_no_outlier, True, 3, NO_OUTLIERS_DEFAULT_3D_MEDIAN_VALUE),
        (default_test_mean_no_outlier, False, 3, [NO_OUTLIERS_DEFAULT_3D_MEAN_VALUE]),
        (default_test_median_no_outlier, False, 3, [NO_OUTLIERS_DEFAULT_3D_MEDIAN_VALUE]),
    ]

    @pytest.mark.parametrize("aggregator_cls,use_per_sample_stats,dims,refs", NO_OUTLIERS_TEST_PARAMS)
    def test_mean_median_agggregators(self, aggregator_cls, refs, tensor_processor, dims, use_per_sample_stats):
        input_ = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        input_with_outliers = np.array(
            [100_000, -100_000, 200_000, -200_000, 300_000, -300_000, 400_000, -400_000, 500_000]
        )
        if dims == 2:
            input_ = input_.reshape((3, 3))
            input_with_outliers = input_with_outliers.reshape((3, 3))
        if dims == 3:
            input_ = input_.reshape((1, 3, 3))
            input_with_outliers = input_with_outliers.reshape((1, 3, 3))

        aggregator = aggregator_cls(tensor_processor, use_per_sample_stats)
        for i in range(1, 6):
            aggregator.register_reduced_input(self.get_nncf_tensor(input_ * i))
        # this registration is to make diff between mean and median bigger
        aggregator.register_reduced_input(self.get_nncf_tensor(input_ * 10))
        is_median = isinstance(aggregator, (MedianAggregator, MedianNoOutliersAggregator))
        # Outliers registration
        for i in range(2):
            # mult is needed to make outlier and no outlier aggreagators differs
            mult = 2.2 * i - 1 if not is_median else 1
            aggregator.register_reduced_input(self.get_nncf_tensor(input_with_outliers * mult))
        ret_val = aggregator.aggregate()
        assert self.all_close(ret_val, refs)

    @pytest.mark.parametrize(
        "reducer_name",
        ["noop", "min", "max", "abs_max", "mean", "quantile", "abs_quantile", "batch_mean", "mean_per_ch"],
    )
    def test_reducers_name_hash_equal(self, reducer_name, reducers):
        if reducer_name == "noop":
            reducers_instances = [reducers[reducer_name]() for _ in range(2)]
            assert hash(reducers_instances[0]) == hash(reducers_instances[1])
            assert reducers_instances[0] == reducers_instances[1]
            assert reducers_instances[0].name == reducers_instances[1].name
            assert len(set(reducers_instances)) == 1
            return

        params = {}
        if reducer_name in ["min", "max", "abs_max", "mean"]:
            params["reduction_shape"] = [None, (0, 1, 3), (1, 2, 3)]
            params["inplace"] = [False, True]
        elif reducer_name in ["quantile", "abs_quantile"]:
            params["reduction_shape"] = [None, (0, 1, 3), (1, 2, 3)]
            params["quantile"] = [[0.01, 0.99], [0.001, 0.999]]
        elif reducer_name == "batch_mean":
            params["inplace"] = [False, True]
        elif reducer_name == "mean_per_ch":
            params["inplace"] = [False, True]
            params["channel_dim"] = [1, 2]
        else:
            raise RuntimeError(
                "test_min_max_mean_reducer_hash_equal configurated in a wrong way."
                f" Wrong reducer_name: {reducer_name}"
            )

        def product_dict(**kwargs):
            keys = kwargs.keys()
            for instance in product(*kwargs.values()):
                yield dict(zip(keys, instance))

        reducer_cls = reducers[reducer_name]
        reducers_instances = []
        for params_ in product_dict(**params):
            reducers_instances.append(reducer_cls(**params_))

        assert len(set(reducers_instances)) == len(reducers_instances)
        assert len({hash(reducer) for reducer in reducers_instances}) == len(reducers_instances)
        assert len({reducer.name for reducer in reducers_instances}) == len(reducers_instances)

        hashes = [hash(reducer) for reducer in reducers_instances]
        test_input = [self.get_nncf_tensor(np.empty((1, 3, 4, 4)))]
        for reducer, init_hash in zip(reducers_instances, hashes):
            reducer(test_input)
            assert hash(reducer) == init_hash
