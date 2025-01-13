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
from dataclasses import dataclass
from functools import partial
from itertools import product
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np
import pytest

import nncf
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.tensor import NNCFTensor
from nncf.experimental.common.tensor_statistics.collectors import AggregationAxes
from nncf.experimental.common.tensor_statistics.collectors import HAWQAggregator
from nncf.experimental.common.tensor_statistics.collectors import MaxAggregator
from nncf.experimental.common.tensor_statistics.collectors import MeanAggregator
from nncf.experimental.common.tensor_statistics.collectors import MeanNoOutliersAggregator
from nncf.experimental.common.tensor_statistics.collectors import MedianAbsoluteDeviationAggregator
from nncf.experimental.common.tensor_statistics.collectors import MedianAggregator
from nncf.experimental.common.tensor_statistics.collectors import MedianNoOutliersAggregator
from nncf.experimental.common.tensor_statistics.collectors import MinAggregator
from nncf.experimental.common.tensor_statistics.collectors import NoopAggregator
from nncf.experimental.common.tensor_statistics.collectors import PercentileAggregator
from nncf.experimental.common.tensor_statistics.collectors import RawReducer
from nncf.experimental.common.tensor_statistics.collectors import ShapeAggregator
from nncf.experimental.common.tensor_statistics.collectors import ShapeReducer
from nncf.tensor import functions as fns

DEFAULT_3D_MEAN_VALUE = [[2503.125, -2493.75, 5009.375], [-4987.5, 7515.625, -7481.25], [10021.875, -9975.0, 12528.125]]


DEFAULT_3D_MEDIAN_VALUE = [[4.5, 5.0, 13.5], [10.0, 22.5, 15.0], [31.5, 20.0, 40.5]]


NO_OUTLIERS_DEFAULT_3D_MEAN_VALUE = [
    [4.16666667, 8.33333333, 12.5],
    [16.66666667, 20.83333333, 25.0],
    [29.16666667, 33.33333333, 37.5],
]


NO_OUTLIERS_DEFAULT_3D_MEDIAN_VALUE = [[5.0, 4.0, 15.0], [8.0, 25.0, 12.0], [35.0, 16.0, 45.0]]


default_test_quantile = 0.1


@dataclass
class OfflineAggregatorTestCase:
    aggregation_axes: Optional[AggregationAxes]
    min_ref: np.ndarray
    max_ref: np.ndarray


OFFLINE_AGGREGATORS_TEST_CASES = [
    OfflineAggregatorTestCase(
        aggregation_axes=(0,),
        min_ref=np.array([[[-50000, -4, -8], [-12, -16, -20], [-24, -28, -32]]]),
        max_ref=np.array([[[50000, 4, 8], [12, 16, 20], [24, 28, 32]]]),
    ),
    OfflineAggregatorTestCase(
        aggregation_axes=(
            0,
            2,
        ),
        min_ref=np.array([[[-50000, -28, -32]]]),
        max_ref=np.array([[[50000, 28, 32]]]),
    ),
    OfflineAggregatorTestCase(
        aggregation_axes=(2,),
        min_ref=np.array(
            [
                [[[-50000, 5, 10]]],
                [[[-50000, 5, 10]]],
                [[[-40000, 4, 8]]],
                [[[-40000, 4, 8]]],
                [[[-30000, 3, 6]]],
                [[[-30000, 3, 6]]],
                [[[-20000, 2, 4]]],
                [[[-20000, 2, 4]]],
                [[[-10000, 1, 2]]],
                [[[-10000, 1, 2]]],
                [[[0, 0, 0]]],
                [[[0, 0, 0]]],
                [[[-6, -7, -8]]],
                [[[-6, -7, -8]]],
                [[[-12, -14, -16]]],
                [[[-12, -14, -16]]],
                [[[-18, -21, -24]]],
                [[[-18, -21, -24]]],
                [[[-24, -28, -32]]],
                [[[-24, -28, -32]]],
            ]
        ),
        max_ref=np.array(
            [
                [[[50000, -5, -10]]],
                [[[50000, -5, -10]]],
                [[[40000, -4, -8]]],
                [[[40000, -4, -8]]],
                [[[30000, -3, -6]]],
                [[[30000, -3, -6]]],
                [[[20000, -2, -4]]],
                [[[20000, -2, -4]]],
                [[[10000, -1, -2]]],
                [[[10000, -1, -2]]],
                [[[0, 0, 0]]],
                [[[0, 0, 0]]],
                [[[6, 7, 8]]],
                [[[6, 7, 8]]],
                [[[12, 14, 16]]],
                [[[12, 14, 16]]],
                [[[18, 21, 24]]],
                [[[18, 21, 24]]],
                [[[24, 28, 32]]],
                [[[24, 28, 32]]],
            ]
        ),
    ),
]


def default_test_mean_no_outlier(aggregation_axes):
    return MeanNoOutliersAggregator(
        aggregation_axes=aggregation_axes,
        quantile=default_test_quantile,
    )


def default_test_median_no_outlier(aggregation_axes):
    return MedianNoOutliersAggregator(
        aggregation_axes=aggregation_axes,
        quantile=default_test_quantile,
    )


class TemplateTestReducersAggregators:
    @abstractmethod
    def get_nncf_tensor(self, x: np.array, dtype: Optional[Dtype] = None):
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

    @abstractmethod
    def squeeze_tensor(self, ref_tensor: List[Any], axes: Optional[Tuple[int]] = None):
        pass

    @abstractmethod
    def cast_tensor(self, tensor, dtype: Dtype):
        pass

    @pytest.mark.parametrize("reducer_cls", [RawReducer])
    @pytest.mark.parametrize("input_data", [np.arange(24).reshape((1, 2, 3, 4)), np.array([])])
    def test_other_reducers(self, reducer_cls, input_data):
        reducer = reducer_cls()
        tensor_data = self.get_nncf_tensor(input_data)
        reduced_input = reducer([tensor_data])
        if tensor_data.isempty():
            assert reduced_input is None
        else:
            assert len(reduced_input) == 1
            assert fns.allclose(reduced_input[0], tensor_data)

    @pytest.mark.parametrize("reducer_cls", [RawReducer, ShapeReducer])
    def test_other_reducers_name_hash_equal(self, reducer_cls):
        reducers_instances = [reducer_cls() for _ in range(2)]
        assert hash(reducers_instances[0]) == hash(reducers_instances[1])
        assert reducers_instances[0] == reducers_instances[1]
        assert reducers_instances[0].name == reducers_instances[1].name
        assert len(set(reducers_instances)) == 1

    @pytest.mark.parametrize("input_data", [np.arange(24).reshape((1, 2, 3, 4)), np.array([1])])
    def test_shape_reducer(self, input_data):
        reducer = ShapeReducer()
        tensor_data = self.get_nncf_tensor(input_data)
        reduced_input = reducer([tensor_data])
        assert len(reduced_input) == 1
        assert all(it[0] == it[1] for it in zip(reduced_input[0], tensor_data.shape))

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
        reduction_axes = (1, 2)
        input_ = np.arange(-26, 10).reshape((4, 3, 3))
        for i, reduction_axes_ in enumerate([reduction_axes, None]):
            reducer = reducers[reducer_name](reduction_axes=reduction_axes_, inplace=False)
            val = reducer([self.get_nncf_tensor(input_, Dtype.FLOAT)])
            assert len(val) == 1
            assert fns.allclose(val[0], self.get_nncf_tensor(ref[i]))

    @pytest.mark.parametrize(
        "reducer_name,ref", [("quantile", ([[[[-20000]]]], [[[[10000]]]])), ("abs_quantile", ([[[[20000]]]],))]
    )
    def test_quantile_reducers(self, reducer_name, ref, reducers):
        reduction_axes = (1, 2, 3)
        input_ = np.arange(-26, 10).reshape((1, 4, 3, 3))
        input_[0][0][0] = -20000
        input_[0][0][1] = 10000
        reducer = reducers[reducer_name](reduction_axes=reduction_axes, inplace=False)
        val = reducer([self.get_nncf_tensor(input_, dtype=Dtype.FLOAT)])
        assert val.shape[0] == len(ref)
        for i, ref_ in enumerate(ref):
            assert fns.allclose(val[i], self.get_nncf_tensor(ref_))

    @pytest.mark.parametrize(
        "reducer_name,ref,kwargs",
        [
            ("batch_mean", [[[[-12.5, -11.5, -10.5], [-9.5, -8.5, -7.5], [-6.5, -5.5, -4.5]]]], {}),
            ("mean_per_ch", [-22.0, -13.0, -4.0, 5.0], {"channel_axis": 0}),
        ],
    )
    def test_batch_mean_mean_per_ch_reducers(self, reducer_name, ref, reducers, kwargs):
        input_ = np.arange(-26, 10).reshape((4, 1, 3, 3))
        reducer = reducers[reducer_name](inplace=False, **kwargs)
        val = reducer([self.get_nncf_tensor(input_, Dtype.FLOAT)])
        assert len(val) == 1
        assert fns.allclose(val[0], self.get_nncf_tensor(ref))

    def test_noop_aggregator(self):
        aggregator = NoopAggregator(None)

        ref_shape = (1, 3, 5, 7, 9)
        input_ = np.arange(np.prod(ref_shape)).reshape(ref_shape)
        for _ in range(3):
            aggregator.register_reduced_input(self.get_nncf_tensor(input_))

        assert aggregator._collected_samples == 3
        aggregated = aggregator.aggregate()
        assert len(aggregated) == 3
        for val in aggregated:
            assert fns.allclose(val, self.get_nncf_tensor(input_))

    def test_shape_aggregator(self):
        aggregator = ShapeAggregator()
        ref_shape = (1, 3, 5, 7, 9)
        input_ = np.empty(ref_shape)
        for _ in range(3):
            aggregator.register_reduced_input(self.get_nncf_tensor(input_))

        assert aggregator._collected_samples == 1
        assert ref_shape == aggregator.aggregate()

    @pytest.mark.parametrize(
        "offline_aggregators_test_desc",
        OFFLINE_AGGREGATORS_TEST_CASES,
    )
    def test_min_max_aggregators(self, offline_aggregators_test_desc: OfflineAggregatorTestCase):
        aggregation_axes = offline_aggregators_test_desc.aggregation_axes
        min_aggregator = MinAggregator(aggregation_axes=aggregation_axes)
        max_aggregator = MaxAggregator(aggregation_axes=aggregation_axes)
        input_ = np.arange(3 * 3).reshape((1, 3, 3))
        input_[0, 0, 0] = -10000
        for i in range(-5, 5):
            min_aggregator.register_reduced_input(self.get_nncf_tensor(input_ * (-i)))
            max_aggregator.register_reduced_input(self.get_nncf_tensor(input_ * i))

            for axis in aggregation_axes:
                if axis:
                    concatted_input = fns.concatenate([self.get_nncf_tensor(input_)] * 2, axis - 1).data
                    min_aggregator.register_reduced_input(self.get_nncf_tensor(concatted_input * (-i)))
                    max_aggregator.register_reduced_input(self.get_nncf_tensor(concatted_input * i))

        min_ref = offline_aggregators_test_desc.min_ref
        max_ref = offline_aggregators_test_desc.max_ref

        assert fns.allclose(min_aggregator.aggregate(), self.get_nncf_tensor(min_ref))
        assert fns.allclose(max_aggregator.aggregate(), self.get_nncf_tensor(max_ref))

    NO_OUTLIERS_TEST_PARAMS = [
        (MeanAggregator, (0, 1), 1, [1404.5138888888905]),
        (MedianAggregator, (0, 1), 1, [24.0]),
        (
            MeanAggregator,
            (0,),
            1,
            [2503.125, -2493.75, 5009.375, -4987.5, 7515.625, -7481.25, 10021.875, -9975.0, 12528.125],
        ),
        (MedianAggregator, (0,), 1, [4.5, 5.0, 13.5, 10.0, 22.5, 15.0, 31.5, 20.0, 40.5]),
        ############################################################
        (MeanAggregator, (0, 1), 2, [[2512.5, -1651.04166667, 3352.08333333]]),
        (MedianAggregator, (0, 1), 2, [[13.0, 12.5, 21.0]]),
        (MeanAggregator, (0,), 2, DEFAULT_3D_MEAN_VALUE),
        (MedianAggregator, (0,), 2, DEFAULT_3D_MEDIAN_VALUE),
        ############################################################
        (MeanAggregator, (0, 1), 3, [DEFAULT_3D_MEAN_VALUE]),
        (MedianAggregator, (0, 1), 3, [DEFAULT_3D_MEDIAN_VALUE]),
        (MeanAggregator, (0,), 3, [DEFAULT_3D_MEAN_VALUE]),
        (MedianAggregator, (0,), 3, [DEFAULT_3D_MEDIAN_VALUE]),
        ############################################################
        (default_test_mean_no_outlier, (0, 1), 1, [20.0893]),
        (default_test_median_no_outlier, (0, 1), 1, [30.0]),
        (
            default_test_mean_no_outlier,
            (0,),
            1,
            [4.16666667, 8.33333333, 12.5, 16.66666667, 20.83333333, 25.0, 29.16666667, 33.33333333, 37.5],
        ),
        (default_test_median_no_outlier, (0,), 1, [5.0, 4.0, 15.0, 8.0, 25.0, 12.0, 35.0, 16.0, 45.0]),
        ############################################################
        (default_test_mean_no_outlier, (0, 1), 2, [[16.66666667, 20.83333333, 25.0]]),
        (default_test_median_no_outlier, (0, 1), 2, [[14.0, 10.0, 24.0]]),
        (default_test_mean_no_outlier, (0,), 2, NO_OUTLIERS_DEFAULT_3D_MEAN_VALUE),
        (default_test_median_no_outlier, (0,), 2, NO_OUTLIERS_DEFAULT_3D_MEDIAN_VALUE),
        ############################################################
        (default_test_mean_no_outlier, (0, 1), 3, [NO_OUTLIERS_DEFAULT_3D_MEAN_VALUE]),
        (default_test_median_no_outlier, (0, 1), 3, [NO_OUTLIERS_DEFAULT_3D_MEDIAN_VALUE]),
        (default_test_mean_no_outlier, (0,), 3, [NO_OUTLIERS_DEFAULT_3D_MEAN_VALUE]),
        (default_test_median_no_outlier, (0,), 3, [NO_OUTLIERS_DEFAULT_3D_MEDIAN_VALUE]),
    ]

    def _get_inputs_for_mean_median_aggregators(
        self,
        dims: int,
        aggregation_axes: Tuple[int, ...],
        is_median: bool,
        different_sizes: bool = False,
    ) -> Iterator[NNCFTensor]:
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

        for i in range(1, 6):
            yield self.get_nncf_tensor(input_ * i, Dtype.FLOAT)
        # this registration is to make diff between mean and median bigger
        yield self.get_nncf_tensor(input_ * 10, Dtype.FLOAT)
        # Outliers registration
        for i in range(2):
            # mult is needed to make outlier and no outlier aggreagators differs
            mult = 2.2 * i - 1 if not is_median else 1
            yield self.get_nncf_tensor(input_with_outliers * mult, Dtype.FLOAT)
            if is_median and dims == 1 and aggregation_axes == (0, 1):
                # To make no outliers and outliers versions return different output
                yield self.get_nncf_tensor(np.full(input_with_outliers.shape, input_with_outliers[-1]), Dtype.FLOAT)

        if different_sizes:
            # Different size input registration:
            cat_input = self.get_nncf_tensor(input_, Dtype.FLOAT)
            for axis in aggregation_axes:
                if axis:
                    yield fns.concatenate([cat_input] * 2, axis=axis - 1)

    @pytest.mark.parametrize("aggregator_cls,aggregation_axes,dims,refs", NO_OUTLIERS_TEST_PARAMS)
    def test_mean_median_aggregators(self, aggregator_cls, refs, dims, aggregation_axes):
        aggregator = aggregator_cls(aggregation_axes=aggregation_axes)
        is_median = isinstance(aggregator, (MedianAggregator, MedianNoOutliersAggregator))
        for input_ in self._get_inputs_for_mean_median_aggregators(dims, aggregation_axes, is_median):
            aggregator.register_reduced_input(input_)

        ret_val = aggregator.aggregate()
        assert fns.allclose(ret_val, self.get_nncf_tensor(refs))

    NO_OUTLIERS_DIFFERENT_SIZES_TEST_PARAMS = [
        (MeanAggregator, (1,), 1, [[5.0], [10.0], [15.0], [20.0], [25.0], [50.0], [-55556.0], [66667.0], [5.0]]),
        (MedianAggregator, (1,), 1, [[5.0], [10.0], [15.0], [20.0], [25.0], [50.0], [100_000.0], [100_000.0], [5.0]]),
        (MeanAggregator, (0, 1), 1, [1124.6111]),
        (MedianAggregator, (0, 1), 1, [15.5]),
        (
            default_test_mean_no_outlier,
            (1,),
            1,
            [[5.0], [10.0], [15.0], [20.0], [25.0], [50.0], [-57143.0], [68571.0], [5.0]],
        ),
        (
            default_test_median_no_outlier,
            (1,),
            1,
            [[5.0], [10.0], [15.0], [20.0], [25.0], [50.0], [100_000.0], [100_000.0], [5.0]],
        ),
        (default_test_mean_no_outlier, (0, 1), 1, [16.8750]),
        (default_test_median_no_outlier, (0, 1), 1, [20]),
    ]

    @pytest.mark.parametrize("aggregator_cls,aggregation_axes,dims,refs", NO_OUTLIERS_DIFFERENT_SIZES_TEST_PARAMS)
    def test_mean_median_agggregators_different_sizes(self, aggregator_cls, refs, dims, aggregation_axes):
        aggregator = aggregator_cls(aggregation_axes=aggregation_axes)
        is_median = isinstance(aggregator, (MedianAggregator, MedianNoOutliersAggregator))
        for input_ in self._get_inputs_for_mean_median_aggregators(
            dims, aggregation_axes, is_median, different_sizes=True
        ):
            aggregator.register_reduced_input(input_)

        ret_val = aggregator.aggregate()
        assert fns.allclose(ret_val, self.get_nncf_tensor(refs))

    @pytest.fixture(
        name="MAD_percentile_aggregator_cls",
        params=[
            MedianAbsoluteDeviationAggregator,
            partial(
                PercentileAggregator,
                percentiles_to_collect=[5, 10, 90, 95],
            ),
        ],
    )
    def aggregator_cls_fixture(self, request):
        return request.param

    REF_MAD_PERCENTILE_REF_VALUES = {
        MedianAbsoluteDeviationAggregator: {
            None: {
                "median_values": np.array([4.5, 9.0, 13.5, 18.0, 22.5, 27.0, 31.5, 36.0, 40.5]),
                "mad_values": np.array([2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5]),
            },
            (0,): {
                "median_values": np.array([4.5, 9.0, 13.5, 18.0, 22.5, 27.0, 31.5, 36.0, 40.5]),
                "mad_values": np.array([2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5]),
            },
            (0, 1): {
                "median_values": np.array([18.0]),
                "mad_values": np.array([12.0]),
            },
        },
        PercentileAggregator: {
            None: {
                5: np.array([0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6]),
                10: np.array([0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 6.4, 7.2]),
                90: np.array([7.2, 14.4, 21.6, 28.8, 36.0, 43.2, 50.4, 57.6, 64.8]),
                95: np.array([7.6, 15.2, 22.8, 30.4, 38.0, 45.6, 53.2, 60.8, 68.4]),
            },
            (0,): {
                5: np.array([0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6]),
                10: np.array([0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 6.4, 7.2]),
                90: np.array([7.2, 14.4, 21.6, 28.8, 36.0, 43.2, 50.4, 57.6, 64.8]),
                95: np.array([7.6, 15.2, 22.8, 30.4, 38.0, 45.6, 53.2, 60.8, 68.4]),
            },
            (0, 1): {
                5: np.array([0.0]),
                10: np.array([0.0]),
                90: np.array([48.0]),
                95: np.array([56.0]),
            },
        },
    }

    @pytest.mark.parametrize("aggregation_axes", [None, (0,), (0, 1)])
    def test_mad_percentile_aggregators(self, MAD_percentile_aggregator_cls, aggregation_axes):
        aggregator = MAD_percentile_aggregator_cls(aggregation_axes=aggregation_axes)
        input_ = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        for i in range(9):
            aggregator.register_reduced_input(self.get_nncf_tensor(input_ * i, Dtype.FLOAT))

        ret_val = aggregator.aggregate()
        ref_values = self.REF_MAD_PERCENTILE_REF_VALUES[aggregator.__class__][aggregation_axes]
        assert len(ret_val) == len(ref_values)
        for k, v in ref_values.items():
            assert fns.allclose(ret_val[k], self.get_nncf_tensor(v)), f"{k}"

    REF_MAD_PERCENTILE_REF_VALUES_DYNAMIC_TENSORS = {
        MedianAbsoluteDeviationAggregator: {
            "median_values": np.array([[28.5, 35.5, 43.5]]).reshape(1, 3, 1),
            "mad_values": np.array([[[24.0, 24.0, 24.0]]]).reshape(1, 3, 1),
        },
        PercentileAggregator: {
            5: np.array([[[0.95, 5.95, 9.95]]]).reshape(1, 3, 1),
            10: np.array([[[1.9, 7.9, 15.5]]]).reshape(1, 3, 1),
            90: np.array([[[75.1, 83.1, 91.1]]]).reshape(1, 3, 1),
            95: np.array([[[77.05, 85.05, 93.05]]]).reshape(1, 3, 1),
        },
    }

    def test_mad_percentile_aggregators_different_sizes(self, MAD_percentile_aggregator_cls):
        aggregator = MAD_percentile_aggregator_cls(aggregation_axes=(0, 1, 3))
        for shape in ((2, 3, 4), (4, 3, 8)):
            aggregator.register_reduced_input(
                self.get_nncf_tensor(np.arange(np.prod(shape)).reshape(shape), Dtype.FLOAT)
            )
        ret_val = aggregator.aggregate()

        ref_values = self.REF_MAD_PERCENTILE_REF_VALUES_DYNAMIC_TENSORS[aggregator.__class__]
        assert len(ret_val) == len(ref_values)
        for k, v in ref_values.items():
            assert fns.allclose(ret_val[k], self.get_nncf_tensor(v))

    def test_mad_percentile_aggregators_not_implemented_aggregation_axes(self, MAD_percentile_aggregator_cls):
        with pytest.raises(NotImplementedError):
            MAD_percentile_aggregator_cls(aggregation_axes=(1, 2, 3))

    @pytest.mark.parametrize(
        "reducer_name",
        ["min", "max", "abs_max", "mean", "quantile", "abs_quantile", "batch_mean", "mean_per_ch"],
    )
    def test_reducers_name_hash_equal(self, reducer_name, reducers):
        params = {}
        if reducer_name in ["min", "max", "abs_max", "mean"]:
            params["reduction_axes"] = [None, (0, 1, 3), (1, 2, 3)]
            params["inplace"] = [False, True]
        elif reducer_name in ["quantile", "abs_quantile"]:
            params["reduction_axes"] = [None, (0, 1, 3), (1, 2, 3)]
            params["quantile"] = [[0.01, 0.99], [0.001, 0.999]]
        elif reducer_name == "batch_mean":
            params["inplace"] = [False, True]
        elif reducer_name == "mean_per_ch":
            params["inplace"] = [False, True]
            params["channel_axis"] = [1, 2]
        else:
            raise nncf.ValidationError(
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

    HAWQ_AGGREGATOR_REFERENCE_VALUES = [
        ([np.arange(10)], 57.0),
        ([np.arange(12).reshape((2, 6)), np.arange(24).reshape((4, 6))], 181.92361111111111),
        ([np.arange(8 * i).reshape((1, 8, i)) for i in range(1, 5)], 165.61627197265625),
    ]

    @pytest.mark.parametrize("inputs,reference_output", HAWQ_AGGREGATOR_REFERENCE_VALUES)
    def test_hawq_aggregator(self, inputs, reference_output):
        aggregator = HAWQAggregator()
        for x in inputs:
            aggregator.register_reduced_input(self.get_nncf_tensor(x, Dtype.FLOAT))

        ret_val = aggregator.aggregate()
        assert fns.allclose(ret_val, reference_output)
