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
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Any, List, Type, Union

import numpy as np
import pytest

import nncf
from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.tensor_statistics.aggregator import EMPTY_DATASET_ERROR
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.experimental.common.tensor_statistics.collectors import NoopAggregator
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.collectors import TensorReducerBase
from nncf.quantization.algorithms.bias_correction.backend import BiasCorrectionAlgoBackend
from nncf.quantization.algorithms.fast_bias_correction.backend import FastBiasCorrectionAlgoBackend
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.range_estimator import AggregatorType
from nncf.quantization.range_estimator import RangeEstimatorParameters
from nncf.quantization.range_estimator import RangeEstimatorParametersSet
from nncf.quantization.range_estimator import StatisticsCollectorParameters
from nncf.quantization.range_estimator import StatisticsType
from nncf.tensor import functions as fns


class MockedDataset:
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration


class BiasCorrectionAlgos(Enum):
    BIAS_CORRECTION = "bias_correction"
    FAST_BIAS_CORRECTION = "fast_bias_correction"


class BCStatsCollectors(Enum):
    MEAN = "mean"
    RAW = "raw"


class TemplateTestStatisticsAggregator:
    @staticmethod
    @abstractmethod
    def get_min_max_algo_backend_cls() -> Type[MinMaxAlgoBackend]:
        pass

    @abstractmethod
    def get_bias_correction_algo_backend_cls(self) -> Type[BiasCorrectionAlgoBackend]:
        pass

    @abstractmethod
    def get_fast_bias_correction_algo_backend_cls(self) -> Type[FastBiasCorrectionAlgoBackend]:
        pass

    @abstractmethod
    def get_backend_model(self, dataset_samples):
        pass

    @abstractmethod
    def get_statistics_aggregator(self, dataset):
        pass

    @abstractmethod
    def get_dataset(self, samples):
        pass

    @staticmethod
    @abstractmethod
    def get_target_point(target_type: TargetType) -> TargetPoint:
        pass

    @abstractmethod
    def get_target_point_cls(self):
        pass

    @abstractmethod
    @pytest.fixture(scope="session")
    def test_params(self):
        """
        Please make the same topologies with the same names as it
        presented in Openvino tests.
        """

    @abstractmethod
    @pytest.fixture
    def dataset_samples(self):
        pass

    @abstractmethod
    @pytest.fixture
    def inplace_statistics(self) -> bool:
        pass

    @abstractmethod
    @pytest.fixture
    def is_backend_support_custom_estimators(self) -> bool:
        """
        False if backend can initialize only following tensor collectors:
        MinMax, MeanMinMax.
        """

    @abstractmethod
    def reducers_map(self) -> List[TensorReducerBase]:
        pass

    @pytest.fixture
    def dataset_values(self):
        return [{"max": 1, "min": -10}, {"max": 0.1, "min": -1}, {"max": 128, "min": -128}]

    @staticmethod
    def get_min_max_algo_cls() -> Type[MinMaxQuantization]:
        return MinMaxQuantization

    @dataclass
    class MinMaxTestParameters:
        range_estimator_params: RangeEstimatorParameters
        target_type: TargetType
        quantization_mode: QuantizationMode
        per_channel: bool
        ref_max_val: Union[np.ndarray, float]
        ref_min_val: Union[np.ndarray, float]

    TEST_MEAN_QUANTILE = RangeEstimatorParameters(
        min=StatisticsCollectorParameters(StatisticsType.QUANTILE, AggregatorType.MEAN, quantile_outlier_prob=0.01),
        max=StatisticsCollectorParameters(StatisticsType.QUANTILE, AggregatorType.MEAN, quantile_outlier_prob=0.01),
    )

    def dataset_samples_to_conv_w(self, dataset_sample):
        # Layout: [O, I, K, K]
        d = dataset_sample
        in_ch = d.shape[0]
        return np.stack([np.stack([d[i]] * in_ch, axis=0) for i in range(in_ch)], axis=0)

    @pytest.mark.parametrize(
        "test_parameters, ",
        # Activation collectors
        (
            (
                MinMaxTestParameters(
                    RangeEstimatorParametersSet.MEAN_MINMAX,
                    TargetType.POST_LAYER_OPERATION,
                    QuantizationMode.ASYMMETRIC,
                    False,
                    64.5,
                    -63.5,
                )
            ),
            (
                MinMaxTestParameters(
                    RangeEstimatorParametersSet.MEAN_MINMAX,
                    TargetType.POST_LAYER_OPERATION,
                    QuantizationMode.ASYMMETRIC,
                    True,
                    np.array((1, 0.55, 64.5)),
                    np.array((-4.5, 0, -63.5)),
                )
            ),
            (
                MinMaxTestParameters(
                    RangeEstimatorParametersSet.MEAN_MINMAX,
                    TargetType.POST_LAYER_OPERATION,
                    QuantizationMode.SYMMETRIC,
                    True,
                    np.array((5.5, 1, 64.5)),
                    np.array((-4.5, 0, -63.5)),
                )
            ),
            (
                MinMaxTestParameters(
                    RangeEstimatorParametersSet.MINMAX,
                    TargetType.POST_LAYER_OPERATION,
                    QuantizationMode.ASYMMETRIC,
                    False,
                    128,
                    -128,
                )
            ),
            (
                MinMaxTestParameters(
                    RangeEstimatorParametersSet.MINMAX,
                    TargetType.POST_LAYER_OPERATION,
                    QuantizationMode.SYMMETRIC,
                    False,
                    128,
                    -128,
                )
            ),
            (
                MinMaxTestParameters(
                    RangeEstimatorParametersSet.MINMAX,
                    TargetType.POST_LAYER_OPERATION,
                    QuantizationMode.ASYMMETRIC,
                    True,
                    np.array((1, 1, 128)),
                    np.array((-10, -1, -128)),
                )
            ),
            (
                MinMaxTestParameters(
                    RangeEstimatorParametersSet.MINMAX,
                    TargetType.POST_LAYER_OPERATION,
                    QuantizationMode.SYMMETRIC,
                    True,
                    np.array((10, 1, 128)),
                    np.array((-10, -1, -128)),
                )
            ),
            (
                MinMaxTestParameters(
                    RangeEstimatorParametersSet.MEDIAN_MINMAX,
                    TargetType.POST_LAYER_OPERATION,
                    QuantizationMode.ASYMMETRIC,
                    False,
                    64.5,
                    -63.5,
                )
            ),
            (
                MinMaxTestParameters(
                    RangeEstimatorParametersSet.MEDIAN_MINMAX,
                    TargetType.POST_LAYER_OPERATION,
                    QuantizationMode.SYMMETRIC,
                    False,
                    64.5,
                    -63.5,
                )
            ),
            (
                MinMaxTestParameters(
                    RangeEstimatorParametersSet.MEDIAN_MINMAX,
                    TargetType.POST_LAYER_OPERATION,
                    QuantizationMode.ASYMMETRIC,
                    True,
                    np.array((1, 0.55, 64.5)),
                    np.array((-4.5, 0.0, -63.5)),
                )
            ),
            (
                MinMaxTestParameters(
                    RangeEstimatorParametersSet.MEDIAN_MINMAX,
                    TargetType.POST_LAYER_OPERATION,
                    QuantizationMode.SYMMETRIC,
                    True,
                    np.array((5.5, 1.0, 64.5)),
                    np.array((-4.5, 0.0, -63.5)),
                )
            ),
            (
                MinMaxTestParameters(
                    RangeEstimatorParametersSet.MEAN_NO_OUTLIERS_MINMAX,
                    TargetType.POST_LAYER_OPERATION,
                    QuantizationMode.ASYMMETRIC,
                    False,
                    0,
                    0,
                )
            ),
            (
                MinMaxTestParameters(
                    RangeEstimatorParametersSet.MEAN_NO_OUTLIERS_MINMAX,
                    TargetType.POST_LAYER_OPERATION,
                    QuantizationMode.SYMMETRIC,
                    False,
                    0,
                    0,
                )
            ),
            (
                MinMaxTestParameters(
                    RangeEstimatorParametersSet.MEAN_NO_OUTLIERS_MINMAX,
                    TargetType.POST_LAYER_OPERATION,
                    QuantizationMode.ASYMMETRIC,
                    True,
                    np.array((1, 0, 0)),
                    np.array((0, 0, 0)),
                )
            ),
            (
                MinMaxTestParameters(
                    RangeEstimatorParametersSet.MEAN_NO_OUTLIERS_MINMAX,
                    TargetType.POST_LAYER_OPERATION,
                    QuantizationMode.SYMMETRIC,
                    True,
                    np.array((0, 1, 0)),
                    np.array((0, 0, 0)),
                )
            ),
            (
                MinMaxTestParameters(
                    TEST_MEAN_QUANTILE,
                    TargetType.POST_LAYER_OPERATION,
                    QuantizationMode.ASYMMETRIC,
                    False,
                    47.9899999999999,
                    -48.15999999999998,
                )
            ),
            (
                MinMaxTestParameters(
                    TEST_MEAN_QUANTILE,
                    TargetType.POST_LAYER_OPERATION,
                    QuantizationMode.SYMMETRIC,
                    False,
                    47.9899999999999,
                    -48.15999999999998,
                )
            ),
            (
                MinMaxTestParameters(
                    TEST_MEAN_QUANTILE,
                    TargetType.POST_LAYER_OPERATION,
                    QuantizationMode.ASYMMETRIC,
                    True,
                    np.array((0.96, 0.546, 59.38)),
                    np.array((-4.100e00, 4.000e-02, -5.838e01)),
                )
            ),
            (
                MinMaxTestParameters(
                    TEST_MEAN_QUANTILE,
                    TargetType.POST_LAYER_OPERATION,
                    QuantizationMode.SYMMETRIC,
                    True,
                    np.array((0.96, 0.546, 59.38)),
                    np.array((-4.100e00, 4.000e-02, -5.838e01)),
                )
            ),
            # Weight collectors
            (
                MinMaxTestParameters(
                    RangeEstimatorParametersSet.MINMAX,
                    TargetType.OPERATION_WITH_WEIGHTS,
                    QuantizationMode.SYMMETRIC,
                    False,
                    128,
                    -128,
                )
            ),
            (
                MinMaxTestParameters(
                    RangeEstimatorParametersSet.MINMAX,
                    TargetType.OPERATION_WITH_WEIGHTS,
                    QuantizationMode.ASYMMETRIC,
                    False,
                    128,
                    -128,
                )
            ),
            (
                MinMaxTestParameters(
                    RangeEstimatorParametersSet.MINMAX,
                    TargetType.OPERATION_WITH_WEIGHTS,
                    QuantizationMode.SYMMETRIC,
                    True,
                    np.array((10, 1, 128)),
                    np.array((-10, -1, -128)),
                )
            ),
            (
                MinMaxTestParameters(
                    RangeEstimatorParametersSet.MINMAX,
                    TargetType.OPERATION_WITH_WEIGHTS,
                    QuantizationMode.ASYMMETRIC,
                    True,
                    np.array((1, 0.1, 128)),
                    np.array((-10, -1, -128)),
                )
            ),
        ),
    )
    def test_statistics_aggregator_min_max(
        self,
        test_parameters: MinMaxTestParameters,
        dataset_samples,
        inplace_statistics,
        is_backend_support_custom_estimators,
        mocker,
    ):
        model = self.get_backend_model(dataset_samples)
        quantizer_config = QuantizerConfig(
            mode=test_parameters.quantization_mode, per_channel=test_parameters.per_channel
        )

        is_standard_estimator = test_parameters.range_estimator_params in [
            RangeEstimatorParametersSet.MINMAX,
            RangeEstimatorParametersSet.MEAN_MINMAX,
        ]
        if not is_standard_estimator and not is_backend_support_custom_estimators:
            pytest.skip("Custom estimators are not supported for this backend yet")

        target_point = self.get_target_point(test_parameters.target_type)
        algorithm_name = "TestAlgo"
        statistic_point = self.create_statistics_point(
            model,
            quantizer_config,
            target_point,
            len(dataset_samples),
            algorithm_name,
            inplace_statistics,
            test_parameters.range_estimator_params,
            mocker,
        )
        statistics_points = StatisticPointsContainer()
        statistics_points.add_statistic_point(statistic_point)

        dataset = self.get_dataset(dataset_samples)
        statistics_aggregator = self.get_statistics_aggregator(dataset)
        statistics_aggregator.register_statistic_points(statistics_points)
        graph = NNCFGraphFactory.create(model)
        statistics_aggregator.collect_statistics(model, graph)

        def filter_func(point):
            return (
                algorithm_name in point.algorithm_to_tensor_collectors and point.target_point.type == target_point.type
            )

        tensor_collectors = list(
            statistics_points.get_algo_statistics_for_node(target_point.target_node_name, filter_func, algorithm_name)
        )

        assert len(tensor_collectors) == 1
        for tensor_collector in tensor_collectors:
            stat = tensor_collector.get_statistics()
            # Torch and Openvino backends tensor collectors return values in shape of scale
            # in comparison to ONNX backends.
            ref_min_val, ref_max_val = test_parameters.ref_min_val, test_parameters.ref_max_val
            if isinstance(ref_min_val, np.ndarray):
                shape = (1, 3, 1, 1)
                if test_parameters.target_type == TargetType.OPERATION_WITH_WEIGHTS:
                    shape = (3, 1, 1, 1)
                ref_min_val, ref_max_val = map(lambda x: np.reshape(x, shape), (ref_min_val, ref_max_val))

            assert np.allclose(stat.min_values.data, ref_min_val)
            assert np.allclose(stat.max_values.data, ref_max_val)
            if isinstance(ref_min_val, np.ndarray):
                assert stat.min_values.shape == ref_min_val.shape
                assert stat.max_values.shape == ref_max_val.shape
            else:
                ref_shape = (1, 1, 1, 1)
                assert stat.min_values.shape == ref_shape
                assert stat.max_values.shape == ref_shape

    @dataclass
    class BCTestParameters:
        algo: BiasCorrectionAlgos
        collector_type: BCStatsCollectors
        target_type: TargetType
        ref_values: Any = None
        axis: int = 1

    MEAN_ACT_AXIS_0_REF = np.array(
        [
            [
                [[1.0, -4.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
                [[0.55, 0.0, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
                [[64.5, -63.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
            ]
        ]
    )

    MEAN_WEIGHTS_AXIS_0_REF = np.array(
        [
            [
                [[43.033337, -46.333332, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[43.033337, -46.333332, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[43.033337, -46.333332, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        ]
    )

    @pytest.mark.parametrize(
        "test_params",
        [
            # TargeType: activations
            BCTestParameters(
                BiasCorrectionAlgos.FAST_BIAS_CORRECTION,
                BCStatsCollectors.MEAN,
                TargetType.POST_LAYER_OPERATION,
                (MEAN_ACT_AXIS_0_REF, (1, 3, 3, 3)),
                axis=0,
            ),
            BCTestParameters(
                BiasCorrectionAlgos.BIAS_CORRECTION,
                BCStatsCollectors.MEAN,
                TargetType.POST_LAYER_OPERATION,
                (MEAN_ACT_AXIS_0_REF, (1, 3, 3, 3)),
                axis=0,
            ),
            BCTestParameters(
                BiasCorrectionAlgos.FAST_BIAS_CORRECTION,
                BCStatsCollectors.MEAN,
                TargetType.POST_LAYER_OPERATION,
                (np.array((0.0, 0.45, 0.5)), (1, 3, 3, 3)),
                axis=1,
            ),
            BCTestParameters(
                BiasCorrectionAlgos.BIAS_CORRECTION,
                BCStatsCollectors.MEAN,
                TargetType.POST_LAYER_OPERATION,
                (np.array((0.0, 0.45, 0.5)), (1, 3, 3, 3)),
                axis=1,
            ),
            BCTestParameters(
                BiasCorrectionAlgos.FAST_BIAS_CORRECTION,
                BCStatsCollectors.MEAN,
                TargetType.POST_LAYER_OPERATION,
                (np.array([-0.04999995, 0.5, 0.5]), (1, 3, 3, 3)),
                axis=2,
            ),
            BCTestParameters(
                BiasCorrectionAlgos.BIAS_CORRECTION,
                BCStatsCollectors.MEAN,
                TargetType.POST_LAYER_OPERATION,
                (np.array([-0.04999995, 0.5, 0.5]), (1, 3, 3, 3)),
                axis=2,
            ),
            BCTestParameters(
                BiasCorrectionAlgos.BIAS_CORRECTION, BCStatsCollectors.RAW, TargetType.POST_LAYER_OPERATION
            ),
            # TargeType: weights
            BCTestParameters(
                BiasCorrectionAlgos.FAST_BIAS_CORRECTION,
                BCStatsCollectors.MEAN,
                TargetType.OPERATION_WITH_WEIGHTS,
                (MEAN_WEIGHTS_AXIS_0_REF, (3, 3, 3, 3)),
                axis=0,
            ),
            BCTestParameters(
                BiasCorrectionAlgos.BIAS_CORRECTION,
                BCStatsCollectors.MEAN,
                TargetType.OPERATION_WITH_WEIGHTS,
                (MEAN_WEIGHTS_AXIS_0_REF, (3, 3, 3, 3)),
                axis=0,
            ),
            BCTestParameters(
                BiasCorrectionAlgos.FAST_BIAS_CORRECTION,
                BCStatsCollectors.MEAN,
                TargetType.OPERATION_WITH_WEIGHTS,
                (np.array([-0.36666664, -0.36666664, -0.36666664]), (3, 3, 3, 3)),
                axis=1,
            ),
            BCTestParameters(
                BiasCorrectionAlgos.BIAS_CORRECTION,
                BCStatsCollectors.MEAN,
                TargetType.OPERATION_WITH_WEIGHTS,
                (np.array([-0.36666664, -0.36666664, -0.36666664]), (3, 3, 3, 3)),
                axis=1,
            ),
            BCTestParameters(
                BiasCorrectionAlgos.FAST_BIAS_CORRECTION,
                BCStatsCollectors.MEAN,
                TargetType.OPERATION_WITH_WEIGHTS,
                (np.array([-1.1, 0.0, 0.0]), (3, 3, 3, 3)),
                axis=2,
            ),
            BCTestParameters(
                BiasCorrectionAlgos.BIAS_CORRECTION,
                BCStatsCollectors.MEAN,
                TargetType.OPERATION_WITH_WEIGHTS,
                (np.array([-1.1, 0.0, 0.0]), (3, 3, 3, 3)),
                axis=2,
            ),
        ],
    )
    def test_statistics_aggregator_bias_correction(
        self, dataset_samples, test_params: BCTestParameters, inplace_statistics
    ):
        name_to_algo_backend_map = {
            BiasCorrectionAlgos.BIAS_CORRECTION: self.get_bias_correction_algo_backend_cls,
            BiasCorrectionAlgos.FAST_BIAS_CORRECTION: self.get_fast_bias_correction_algo_backend_cls,
        }
        algo_backend = name_to_algo_backend_map[test_params.algo]()
        if test_params.collector_type == BCStatsCollectors.MEAN:
            tensor_collector = algo_backend.mean_statistic_collector(
                test_params.axis, inplace_statistics, len(dataset_samples)
            )
        elif test_params.collector_type == BCStatsCollectors.RAW:
            tensor_collector = algo_backend.raw_statistic_collector(len(dataset_samples))
        else:
            raise nncf.InvalidCollectorTypeError(f"Invalid collector type: {test_params.collector_type}")

        target_point = self.get_target_point(test_params.target_type)

        statistics_points = StatisticPointsContainer()
        algorithm_name = "TestAlgo"
        statistics_points.add_statistic_point(
            StatisticPoint(target_point=target_point, tensor_collector=tensor_collector, algorithm=algorithm_name)
        )
        dataset = self.get_dataset(dataset_samples)
        statistics_aggregator = self.get_statistics_aggregator(dataset)
        statistics_aggregator.register_statistic_points(statistics_points)
        model = self.get_backend_model(dataset_samples)
        graph = NNCFGraphFactory.create(model)
        statistics_aggregator.collect_statistics(model, graph)

        def filter_func(point):
            return (
                algorithm_name in point.algorithm_to_tensor_collectors and point.target_point.type == target_point.type
            )

        tensor_collectors = list(
            statistics_points.get_algo_statistics_for_node(target_point.target_node_name, filter_func, algorithm_name)
        )
        assert len(tensor_collectors) == 1

        for tensor_collector in tensor_collectors:
            stat = tensor_collector.get_statistics()
            if test_params.collector_type == BCStatsCollectors.MEAN:
                ret_val = [stat.mean_values, stat.shape]
            elif test_params.collector_type == BCStatsCollectors.RAW:
                ret_val = stat.values
                test_params.ref_values = dataset_samples
            else:
                raise nncf.InvalidCollectorTypeError(f"Invalid collector type: {test_params.collector_type}")

            for val, ref in zip(ret_val, test_params.ref_values):
                if isinstance(ref, np.ndarray):
                    assert ref.shape == val.shape

                if isinstance(val, tuple):
                    assert val == ref
                else:
                    assert np.allclose(val.data, ref)

    @classmethod
    def create_statistics_point(
        cls, model, q_config, target_point, subset_size, algorithm_name, inplace_statistics, range_estimator, mocker
    ):
        _ = mocker.patch(
            "nncf.quantization.algorithms.min_max.algorithm.MinMaxQuantization._get_range_estimator_parameters",
            return_value=range_estimator,
        )
        algo = cls.get_min_max_algo_cls()(
            subset_size=subset_size,
            inplace_statistics=inplace_statistics,
        )
        algo._set_backend_entity(model)
        nncf_graph = NNCFGraphFactory.create(model)
        algo._subset_size = subset_size
        tensor_collector = algo._get_stat_collector(nncf_graph, target_point, q_config, False)
        return StatisticPoint(target_point=target_point, tensor_collector=tensor_collector, algorithm=algorithm_name)

    @pytest.mark.parametrize(
        "statistic_point_params",
        (
            (
                ("AAA", RangeEstimatorParametersSet.MINMAX, TargetType.PRE_LAYER_OPERATION, -128.0, 128),
                ("BBB", RangeEstimatorParametersSet.MINMAX, TargetType.POST_LAYER_OPERATION, -128.0, 128),
                ("CCC", RangeEstimatorParametersSet.MEAN_MINMAX, TargetType.POST_LAYER_OPERATION, -63.5, 64.5),
            ),
        ),
    )
    def test_statistics_merging_simple(self, dataset_samples, inplace_statistics, statistic_point_params, mocker):
        model = self.get_backend_model(dataset_samples)
        quantizer_config = QuantizerConfig(mode=QuantizationMode.SYMMETRIC, per_channel=False)
        subset_size = len(dataset_samples)

        statistics_points = StatisticPointsContainer()
        ref_val = {}

        for statistic_point_param in statistic_point_params:
            algorithm_name, range_estimator, target_point_type, ref_min_val, ref_max_val = statistic_point_param
            ref_val[algorithm_name] = (ref_min_val, ref_max_val)
            target_point = self.get_target_point(target_point_type)
            statistics_point = self.create_statistics_point(
                model,
                quantizer_config,
                target_point,
                subset_size,
                algorithm_name,
                inplace_statistics,
                range_estimator,
                mocker,
            )
            statistics_points.add_statistic_point(statistics_point)

        dataset = self.get_dataset(dataset_samples)
        statistics_aggregator = self.get_statistics_aggregator(dataset)
        statistics_aggregator.register_statistic_points(statistics_points)
        graph = NNCFGraphFactory.create(model)
        statistics_aggregator.collect_statistics(model, graph)

        tensor_collectors = list(statistics_points.get_tensor_collectors())
        assert len(tensor_collectors) == 3
        for algorithm, _, tensor_collector in tensor_collectors:
            stat = tensor_collector.get_statistics()
            ref_min_val, ref_max_val = ref_val[algorithm]
            assert fns.allclose(stat.min_values, ref_min_val)
            assert fns.allclose(stat.max_values, ref_max_val)

    @classmethod
    def _check_static_point_common(cls, stat_point, ref_type=TargetType.POST_LAYER_OPERATION):
        assert stat_point.target_point.type == ref_type
        assert len(stat_point.algorithm_to_tensor_collectors["Merged"]) == 1
        stat_collector = stat_point.algorithm_to_tensor_collectors["Merged"][0]
        assert len(stat_collector.reducers) == 2
        assert len(stat_collector.aggregators) == 4

    @classmethod
    def _check_split_concat_merged_stats(cls, merged_statistics):
        assert len(merged_statistics) == 5
        assert len(merged_statistics["split"]) == 3
        port_ids = set()
        for stat_point in merged_statistics["split"]:
            cls._check_static_point_common(stat_point)
            port_ids.add(stat_point.target_point.port_id)

        assert sorted(list(port_ids)) == [0, 1, 2]
        for key in ["add_1", "add_2", "add_3", "concat"]:
            assert len(merged_statistics[key]) == 1
            cls._check_static_point_common(merged_statistics[key][0])

    @classmethod
    def _check_shared_convs_merged_stats(cls, merged_statistics):
        assert len(merged_statistics) == 1
        assert len(merged_statistics["Conv_1"]) == 1
        stat_point = merged_statistics["Conv_1"][0]
        cls._check_static_point_common(stat_point, TargetType.OPERATION_WITH_WEIGHTS)
        assert stat_point.target_point.port_id == 1

    MERGED_TARGET_POINT_AND_REFS = {
        "split_concat": [
            # Split output target points
            ((TargetType.POST_LAYER_OPERATION, "split", 0), {"min_max": (-10, 10), "mean_min_max": (-4.5, 5.5)}),
            ((TargetType.PRE_LAYER_OPERATION, "add_1", 0), {"min_max": (-10, 10), "mean_min_max": (-4.5, 5.5)}),
            ((TargetType.POST_LAYER_OPERATION, "split", 1), {"min_max": (-1, 1), "mean_min_max": (0, 1)}),
            ((TargetType.PRE_LAYER_OPERATION, "add_2", 0), {"min_max": (-1, 1), "mean_min_max": (0, 1)}),
            ((TargetType.POST_LAYER_OPERATION, "split", 2), {"min_max": (-128, 128), "mean_min_max": (-63.5, 64.5)}),
            ((TargetType.PRE_LAYER_OPERATION, "add_3", 0), {"min_max": (-128, 128), "mean_min_max": (-63.5, 64.5)}),
            # Concat input target points
            ((TargetType.POST_LAYER_OPERATION, "add_1", 0), {"min_max": (-9, 9), "mean_min_max": (-3.5, 5.5)}),
            ((TargetType.PRE_LAYER_OPERATION, "concat", 0), {"min_max": (-9, 9), "mean_min_max": (-3.5, 5.5)}),
            ((TargetType.POST_LAYER_OPERATION, "add_2", 0), {"min_max": (0, 2), "mean_min_max": (1, 1.55)}),
            ((TargetType.PRE_LAYER_OPERATION, "concat", 1), {"min_max": (0, 2), "mean_min_max": (1, 1.55)}),
            ((TargetType.POST_LAYER_OPERATION, "add_3", 0), {"min_max": (-127, 129), "mean_min_max": (-62.5, 65.5)}),
            ((TargetType.PRE_LAYER_OPERATION, "concat", 2), {"min_max": (-127, 129), "mean_min_max": (-62.5, 65.5)}),
            # One output to Several branch target points
            ((TargetType.POST_LAYER_OPERATION, "concat", 0), {"min_max": (-127, 129), "mean_min_max": (-62.5, 65.5)}),
            ((TargetType.PRE_LAYER_OPERATION, "add_4", 0), {"min_max": (-127, 129), "mean_min_max": (-62.5, 65.5)}),
            ((TargetType.PRE_LAYER_OPERATION, "add_5", 0), {"min_max": (-127, 129), "mean_min_max": (-62.5, 65.5)}),
        ],
        "shared_conv": [
            ((TargetType.OPERATION_WITH_WEIGHTS, "Conv_1", 1), {"min_max": (-128, 128), "mean_min_max": (-128, 128)}),
            ((TargetType.OPERATION_WITH_WEIGHTS, "Conv_2", 1), {"min_max": (-128, 128), "mean_min_max": (-128, 128)}),
        ],
    }

    @pytest.mark.parametrize("key", ["split_concat", "shared_conv"])
    def test_statistic_merging(self, test_params, key, dataset_samples, inplace_statistics, mocker):
        params = test_params["test_statistic_merging"][key]
        model = params["model"](dataset_samples)
        nncf_graph = NNCFGraphFactory.create(model)

        quantizer_config = QuantizerConfig(mode=QuantizationMode.SYMMETRIC, per_channel=False)
        statistics_points = StatisticPointsContainer()
        target_point_cls = self.get_target_point_cls()
        sp_and_refs = []
        for target_point_args, ref in self.MERGED_TARGET_POINT_AND_REFS[key]:
            target_point = target_point_cls(*target_point_args)
            for estimator, ref_val in (
                (RangeEstimatorParametersSet.MINMAX, ref["min_max"]),
                (RangeEstimatorParametersSet.MEAN_MINMAX, ref["mean_min_max"]),
            ):
                s_p = self.create_statistics_point(
                    model,
                    quantizer_config,
                    target_point,
                    len(dataset_samples),
                    "TEST",
                    inplace_statistics,
                    estimator,
                    mocker,
                )
                statistics_points.add_statistic_point(s_p)
                sp_and_refs.append((s_p, ref_val))

        dataset = self.get_dataset(dataset_samples)
        statistics_aggregator = self.get_statistics_aggregator(dataset)

        merged_statistics = statistics_aggregator._get_merged_statistic_points(statistics_points, model, nncf_graph)
        merged_stats_checkers_map = {
            "split_concat": self._check_split_concat_merged_stats,
            "shared_conv": self._check_shared_convs_merged_stats,
        }
        merged_stats_checkers_map[key](merged_statistics)

        statistics_aggregator.register_statistic_points(statistics_points)
        statistics_aggregator.collect_statistics(model, nncf_graph)

        for sp, ref in sp_and_refs:
            collector = sp.algorithm_to_tensor_collectors["TEST"][0]
            stat = collector.get_statistics()
            assert fns.allclose(stat.min_values, ref[0])
            assert fns.allclose(stat.max_values, ref[1])

            if isinstance(ref[0], np.ndarray):
                assert stat.min_values.shape == ref[0].shape
                assert stat.max_values.shape == ref[1].shape

    @pytest.mark.parametrize(
        "statistics_type",
        [
            StatisticsType.MIN,
            StatisticsType.MAX,
            StatisticsType.ABS_MAX,
            StatisticsType.MEAN,
            StatisticsType.QUANTILE,
            StatisticsType.ABS_QUANTILE,
            "batch_mean",
            "mean_per_ch",
        ],
    )
    def test_same_collectors_different_attrs_dont_merge(self, statistics_type, test_params, dataset_samples):
        params = test_params["test_statistic_merging"]["split_concat"]
        model = params["model"](dataset_samples)
        params = {}
        if statistics_type in [StatisticsType.MIN, StatisticsType.MAX, StatisticsType.ABS_MAX, StatisticsType.MEAN]:
            params["reduction_axes"] = [None, (0, 1, 3), (1, 2, 3)]
            params["inplace"] = [False, True]
        elif statistics_type in [StatisticsType.QUANTILE, StatisticsType.ABS_QUANTILE]:
            params["reduction_axes"] = [None, (0, 1, 3), (1, 2, 3)]
            params["quantile"] = [[0.01, 0.99], [0.001, 0.999]]
        elif statistics_type == "batch_mean":
            params["inplace"] = [False, True]
        elif statistics_type == "mean_per_ch":
            params["inplace"] = [False, True]
            params["channel_axis"] = [1, 2]

        def product_dict(**kwargs):
            keys = kwargs.keys()
            for instance in product(*kwargs.values()):
                yield dict(zip(keys, instance))

        tensor_collector = TensorCollector()
        statistics_points = StatisticPointsContainer()
        target_point_cls = self.get_target_point_cls()
        target_point_args = (TargetType.POST_LAYER_OPERATION, "split", 0)
        for params_ in product_dict(**params):
            reducer = self.reducers_map()[statistics_type](**params_)
            aggregator = NoopAggregator(1)
            tensor_collector.register_statistic_branch(str(params_), reducer, aggregator)
            target_point = target_point_cls(*target_point_args)
            stat_point = StatisticPoint(target_point, tensor_collector, "TEST")
            statistics_points.add_statistic_point(stat_point)

        dataset = self.get_dataset(dataset_samples)
        statistics_aggregator = self.get_statistics_aggregator(dataset)
        statistics_aggregator.register_statistic_points(statistics_points)
        # Run statistic collection to check output names matches reducer names
        graph = NNCFGraphFactory.create(model)
        statistics_aggregator.collect_statistics(model, graph)

    @pytest.mark.parametrize(
        "statistic_point_params",
        (
            (
                ("AAA", RangeEstimatorParametersSet.MINMAX, TargetType.PRE_LAYER_OPERATION, 100),
                ("BBB", RangeEstimatorParametersSet.MINMAX, TargetType.POST_LAYER_OPERATION, 10),
                ("CCC", RangeEstimatorParametersSet.MEAN_MINMAX, TargetType.PRE_LAYER_OPERATION, None),
                ("CCC", RangeEstimatorParametersSet.MEAN_MINMAX, TargetType.PRE_LAYER_OPERATION, -1),
            ),
        ),
    )
    def test_register_statistics(self, dataset_samples, statistic_point_params, mocker):
        model = self.get_backend_model(dataset_samples)
        quantizer_config = QuantizerConfig(mode=QuantizationMode.SYMMETRIC, per_channel=False)
        statistics_points = StatisticPointsContainer()
        ref_val = {}

        for statistic_point_param in statistic_point_params:
            algorithm_name, range_estimator, target_point_type, subset_size = statistic_point_param
            ref_val[algorithm_name] = subset_size
            target_point = self.get_target_point(target_point_type)
            statistics_point = self.create_statistics_point(
                model, quantizer_config, target_point, subset_size, algorithm_name, True, range_estimator, mocker
            )
            statistics_points.add_statistic_point(statistics_point)

        dataset = self.get_dataset(dataset_samples)
        statistics_aggregator = self.get_statistics_aggregator(dataset)
        statistics_aggregator.register_statistic_points(statistics_points)
        assert Counter(statistics_points) == Counter(statistics_aggregator.statistic_points)
        ref_subset_size = None
        for subset_size in ref_val.values():
            if subset_size and ref_subset_size:
                ref_subset_size = max(ref_subset_size, subset_size)
            else:
                ref_subset_size = subset_size
        assert statistics_aggregator.stat_subset_size == ref_subset_size

    def test_collect_with_empty_dataset_no_len(self, dataset_samples):
        """
        Checks a correct raising of an error when dataset has no elements to iterate.
        """
        model = self.get_backend_model(dataset_samples)
        dummy_statistic_point = StatisticPoint(
            target_point=self.get_target_point(TargetType.POST_LAYER_OPERATION),
            tensor_collector=TensorCollector(),
            algorithm="dummy",
        )
        statistics_points = StatisticPointsContainer()
        statistics_points.add_statistic_point(dummy_statistic_point)
        dataset = nncf.Dataset(MockedDataset())
        graph = NNCFGraphFactory.create(model)
        statistics_aggregator = self.get_statistics_aggregator(dataset)
        statistics_aggregator.register_statistic_points(statistics_points)
        with pytest.raises(nncf.ValidationError) as e:
            statistics_aggregator.collect_statistics(model, graph)
        assert EMPTY_DATASET_ERROR in str(e)
