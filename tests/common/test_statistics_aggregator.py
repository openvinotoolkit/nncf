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
from dataclasses import dataclass
from enum import Enum
from typing import Any, Type, Union

import numpy as np
import pytest

from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.quantization.algorithms.bias_correction.backend import BiasCorrectionAlgoBackend
from nncf.quantization.algorithms.fast_bias_correction.backend import FastBiasCorrectionAlgoBackend
from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.range_estimator import AggregatorType
from nncf.quantization.range_estimator import RangeEstimatorParameters
from nncf.quantization.range_estimator import RangeEstimatorParametersSet
from nncf.quantization.range_estimator import StatisticsCollectorParameters
from nncf.quantization.range_estimator import StatisticsType


class TemplateTestStatisticsAggregator:
    @abstractmethod
    def get_min_max_algo_backend_cls(self) -> Type[MinMaxAlgoBackend]:
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

    @abstractmethod
    def get_target_point(self, target_type: TargetType) -> TargetPoint:
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
    def is_stat_in_shape_of_scale(self) -> bool:
        pass

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
        pass

    @pytest.fixture
    def dataset_values(self):
        return [{"max": 1, "min": -10}, {"max": 0.1, "min": -1}, {"max": 128, "min": -128}]

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
                (
                    MinMaxTestParameters(
                        RangeEstimatorParametersSet.MINMAX,
                        TargetType.OPERATION_WITH_WEIGHTS,
                        QuantizationMode.SYMMETRIC,
                        False,
                        128,
                        -128,
                    )
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
        is_stat_in_shape_of_scale,
        inplace_statistics,
        is_backend_support_custom_estimators,
    ):
        algo_backend = self.get_min_max_algo_backend_cls()
        model = self.get_backend_model(dataset_samples)
        nncf_graph = NNCFGraphFactory.create(model)

        quantizer_config = QuantizerConfig(
            mode=test_parameters.quantization_mode, per_channel=test_parameters.per_channel
        )
        target_point = self.get_target_point(test_parameters.target_type)

        is_standart_estimator = test_parameters.range_estimator_params in [
            RangeEstimatorParametersSet.MINMAX,
            RangeEstimatorParametersSet.MEAN_MINMAX,
        ]
        if not is_standart_estimator and not is_backend_support_custom_estimators:
            pytest.skip("Custom estimators are not supported for this backend yet")

        tensor_collector = algo_backend.get_statistic_collector(
            test_parameters.range_estimator_params,
            nncf_graph=nncf_graph,
            target_point=target_point,
            quantizer_config=quantizer_config,
            num_samples=len(dataset_samples),
            inplace=inplace_statistics,
        )

        statistics_points = StatisticPointsContainer()
        algorithm_name = "TestAlgo"
        statistics_points.add_statistic_point(
            StatisticPoint(target_point=target_point, tensor_collector=tensor_collector, algorithm=algorithm_name)
        )
        dataset = self.get_dataset(dataset_samples)
        statistics_aggregator = self.get_statistics_aggregator(dataset)
        statistics_aggregator.register_statistic_points(statistics_points)
        statistics_aggregator.collect_statistics(model)

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
            if isinstance(ref_min_val, np.ndarray) and is_stat_in_shape_of_scale:
                shape = (1, 3, 1, 1)
                if test_parameters.target_type == TargetType.OPERATION_WITH_WEIGHTS:
                    shape = (3, 1, 1, 1)
                ref_min_val, ref_max_val = map(lambda x: np.reshape(x, shape), (ref_min_val, ref_max_val))

            assert np.allclose(stat.min_values, ref_min_val)
            assert np.allclose(stat.max_values, ref_max_val)
            if isinstance(ref_min_val, np.ndarray):
                assert stat.min_values.shape == ref_min_val.shape
                assert stat.max_values.shape == ref_max_val.shape

    class BiasCorrectionAlgos(Enum):
        BIAS_CORRECTION = "bias_correction"
        FAST_BIAS_CORRECTION = "fast_bias_correction"

    class BCStatsCollectors(Enum):
        MEAN = "mean"
        BATCH_MEAN = "batch_mean"

    @dataclass
    class BCTestParameters:
        algo: "BiasCorrectionAlgos"
        collector_type: "BCStatsCollectors"
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
                BiasCorrectionAlgos.BIAS_CORRECTION, BCStatsCollectors.BATCH_MEAN, TargetType.POST_LAYER_OPERATION
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
        self, dataset_samples, test_params: BCTestParameters, inplace_statistics, is_stat_in_shape_of_scale
    ):
        name_to_algo_backend_map = {
            self.BiasCorrectionAlgos.BIAS_CORRECTION: self.get_bias_correction_algo_backend_cls,
            self.BiasCorrectionAlgos.FAST_BIAS_CORRECTION: self.get_fast_bias_correction_algo_backend_cls,
        }
        algo_backend = name_to_algo_backend_map[test_params.algo]()
        if test_params.collector_type == self.BCStatsCollectors.MEAN:
            tensor_collector = algo_backend.mean_statistic_collector(
                test_params.axis, inplace_statistics, len(dataset_samples)
            )
        elif test_params.collector_type == self.BCStatsCollectors.BATCH_MEAN:
            tensor_collector = algo_backend.batch_statistic_collector(inplace_statistics, len(dataset_samples))
        else:
            raise RuntimeError()

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
        statistics_aggregator.collect_statistics(model)

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
            if test_params.collector_type == self.BCStatsCollectors.MEAN:
                ret_val = [stat.mean_values, stat.shape]
            elif test_params.collector_type == self.BCStatsCollectors.BATCH_MEAN:
                ret_val = stat.values
                test_params.ref_values = dataset_samples
                if not is_stat_in_shape_of_scale:
                    ret_val = [np.squeeze(x) for x in ret_val]
            else:
                raise RuntimeError()

            for val, ref in zip(ret_val, test_params.ref_values):
                if isinstance(ref, np.ndarray):
                    assert ref.shape == val.shape
                assert np.allclose(val, ref)

    def test_statistics_merging_simple(self, dataset_samples, inplace_statistics):
        algo_backend = self.get_min_max_algo_backend_cls()
        model = self.get_backend_model(dataset_samples)
        nncf_graph = NNCFGraphFactory.create(model)

        quantizer_config = QuantizerConfig(mode=QuantizationMode.SYMMETRIC, per_channel=False)
        pre_layer_target_point = self.get_target_point(TargetType.PRE_LAYER_OPERATION)
        pre_tensor_collector = algo_backend.get_statistic_collector(
            RangeEstimatorParametersSet.MINMAX,
            nncf_graph=nncf_graph,
            target_point=pre_layer_target_point,
            quantizer_config=quantizer_config,
            num_samples=len(dataset_samples),
            inplace=inplace_statistics,
        )

        post_layer_target_point = self.get_target_point(TargetType.POST_LAYER_OPERATION)
        post_tensor_collector = algo_backend.get_statistic_collector(
            RangeEstimatorParametersSet.MINMAX,
            nncf_graph=nncf_graph,
            target_point=post_layer_target_point,
            quantizer_config=quantizer_config,
            num_samples=len(dataset_samples),
            inplace=inplace_statistics,
        )
        unique_post_tensor_collector = algo_backend.get_statistic_collector(
            RangeEstimatorParametersSet.MEAN_MINMAX,
            nncf_graph=nncf_graph,
            target_point=post_layer_target_point,
            quantizer_config=quantizer_config,
            num_samples=len(dataset_samples),
            inplace=inplace_statistics,
        )

        statistics_points = StatisticPointsContainer()
        algorithm_names = ["AAA", "BBB", "CCC"]
        statistics_points.add_statistic_point(
            StatisticPoint(
                target_point=pre_layer_target_point, tensor_collector=pre_tensor_collector, algorithm=algorithm_names[0]
            )
        )
        statistics_points.add_statistic_point(
            StatisticPoint(
                target_point=post_layer_target_point,
                tensor_collector=post_tensor_collector,
                algorithm=algorithm_names[1],
            )
        )
        statistics_points.add_statistic_point(
            StatisticPoint(
                target_point=post_layer_target_point,
                tensor_collector=unique_post_tensor_collector,
                algorithm=algorithm_names[2],
            )
        )
        dataset = self.get_dataset(dataset_samples)
        statistics_aggregator = self.get_statistics_aggregator(dataset)
        statistics_aggregator.register_statistic_points(statistics_points)
        statistics_aggregator.collect_statistics(model)

        tensor_collectors = list(statistics_points.get_tensor_collectors())
        assert len(tensor_collectors) == 3
        for _, _, tensor_collector in tensor_collectors:
            stat = tensor_collector.get_statistics()
            ref_min_val, ref_max_val = -128.0, 128
            if tensor_collector is unique_post_tensor_collector:
                ref_min_val, ref_max_val = -63.5, 64.5
            assert np.allclose(stat.min_values, ref_min_val)
            assert np.allclose(stat.max_values, ref_max_val)

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
    def test_statistic_merging(self, test_params, key, dataset_samples, inplace_statistics):
        params = test_params["test_statistic_merging"][key]
        model = params["model"](dataset_samples)
        nncf_graph = NNCFGraphFactory.create(model)

        quantizer_config = QuantizerConfig(mode=QuantizationMode.SYMMETRIC, per_channel=False)
        statistics_points = StatisticPointsContainer()
        collectors_and_refs = []
        algo_backend = self.get_min_max_algo_backend_cls()
        target_point_cls = self.get_target_point_cls()
        for target_point_args, ref in self.MERGED_TARGET_POINT_AND_REFS[key]:
            target_point = target_point_cls(*target_point_args)
            min_max_tensor_collector = algo_backend.get_statistic_collector(
                RangeEstimatorParametersSet.MINMAX,
                nncf_graph=nncf_graph,
                target_point=target_point,
                quantizer_config=quantizer_config,
                num_samples=len(dataset_samples),
                inplace=inplace_statistics,
            )
            mean_min_max_tensor_collector = algo_backend.get_statistic_collector(
                RangeEstimatorParametersSet.MEAN_MINMAX,
                nncf_graph=nncf_graph,
                target_point=target_point,
                quantizer_config=quantizer_config,
                num_samples=len(dataset_samples),
                inplace=inplace_statistics,
            )

            for tensor_collector in [min_max_tensor_collector, mean_min_max_tensor_collector]:
                stat_point = StatisticPoint(target_point, tensor_collector, "TEST")
                statistics_points.add_statistic_point(stat_point)
            collectors_and_refs.append((min_max_tensor_collector, ref["min_max"]))
            collectors_and_refs.append((mean_min_max_tensor_collector, ref["mean_min_max"]))

        dataset = self.get_dataset(dataset_samples)
        statistics_aggregator = self.get_statistics_aggregator(dataset)
        # pylint: disable=protected-access
        merged_statistics = statistics_aggregator._get_merged_statistic_points(statistics_points, model)
        merged_stats_checkers_map = {
            "split_concat": self._check_split_concat_merged_stats,
            "shared_conv": self._check_shared_convs_merged_stats,
        }
        merged_stats_checkers_map[key](merged_statistics)

        statistics_aggregator.register_statistic_points(statistics_points)
        statistics_aggregator.collect_statistics(model)

        for collector, ref in collectors_and_refs:
            stat = collector.get_statistics()
            assert np.allclose(stat.min_values, ref[0])
            assert np.allclose(stat.max_values, ref[1])

            if isinstance(ref[0], np.ndarray):
                assert stat.min_values.shape == ref[0].shape
                assert stat.max_values.shape == ref[1].shape
