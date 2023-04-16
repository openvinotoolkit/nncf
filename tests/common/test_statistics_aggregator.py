"""
 Copyright (c) 2023 Intel Corporation
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

from abc import abstractmethod
from dataclasses import dataclass
from typing import List
from typing import Union

import numpy as np
import pytest

from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.quantization.algorithms.definitions import RangeType
from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend


class TemplateTestStatisticsAggregator:
    @abstractmethod
    def get_algo_backend_cls(self) -> MinMaxAlgoBackend:
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
    @pytest.fixture(scope='session')
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

    @pytest.fixture
    def dataset_values(self):
        return [{'max': 1, 'min': -10},
                {'max': 0.1, 'min': -1},
                {'max': 128, 'min': -128}]

    @dataclass
    class TestParameters:
        range_type: RangeType
        target_type: TargetType
        quantization_mode: QuantizationMode
        per_channel: bool
        ref_max_val: Union[np.ndarray, float]
        ref_min_val: Union[np.ndarray, float]

    def dataset_samples_to_conv_w(self, dataset_sample):
        # Layout: [O, I, K, K]
        d = dataset_sample
        in_ch = d.shape[0]
        return np.stack([np.stack([d[i]] * in_ch, axis=0) for i in range(in_ch)], axis=0)

    @pytest.mark.parametrize('test_parameters, ',
                              # Activation collectors
                             (
                              (TestParameters(RangeType.MEAN_MINMAX, TargetType.POST_LAYER_OPERATION,
                                           QuantizationMode.ASYMMETRIC, False, 64.5, -63.5)),
                              (TestParameters(RangeType.MEAN_MINMAX, TargetType.POST_LAYER_OPERATION,
                                              QuantizationMode.ASYMMETRIC, True,
                                              np.array((1, 0.55, 64.5)), np.array((-4.5, 0, -63.5)))),
                              (TestParameters(RangeType.MEAN_MINMAX, TargetType.POST_LAYER_OPERATION,
                                              QuantizationMode.SYMMETRIC, True,
                                              np.array((5.5, 1, 64.5)), np.array((-4.5, 0, -63.5)))),
                              (TestParameters(RangeType.MINMAX, TargetType.POST_LAYER_OPERATION,
                                              QuantizationMode.ASYMMETRIC, False, 128, -128)),
                              (TestParameters(RangeType.MINMAX, TargetType.POST_LAYER_OPERATION,
                                              QuantizationMode.SYMMETRIC, False, 128, -128)),
                              (TestParameters(RangeType.MINMAX, TargetType.POST_LAYER_OPERATION,
                                              QuantizationMode.ASYMMETRIC, True,
                                              np.array((1, 1, 128)), np.array((-10, -1, -128)))),
                              (TestParameters(RangeType.MINMAX, TargetType.POST_LAYER_OPERATION,
                                              QuantizationMode.SYMMETRIC, True,
                                              np.array((10, 1, 128)), np.array((-10, -1, -128)))),
                              # Weight collectors
                              ((TestParameters(RangeType.MINMAX, TargetType.OPERATION_WITH_WEIGHTS,
                                              QuantizationMode.SYMMETRIC, False, 128, -128))),
                              (TestParameters(RangeType.MINMAX, TargetType.OPERATION_WITH_WEIGHTS,
                                              QuantizationMode.ASYMMETRIC, False, 128, -128)),
                              (TestParameters(RangeType.MINMAX, TargetType.OPERATION_WITH_WEIGHTS,
                                              QuantizationMode.SYMMETRIC, True,
                                              np.array((10, 1, 128)), np.array((-10, -1, -128)))),
                              (TestParameters(RangeType.MINMAX, TargetType.OPERATION_WITH_WEIGHTS,
                                              QuantizationMode.ASYMMETRIC, True,
                                              np.array((1, 0.1, 128)), np.array((-10, -1, -128)))),
                             ))
    def test_statistics_aggregator(self, test_parameters: TestParameters, dataset_samples, is_stat_in_shape_of_scale,
                                   inplace_statistics):
        algo_backend = self.get_algo_backend_cls()
        model = self.get_backend_model(dataset_samples)
        nncf_graph = NNCFGraphFactory.create(model)

        quantizer_config = QuantizerConfig(mode=test_parameters.quantization_mode,
                                           per_channel=test_parameters.per_channel)
        target_point = self.get_target_point(test_parameters.target_type)
        if test_parameters.range_type == RangeType.MINMAX:
            tensor_collector = algo_backend.minmax_statistic_collector(nncf_graph=nncf_graph,
                                                                       target_point=target_point,
                                                                       quantizer_config=quantizer_config,
                                                                       num_samples=len(dataset_samples),
                                                                       inplace=inplace_statistics)
        if test_parameters.range_type == RangeType.MEAN_MINMAX:
            tensor_collector = algo_backend.mean_minmax_statistic_collector(nncf_graph=nncf_graph,
                                                                            target_point=target_point,
                                                                            quantizer_config=quantizer_config,
                                                                            use_per_sample_stats=False,
                                                                            num_samples=len(dataset_samples),
                                                                            inplace=inplace_statistics)
        statistics_points = StatisticPointsContainer()
        algorithm_name = 'TestAlgo'
        statistics_points.add_statistic_point(StatisticPoint(target_point=target_point,
                                                             tensor_collector=tensor_collector,
                                                             algorithm=algorithm_name))
        dataset = self.get_dataset(dataset_samples)
        statistics_aggregator = self.get_statistics_aggregator(dataset)
        statistics_aggregator.register_statistic_points(statistics_points)
        statistics_aggregator.collect_statistics(model)

        def filter_func(point):
            return algorithm_name in point.algorithm_to_tensor_collectors and \
                   point.target_point.type == target_point.type

        for tensor_collector in statistics_points.get_algo_statistics_for_node(
                target_point.target_node_name,
                filter_func,
                algorithm_name):
            stat = tensor_collector.get_statistics()
            # Torch and Openvino backends tensor collectors return values in shape of scale
            # in comparison with ONNX backends.
            ref_min_val, ref_max_val = test_parameters.ref_min_val, test_parameters.ref_max_val
            if isinstance(ref_min_val, np.ndarray) and is_stat_in_shape_of_scale:
                shape = (1, 3, 1, 1)
                if test_parameters.target_type == TargetType.OPERATION_WITH_WEIGHTS:
                    shape = (3, 1, 1, 1)
                ref_min_val, ref_max_val = map(lambda x: np.reshape(x, shape), (ref_min_val, ref_max_val))

            assert np.allclose(stat.min_values, ref_min_val)
            assert np.allclose(stat.max_values, ref_max_val)

    def test_statistics_merging_simple(self, dataset_samples, inplace_statistics):
        algo_backend = self.get_algo_backend_cls()
        model = self.get_backend_model(dataset_samples)
        nncf_graph = NNCFGraphFactory.create(model)

        quantizer_config = QuantizerConfig(mode=QuantizationMode.SYMMETRIC,
                                           per_channel=False)
        pre_layer_target_point = self.get_target_point(TargetType.PRE_LAYER_OPERATION)
        pre_tensor_collector = algo_backend.minmax_statistic_collector(
            nncf_graph=nncf_graph,
            target_point=pre_layer_target_point,
            quantizer_config=quantizer_config,
            num_samples=len(dataset_samples),
            inplace=inplace_statistics)

        post_layer_target_point = self.get_target_point(TargetType.POST_LAYER_OPERATION)
        post_tensor_collector = algo_backend.minmax_statistic_collector(
            nncf_graph=nncf_graph,
            target_point=post_layer_target_point,
            quantizer_config=quantizer_config,
            num_samples=len(dataset_samples),
            inplace=inplace_statistics)
        unique_post_tensor_collector = algo_backend.mean_minmax_statistic_collector(
            nncf_graph=nncf_graph,
            target_point=post_layer_target_point,
            quantizer_config=quantizer_config,
            use_per_sample_stats=False,
            num_samples=len(dataset_samples),
            inplace=inplace_statistics)

        statistics_points = StatisticPointsContainer()
        algorithm_names = ['AAA', 'BBB', 'CCC']
        statistics_points.add_statistic_point(StatisticPoint(target_point=pre_layer_target_point,
                                                             tensor_collector=pre_tensor_collector,
                                                             algorithm=algorithm_names[0]))
        statistics_points.add_statistic_point(StatisticPoint(target_point=post_layer_target_point,
                                                             tensor_collector=post_tensor_collector,
                                                             algorithm=algorithm_names[1]))
        statistics_points.add_statistic_point(StatisticPoint(target_point=post_layer_target_point,
                                                             tensor_collector=unique_post_tensor_collector,
                                                             algorithm=algorithm_names[2]))
        dataset = self.get_dataset(dataset_samples)
        statistics_aggregator = self.get_statistics_aggregator(dataset)
        statistics_aggregator.register_statistic_points(statistics_points)
        statistics_aggregator.collect_statistics(model)

        for _, _, tensor_collector in statistics_points.get_tensor_collectors():
            stat = tensor_collector.get_statistics()
            ref_min_val, ref_max_val = -128., 128
            if tensor_collector is unique_post_tensor_collector:
                ref_min_val, ref_max_val = -63.5, 64.5
            assert np.allclose(stat.min_values, ref_min_val)
            assert np.allclose(stat.max_values, ref_max_val)

    @classmethod
    def _check_static_point_common(cls, stat_point,
                                   ref_type=TargetType.POST_LAYER_OPERATION):
        assert stat_point.target_point.type == ref_type
        assert len(stat_point.algorithm_to_tensor_collectors['Merged']) == 1
        stat_collector = stat_point.algorithm_to_tensor_collectors['Merged'][0]
        assert len(stat_collector.reducers) == 2
        assert len(stat_collector.aggregators) == 4

    @classmethod
    def _check_split_concat_merged_stats(cls, merged_statistics):
        assert len(merged_statistics) == 5
        assert len(merged_statistics['split']) == 3
        port_ids = set()
        for stat_point in merged_statistics['split']:
            cls._check_static_point_common(stat_point)
            port_ids.add(stat_point.target_point.port_id)

        assert sorted(list(port_ids)) == [0, 1, 2]
        for key in ['add_1', 'add_2', 'add_3', 'concat']:
            assert len(merged_statistics[key]) == 1
            cls._check_static_point_common(merged_statistics[key][0])

    @classmethod
    def _check_shared_convs_merged_stats(cls, merged_statistics):
        assert len(merged_statistics) == 1
        assert len(merged_statistics['Conv_1']) == 1
        stat_point = merged_statistics['Conv_1'][0]
        cls._check_static_point_common(stat_point, TargetType.OPERATION_WITH_WEIGHTS)
        assert stat_point.target_point.port_id == 1

    MERGED_TARGET_POINT_AND_REFS = {
    'split_concat': [
            # Split output target points
            ((TargetType.POST_LAYER_OPERATION, 'split', 0),
             {'min_max': (-10, 10), 'mean_min_max': (-4.5, 5.5)}),
            ((TargetType.PRE_LAYER_OPERATION, 'add_1', 0),
             {'min_max': (-10, 10), 'mean_min_max': (-4.5, 5.5)}),

            ((TargetType.POST_LAYER_OPERATION, 'split', 1),
             {'min_max': (-1, 1), 'mean_min_max': (0, 1)}),
            ((TargetType.PRE_LAYER_OPERATION, 'add_2', 0),
             {'min_max': (-1, 1), 'mean_min_max': (0, 1)}),

            ((TargetType.POST_LAYER_OPERATION, 'split', 2),
             {'min_max': (-128, 128), 'mean_min_max': (-63.5, 64.5)}),
            ((TargetType.PRE_LAYER_OPERATION, 'add_3', 0),
             {'min_max': (-128, 128), 'mean_min_max': (-63.5, 64.5)}),

            # Concat input target points
            ((TargetType.POST_LAYER_OPERATION, 'add_1', 0),
             {'min_max': (-9, 9), 'mean_min_max': (-3.5, 5.5)}),
            ((TargetType.PRE_LAYER_OPERATION, 'concat', 0),
             {'min_max': (-9, 9), 'mean_min_max': (-3.5, 5.5)}),

            ((TargetType.POST_LAYER_OPERATION, 'add_2', 0),
             {'min_max': (0, 2), 'mean_min_max': (1, 1.55)}),
            ((TargetType.PRE_LAYER_OPERATION, 'concat', 1),
             {'min_max': (0, 2), 'mean_min_max': (1, 1.55)}),

            ((TargetType.POST_LAYER_OPERATION, 'add_3', 0),
             {'min_max': (-127, 129), 'mean_min_max': (-62.5, 65.5)}),
            ((TargetType.PRE_LAYER_OPERATION, 'concat', 2),
             {'min_max': (-127, 129), 'mean_min_max': (-62.5, 65.5)}),

            # One output to Several branch target points
            ((TargetType.POST_LAYER_OPERATION, 'concat', 0),
             {'min_max': (-127, 129), 'mean_min_max': (-62.5, 65.5)}),
            ((TargetType.PRE_LAYER_OPERATION, 'add_4', 0),
             {'min_max': (-127, 129), 'mean_min_max': (-62.5, 65.5)}),
            ((TargetType.PRE_LAYER_OPERATION, 'add_5', 0),
             {'min_max': (-127, 129), 'mean_min_max': (-62.5, 65.5)}),
        ],
        'shared_conv': [
            ((TargetType.OPERATION_WITH_WEIGHTS, 'Conv_1', 1),
             {'min_max': (-128, 128), 'mean_min_max': (-128, 128)}),
            ((TargetType.OPERATION_WITH_WEIGHTS, 'Conv_2', 1),
             {'min_max': (-128, 128), 'mean_min_max': (-128, 128)}),
        ]
    }

    @pytest.mark.parametrize('key', ['split_concat', 'shared_conv'])
    def test_statistic_merging(self, test_params, key, dataset_samples, inplace_statistics):
        params = test_params['test_statistic_merging'][key]
        model = params['model'](dataset_samples)
        nncf_graph = NNCFGraphFactory.create(model)

        quantizer_config = QuantizerConfig(mode=QuantizationMode.SYMMETRIC,
                                           per_channel=False)
        statistics_points = StatisticPointsContainer()
        collectors_and_refs = []
        algo_backend = self.get_algo_backend_cls()
        target_point_cls = self.get_target_point_cls()
        for target_point_args, ref in self.MERGED_TARGET_POINT_AND_REFS[key]:
            target_point = target_point_cls(*target_point_args)
            min_max_tensor_collector = algo_backend.minmax_statistic_collector(
                nncf_graph=nncf_graph,
                target_point=target_point,
                quantizer_config=quantizer_config,
                num_samples=len(dataset_samples),
                inplace=inplace_statistics)
            mean_min_max_tensor_collector = algo_backend.mean_minmax_statistic_collector(
                nncf_graph=nncf_graph,
                target_point=target_point,
                quantizer_config=quantizer_config,
                use_per_sample_stats=False,
                num_samples=len(dataset_samples),
                inplace=inplace_statistics)

            for tensor_collector in [min_max_tensor_collector, mean_min_max_tensor_collector]:
                stat_point = StatisticPoint(target_point, tensor_collector, 'TEST')
                statistics_points.add_statistic_point(stat_point)
            collectors_and_refs.append((min_max_tensor_collector, ref['min_max']))
            collectors_and_refs.append((mean_min_max_tensor_collector, ref['mean_min_max']))

        dataset = self.get_dataset(dataset_samples)
        statistics_aggregator = self.get_statistics_aggregator(dataset)
        # pylint: disable=protected-access
        merged_statistics = statistics_aggregator._get_merged_statistic_points(statistics_points, model)
        merged_stats_checkers_map = {
            'split_concat': self._check_split_concat_merged_stats,
            'shared_conv': self._check_shared_convs_merged_stats,
        }
        merged_stats_checkers_map[key](merged_statistics)

        statistics_aggregator.register_statistic_points(statistics_points)
        statistics_aggregator.collect_statistics(model)

        for collector, ref in collectors_and_refs:
            stat = collector.get_statistics()
            assert np.allclose(stat.min_values, ref[0])
            assert np.allclose(stat.max_values, ref[1])
