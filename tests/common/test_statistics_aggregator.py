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

import pytest
import numpy as np
from abc import abstractmethod
from typing import Union
from dataclasses import dataclass

from nncf.common.factory import NNCFGraphFactory
from nncf.quantization.algorithms.definitions import RangeType
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.graph.transformations.commands import TargetType
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
    @pytest.fixture
    def is_stat_in_shape_of_scale(self) -> bool:
        pass

    @abstractmethod
    @pytest.fixture
    def dataset_samples(self):
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
    def test_statistics_aggregator(self, test_parameters: TestParameters, dataset_samples, is_stat_in_shape_of_scale):
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
                                                                       num_samples=len(dataset_samples))
        if test_parameters.range_type == RangeType.MEAN_MINMAX:
            tensor_collector = algo_backend.mean_minmax_statistic_collector(nncf_graph=nncf_graph,
                                                                            target_point=target_point,
                                                                            quantizer_config=quantizer_config,
                                                                            use_per_sample_stats=False,
                                                                            num_samples=len(dataset_samples))
        statistics_points = StatisticPointsContainer()
        algorithm_name = 'TestAlgo'
        statistics_points.add_statistic_point(StatisticPoint(target_point=target_point,
                                                             tensor_collector=tensor_collector,
                                                             algorithm=algorithm_name))
        dataset = self.get_dataset(dataset_samples)
        statistics_aggregator = self.get_statistics_aggregator(dataset)
        statistics_aggregator.register_stastistic_points(statistics_points)
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
