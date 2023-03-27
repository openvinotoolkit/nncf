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
from abc import abstractmethod
from dataclasses import dataclass
from typing import List

from nncf.common.graph.patterns import GraphPattern
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizationPoint
from nncf.common.quantization.quantizer_setup import WeightQuantizationInsertionPoint
from nncf.common.quantization.quantizer_setup import ActivationQuantizationInsertionPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.quantization.structs import QuantizationMode
from nncf.quantization.algorithms.definitions import RangeType
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantizationParameters
from nncf.quantization.algorithms.definitions import Granularity

from tests.post_training.models import NNCFGraphToTest
from tests.post_training.models import NNCFGraphToTestDepthwiseConv
from tests.post_training.models import NNCFGraphToTestSumAggregation


class TemplateTestQuantizerConfig:
    @abstractmethod
    def get_algo_backend(self):
        pass

    @abstractmethod
    def get_min_max_statistic_collector_cls(self):
        pass

    @abstractmethod
    def get_mean_max_statistic_collector_cls(self):
        pass

    @abstractmethod
    @pytest.fixture
    def single_conv_nncf_graph(self) -> NNCFGraphToTest:
        pass

    @abstractmethod
    @pytest.fixture
    def depthwise_conv_nncf_graph(self) -> NNCFGraphToTestDepthwiseConv:
        pass

    @abstractmethod
    @pytest.fixture
    def conv_sum_aggregation_nncf_graph(self) -> NNCFGraphToTestSumAggregation:
        pass

    @dataclass
    class TestGetStatisticsCollectorParameters:
        target_type: TargetType
        target_node_name: str
        ref_per_ch_reduction_shape: List[int]
        ref_per_tensor_reduction_shape: List[int]

    @abstractmethod
    @pytest.fixture
    def statistic_collector_parameters(self, request) ->\
        TestGetStatisticsCollectorParameters:
        pass

    def test_default_quantizer_config(self, single_conv_nncf_graph):
        algo = PostTrainingQuantization(PostTrainingQuantizationParameters())
        min_max_algo = algo.algorithms[0]
        min_max_algo._backend_entity = self.get_algo_backend()
        q_setup = min_max_algo._get_quantizer_setup(single_conv_nncf_graph.nncf_graph, GraphPattern())

        weight_default_config = QuantizerConfig(mode=QuantizationMode.SYMMETRIC,
                                                num_bits=8,
                                                signedness_to_force=True,
                                                per_channel=True)
        activation_default_config = QuantizerConfig(mode=QuantizationMode.SYMMETRIC,
                                                    num_bits=8,
                                                    signedness_to_force=None,
                                                    per_channel=False)

        assert len(q_setup.quantization_points) == 2

        for quantization_point in q_setup.quantization_points.values():
            if quantization_point.is_weight_quantization_point():
                assert quantization_point.qconfig == weight_default_config
            if quantization_point.is_activation_quantization_point():
                assert quantization_point.qconfig == activation_default_config


    @pytest.mark.parametrize('weight_granularity', [Granularity.PERCHANNEL, Granularity.PERTENSOR])
    @pytest.mark.parametrize('activation_granularity', [Granularity.PERTENSOR])
    @pytest.mark.parametrize('preset', [QuantizationPreset.MIXED, QuantizationPreset.PERFORMANCE])
    @pytest.mark.parametrize('weight_bits', [8])
    @pytest.mark.parametrize('activation_bits', [8])
    @pytest.mark.parametrize('signed_weights', [None])
    @pytest.mark.parametrize('signed_activations', [None])
    # TODO(kshpv): add signed_activations and signed_weights which should be independent from HW config.
    def test_quantizer_config_from_ptq_params(self, weight_granularity, activation_granularity, preset, weight_bits,
                                              activation_bits, signed_weights, signed_activations, single_conv_nncf_graph):
        algo = PostTrainingQuantization(
            PostTrainingQuantizationParameters(preset=preset,
                                               weight_bits=weight_bits,
                                               weight_granularity=weight_granularity,
                                               signed_weights=signed_weights,
                                               activation_bits=activation_bits,
                                               activation_granularity=activation_granularity,
                                               signed_activations=signed_activations
                                               ))
        min_max_algo = algo.algorithms[0]
        min_max_algo._backend_entity = self.get_algo_backend()
        q_setup = min_max_algo._get_quantizer_setup(single_conv_nncf_graph.nncf_graph, GraphPattern())
        q_g_to_quantization_mode = {}
        for q_g in QuantizerGroup:
            q_g_to_quantization_mode[q_g] = preset.get_params_configured_by_preset(q_g)['mode']

        assert len(q_setup.quantization_points) == 2

        for quantization_point in q_setup.quantization_points.values():
            if quantization_point.is_weight_quantization_point():
                assert quantization_point.qconfig.mode == q_g_to_quantization_mode[QuantizerGroup.WEIGHTS]
                assert quantization_point.qconfig.per_channel == (weight_granularity == Granularity.PERCHANNEL)
                assert quantization_point.qconfig.num_bits == weight_bits
                if signed_weights is not None:
                    assert quantization_point.qconfig.signedness_to_force == signed_weights
            if quantization_point.is_activation_quantization_point():
                assert quantization_point.qconfig.per_channel == (activation_granularity == Granularity.PERCHANNEL)
                assert quantization_point.qconfig.num_bits == activation_bits
                assert quantization_point.qconfig.mode == q_g_to_quantization_mode[QuantizerGroup.ACTIVATIONS]
                if signed_activations is not None:
                    assert quantization_point.qconfig.signedness_to_force == signed_activations


    def test_depthwise_conv_default_quantizer_config(self, depthwise_conv_nncf_graph):
        algo = PostTrainingQuantization(PostTrainingQuantizationParameters())
        min_max_algo = algo.algorithms[0]
        min_max_algo._backend_entity = self.get_algo_backend()
        q_setup = min_max_algo._get_quantizer_setup(depthwise_conv_nncf_graph.nncf_graph, GraphPattern())

        weight_default_config = QuantizerConfig(mode=QuantizationMode.SYMMETRIC,
                                                num_bits=8,
                                                signedness_to_force=True,
                                                per_channel=True)
        activation_default_config = QuantizerConfig(mode=QuantizationMode.SYMMETRIC,
                                                    num_bits=8,
                                                    signedness_to_force=None,
                                                    per_channel=True)

        assert len(q_setup.quantization_points) == 2

        for quantization_point in q_setup.quantization_points.values():
            if quantization_point.is_weight_quantization_point():
                assert quantization_point.qconfig == weight_default_config
            if quantization_point.is_activation_quantization_point():
                assert quantization_point.qconfig == activation_default_config

    @pytest.mark.parametrize('range_type', [RangeType.MINMAX, RangeType.MEAN_MINMAX])
    @pytest.mark.parametrize('q_config_mode', [QuantizationMode.SYMMETRIC, QuantizationMode.ASYMMETRIC])
    @pytest.mark.parametrize('q_config_per_channel', [True, False])
    def test_get_stat_collector(self, range_type, q_config_mode, q_config_per_channel,
                                conv_sum_aggregation_nncf_graph,
                                statistic_collector_parameters: TestGetStatisticsCollectorParameters):
        params = statistic_collector_parameters
        algo = PostTrainingQuantization(PostTrainingQuantizationParameters(range_type=range_type))
        min_max_algo = algo.algorithms[0]
        min_max_algo._backend_entity = self.get_algo_backend()
        q_config = QuantizerConfig(num_bits=8,
                                   mode=q_config_mode,
                                   per_channel=q_config_per_channel)

        if params.target_type in [TargetType.PRE_LAYER_OPERATION, TargetType.POST_LAYER_OPERATION]:
            port_id = None if TargetType.POST_LAYER_OPERATION else 0
            ip = ActivationQuantizationInsertionPoint(params.target_node_name, port_id)
            qp = SingleConfigQuantizationPoint(ip, q_config, [params.target_node_name])
            min_max_algo._add_activation_quantization_target_point(qp)
        else:
            ip = WeightQuantizationInsertionPoint(params.target_node_name)
            qp = SingleConfigQuantizationPoint(ip, q_config, [params.target_node_name])
            min_max_algo._add_weight_quantization_target_point(qp,
                                                               conv_sum_aggregation_nncf_graph.nncf_graph)

        target_point = list(min_max_algo._quantization_target_points_to_qconfig.keys())[0]
        tensor_collector = min_max_algo._get_stat_collector(conv_sum_aggregation_nncf_graph.nncf_graph,
                                                            target_point, q_config)

        is_weight_tp = target_point.is_weight_target_point()
        # tensor_collector type check
        if is_weight_tp or q_config_per_channel:
            assert isinstance(tensor_collector, self.get_min_max_statistic_collector_cls())
        else:
            if range_type == RangeType.MINMAX:
                assert isinstance(tensor_collector, self.get_min_max_statistic_collector_cls())
            elif range_type == RangeType.MEAN_MINMAX:
                assert isinstance(tensor_collector, self.get_mean_max_statistic_collector_cls())

        # reduction_shape check
        if q_config_per_channel:
            assert tensor_collector._reduction_shape == params.ref_per_ch_reduction_shape
        else:
            assert tensor_collector._reduction_shape == params.ref_per_tensor_reduction_shape

        # use_abs_max check
        if q_config_mode == QuantizationMode.SYMMETRIC:
            assert tensor_collector._use_abs_max
        elif q_config_mode == QuantizationMode.ASYMMETRIC:
            assert not tensor_collector._use_abs_max
