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

from nncf.parameters import ModelType
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.structs import QuantizationMode
from nncf.quantization.algorithms.definitions import RangeType
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantizationParameters
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization


# pylint: disable=protected-access


class TemplateTestPTQParams:
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
    def check_quantize_outputs_fq_num(self, quantize_outputs,
                                      act_num_q, weight_num_q):
        pass

    @abstractmethod
    @pytest.fixture(scope='session')
    def test_params(self):
        pass

    @abstractmethod
    @pytest.fixture
    def ignored_scopes_data(self, request):
        pass

    @pytest.mark.parametrize('range_type', [RangeType.MINMAX, RangeType.MEAN_MINMAX, None])
    def test_range_type_per_tensor(self, test_params, range_type):
        algo = PostTrainingQuantization(PostTrainingQuantizationParameters(range_type=range_type))
        min_max_algo = algo.algorithms[0]
        min_max_algo._backend_entity = self.get_algo_backend()
        assert min_max_algo._parameters.range_type == range_type

        params = test_params['test_range_type_per_tensor']
        stat_points = min_max_algo.get_statistic_points(params['model'])
        assert len(stat_points) == params['stat_points_num']

        for _, stat_point in stat_points.items():
            for stat_point_ in stat_point:
                for tensor_collector in stat_point_.algorithm_to_tensor_collectors[MinMaxQuantization]:
                    if stat_point_.target_point.is_weight_target_point():
                        # default tensor_collector for weights
                        assert isinstance(tensor_collector, self.get_min_max_statistic_collector_cls())
                        continue
                    if range_type is None:
                        # default tensor_collector for per-tensor
                        assert isinstance(tensor_collector, self.get_mean_max_statistic_collector_cls())
                    if range_type == RangeType.MINMAX:
                        assert isinstance(tensor_collector, self.get_min_max_statistic_collector_cls())
                    elif range_type == RangeType.MEAN_MINMAX:
                        assert isinstance(tensor_collector, self.get_mean_max_statistic_collector_cls())

    @pytest.mark.parametrize('range_type', [RangeType.MINMAX, RangeType.MEAN_MINMAX, None])
    def test_range_type_per_channel(self, test_params, range_type):
        algo = PostTrainingQuantization(PostTrainingQuantizationParameters(range_type=range_type))
        min_max_algo = algo.algorithms[0]
        min_max_algo._backend_entity = self.get_algo_backend()
        assert min_max_algo._parameters.range_type == range_type

        params = test_params['test_range_type_per_channel']
        stat_points = min_max_algo.get_statistic_points(params['model'])
        assert len(stat_points) == params['stat_points_num']

        for _, stat_point in stat_points.items():
            for stat_point_ in stat_point:
                for tensor_collector in stat_point_.algorithm_to_tensor_collectors[MinMaxQuantization]:
                    # Range_type does not affect per-channel tensor_collector
                    assert isinstance(tensor_collector, self.get_min_max_statistic_collector_cls())

    @pytest.mark.parametrize('quantize_outputs', [False, True])
    def test_quantize_outputs(self, test_params, quantize_outputs):
        algo = PostTrainingQuantization(PostTrainingQuantizationParameters(quantize_outputs=quantize_outputs))
        min_max_algo = algo.algorithms[0]
        min_max_algo._backend_entity = self.get_algo_backend()

        nncf_graph = test_params['test_quantize_outputs']['nncf_graph']
        pattern = test_params['test_quantize_outputs']['pattern']

        assert min_max_algo._parameters.quantize_outputs == quantize_outputs
        q_setup = min_max_algo._get_quantizer_setup(nncf_graph, pattern)
        act_num_q, weight_num_q = 0, 0
        for quantization_point in q_setup.quantization_points.values():
            if quantization_point.is_activation_quantization_point():
                act_num_q += 1
            if quantization_point.is_weight_quantization_point():
                weight_num_q += 1

        self.check_quantize_outputs_fq_num(quantize_outputs,
                                           act_num_q, weight_num_q)

    def test_ignored_scopes(self, test_params, ignored_scopes_data):
        ignored_scopes, act_num_ref, weight_num_ref = ignored_scopes_data
        algo = PostTrainingQuantization(PostTrainingQuantizationParameters(ignored_scopes=ignored_scopes))
        min_max_algo = algo.algorithms[0]
        min_max_algo._backend_entity = self.get_algo_backend()
        assert min_max_algo._parameters.ignored_scopes == ignored_scopes

        nncf_graph = test_params['test_ignored_scopes']['nncf_graph']
        pattern = test_params['test_ignored_scopes']['pattern']

        q_setup = min_max_algo._get_quantizer_setup(nncf_graph, pattern)
        act_num_q, weight_num_q = 0, 0
        for quantization_point in q_setup.quantization_points.values():
            if quantization_point.is_activation_quantization_point():
                act_num_q += 1
            if quantization_point.is_weight_quantization_point():
                weight_num_q += 1

        assert act_num_q == act_num_ref
        assert weight_num_q == weight_num_ref

    @pytest.mark.parametrize('model_type', [ModelType.TRANSFORMER])
    def test_model_type_pass(self, test_params, model_type):
        algo = PostTrainingQuantization(PostTrainingQuantizationParameters(preset=QuantizationPreset.MIXED,
                                                                           model_type=model_type))
        min_max_algo = algo.algorithms[0]
        min_max_algo._backend_entity = self.get_algo_backend()

        nncf_graph = test_params['test_model_type_pass']['nncf_graph']
        pattern = test_params['test_model_type_pass']['pattern']
        q_setup = min_max_algo._get_quantizer_setup(nncf_graph, pattern)
        for quantization_point in q_setup.quantization_points.values():
            if quantization_point.is_activation_quantization_point():
                node_names = quantization_point.directly_quantized_operator_node_names
                for node_name in node_names:
                    if nncf_graph.get_node_by_name(node_name).metatype == min_max_algo._backend_entity.mat_mul_metatype:
                        assert quantization_point.qconfig.mode == QuantizationMode.ASYMMETRIC
        min_max_algo._apply_model_type_pass(model_type, q_setup, nncf_graph)
        for quantization_point in q_setup.quantization_points.values():
            if quantization_point.is_activation_quantization_point():
                node_names = quantization_point.directly_quantized_operator_node_names
                for node_name in node_names:
                    if nncf_graph.get_node_by_name(node_name).metatype == min_max_algo._backend_entity.mat_mul_metatype:
                        assert quantization_point.qconfig.mode == QuantizationMode.SYMMETRIC

