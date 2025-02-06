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
from copy import deepcopy
from dataclasses import dataclass
from typing import List

import pytest

from nncf import ModelType
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns.manager import PatternsManager
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.quantization.quantizer_setup import ActivationQuantizationInsertionPoint
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizationPoint
from nncf.common.quantization.quantizer_setup import WeightQuantizationInsertionPoint
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.tensor_statistics.collectors import ReductionAxes
from nncf.experimental.common.tensor_statistics.collectors import AbsMaxReducer
from nncf.experimental.common.tensor_statistics.collectors import MaxAggregator
from nncf.experimental.common.tensor_statistics.collectors import MaxReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanAggregator
from nncf.experimental.common.tensor_statistics.collectors import MinAggregator
from nncf.experimental.common.tensor_statistics.collectors import MinReducer
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.collectors import TensorReducerBase
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import QuantizationParameters
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.passes import transform_to_inference_graph
from nncf.quantization.range_estimator import RangeEstimatorParametersSet
from tests.cross_fw.test_templates.models import NNCFGraphToTest
from tests.cross_fw.test_templates.models import NNCFGraphToTestDepthwiseConv
from tests.cross_fw.test_templates.models import NNCFGraphToTestSumAggregation


class TemplateTestQuantizerConfig:
    @abstractmethod
    def get_algo_backend(self):
        pass

    @abstractmethod
    def get_backend_type(self):
        pass

    def check_is_min_max_statistic_collector(self, tensor_collector: TensorCollector):
        aggrs = [aggr.__class__ for aggr in tensor_collector.aggregators.values()]
        assert len(aggrs) == 2
        assert MinAggregator in aggrs
        assert MaxAggregator in aggrs

    def check_is_mean_min_max_statistic_collector(self, tensor_collector: TensorCollector):
        aggrs = [aggr.__class__ for aggr in tensor_collector.aggregators.values()]
        assert len(aggrs) == 2
        assert MeanAggregator in aggrs
        assert aggrs[0].__class__ == aggrs[1].__class__

    def get_reduction_axes(self, reducer: TensorReducerBase) -> ReductionAxes:
        return reducer._reduction_axes

    @staticmethod
    def _transform_to_inference_graph(nncf_graph: NNCFGraph, min_max_algo: MinMaxQuantization):
        return transform_to_inference_graph(
            deepcopy(nncf_graph),
            min_max_algo._backend_entity.get_start_nodes_for_activation_path_tracing(nncf_graph),
            min_max_algo._backend_entity.shapeof_metatypes,
            min_max_algo._backend_entity.dropout_metatypes,
            min_max_algo._backend_entity.preserved_metatypes,
        )

    @abstractmethod
    @pytest.fixture
    def single_conv_nncf_graph(self) -> NNCFGraphToTest:
        pass

    @abstractmethod
    @pytest.fixture
    def transformer_nncf_graph(self) -> NNCFGraphToTest:
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
    class GetStatisticsCollectorParameters:
        target_type: TargetType
        target_node_name: str
        batchwise_statistics: bool
        ref_per_ch_reduction_axes: List[int]
        ref_per_tensor_reduction_axes: List[int]

    @pytest.fixture(
        params=[
            pytest.param(
                GetStatisticsCollectorParameters(TargetType.PRE_LAYER_OPERATION, "/Sum_1_0", True, (2,), (1, 2)),
            ),
            GetStatisticsCollectorParameters(
                TargetType.POST_LAYER_OPERATION,
                "/Conv_1_0",
                True,
                (2, 3),
                (1, 2, 3),
            ),
            GetStatisticsCollectorParameters(
                TargetType.OPERATION_WITH_WEIGHTS,
                "/Conv_1_0",
                True,
                (1, 2, 3),
                (0, 1, 2, 3),
            ),
            GetStatisticsCollectorParameters(TargetType.PRE_LAYER_OPERATION, "/Sum_1_0", False, (0, 2), (0, 1, 2)),
            GetStatisticsCollectorParameters(
                TargetType.POST_LAYER_OPERATION,
                "/Conv_1_0",
                False,
                (0, 2, 3),
                (0, 1, 2, 3),
            ),
            GetStatisticsCollectorParameters(
                TargetType.OPERATION_WITH_WEIGHTS,
                "/Conv_1_0",
                False,
                (1, 2, 3),
                (0, 1, 2, 3),
            ),
        ]
    )
    def statistic_collector_parameters(self, request) -> GetStatisticsCollectorParameters:
        return request.param

    def test_default_quantizer_config(self, single_conv_nncf_graph):
        min_max_algo = MinMaxQuantization()
        min_max_algo._backend_entity = self.get_algo_backend()
        nncf_graph = single_conv_nncf_graph.nncf_graph
        inference_nncf_graph = self._transform_to_inference_graph(nncf_graph, min_max_algo)
        q_setup = min_max_algo._get_quantizer_setup(
            nncf_graph, inference_nncf_graph, hw_patterns=GraphPattern(), ignored_patterns=GraphPattern()
        )

        weight_default_config = QuantizerConfig(
            mode=QuantizationMode.SYMMETRIC, num_bits=8, signedness_to_force=True, per_channel=True
        )
        activation_default_config = QuantizerConfig(
            mode=QuantizationMode.SYMMETRIC, num_bits=8, signedness_to_force=None, per_channel=False
        )

        assert len(q_setup.quantization_points) == 2

        for quantization_point in q_setup.quantization_points.values():
            if quantization_point.is_weight_quantization_point():
                assert quantization_point.qconfig == weight_default_config
            if quantization_point.is_activation_quantization_point():
                assert quantization_point.qconfig == activation_default_config

    @pytest.mark.parametrize("weight_per_channel", [True, False])
    @pytest.mark.parametrize("activation_per_channel", [False])
    @pytest.mark.parametrize("preset", [QuantizationPreset.MIXED, QuantizationPreset.PERFORMANCE])
    @pytest.mark.parametrize("weight_bits", [8])
    @pytest.mark.parametrize("activation_bits", [8])
    @pytest.mark.parametrize("signed_weights", [None, True, False])
    @pytest.mark.parametrize("signed_activations", [None, True, False])
    def test_quantizer_config_from_ptq_params_for_CPU(
        self,
        weight_per_channel,
        activation_per_channel,
        preset,
        weight_bits,
        activation_bits,
        signed_weights,
        signed_activations,
        single_conv_nncf_graph,
    ):
        min_max_algo = MinMaxQuantization(
            preset=preset,
            activations_quantization_params=QuantizationParameters(
                num_bits=activation_bits, per_channel=activation_per_channel, signedness_to_force=signed_activations
            ),
            weights_quantization_params=QuantizationParameters(
                num_bits=weight_bits, per_channel=weight_per_channel, signedness_to_force=signed_weights
            ),
        )
        min_max_algo._backend_entity = self.get_algo_backend()
        nncf_graph = single_conv_nncf_graph.nncf_graph
        inference_nncf_graph = self._transform_to_inference_graph(nncf_graph, min_max_algo)
        if signed_weights is False or signed_activations in [True, False]:  # Incompatible with HW CPU config
            with pytest.raises(
                ValueError,
                match=".*?Quantization parameter constraints specified in NNCF config are incompatible.*?",
            ):
                q_setup = min_max_algo._get_quantizer_setup(
                    nncf_graph, inference_nncf_graph, hw_patterns=GraphPattern(), ignored_patterns=GraphPattern()
                )
        else:
            q_setup = min_max_algo._get_quantizer_setup(
                nncf_graph, inference_nncf_graph, hw_patterns=GraphPattern(), ignored_patterns=GraphPattern()
            )
            q_g_to_quantization_mode = {}
            for q_g in QuantizerGroup:
                q_g_to_quantization_mode[q_g] = preset.get_params_configured_by_preset(q_g)["mode"]

            assert len(q_setup.quantization_points) == 2

            for quantization_point in q_setup.quantization_points.values():
                if quantization_point.is_weight_quantization_point():
                    assert quantization_point.qconfig.mode == q_g_to_quantization_mode[QuantizerGroup.WEIGHTS]
                    assert quantization_point.qconfig.per_channel == weight_per_channel
                    assert quantization_point.qconfig.num_bits == weight_bits
                    if signed_weights is not None:
                        assert quantization_point.qconfig.signedness_to_force == signed_weights
                if quantization_point.is_activation_quantization_point():
                    assert quantization_point.qconfig.per_channel == activation_per_channel
                    assert quantization_point.qconfig.num_bits == activation_bits
                    assert quantization_point.qconfig.mode == q_g_to_quantization_mode[QuantizerGroup.ACTIVATIONS]
                    if signed_activations is not None:
                        assert quantization_point.qconfig.signedness_to_force == signed_activations

    def test_depthwise_conv_default_quantizer_config(self, depthwise_conv_nncf_graph):
        min_max_algo = MinMaxQuantization()
        min_max_algo._backend_entity = self.get_algo_backend()
        nncf_graph = depthwise_conv_nncf_graph.nncf_graph
        inference_nncf_graph = self._transform_to_inference_graph(nncf_graph, min_max_algo)
        q_setup = min_max_algo._get_quantizer_setup(
            nncf_graph, inference_nncf_graph, hw_patterns=GraphPattern(), ignored_patterns=GraphPattern()
        )

        weight_default_config = QuantizerConfig(
            mode=QuantizationMode.SYMMETRIC, num_bits=8, signedness_to_force=True, per_channel=True
        )
        activation_default_config = QuantizerConfig(
            mode=QuantizationMode.SYMMETRIC, num_bits=8, signedness_to_force=None, per_channel=True
        )

        assert len(q_setup.quantization_points) == 2

        for quantization_point in q_setup.quantization_points.values():
            if quantization_point.is_weight_quantization_point():
                assert quantization_point.qconfig == weight_default_config
            if quantization_point.is_activation_quantization_point():
                assert quantization_point.qconfig == activation_default_config

    REF_TRANSFORMER_SETUP_STATE = {
        "quantization_points": {
            4: {
                "qip": {"target_node_name": "/K_0", "input_port_id": None},
                "qip_class": "ActivationQuantizationInsertionPoint",
                "qconfig": {"num_bits": 8, "mode": "symmetric", "signedness_to_force": None, "per_channel": False},
                "directly_quantized_operator_node_names": ["/K_Q_0"],
            },
            5: {
                "qip": {"target_node_name": "/Q_0", "input_port_id": None},
                "qip_class": "ActivationQuantizationInsertionPoint",
                "qconfig": {"num_bits": 8, "mode": "symmetric", "signedness_to_force": None, "per_channel": False},
                "directly_quantized_operator_node_names": ["/K_Q_0"],
            },
            6: {
                "qip": {"target_node_name": "/Input_1_0", "input_port_id": None},
                "qip_class": "ActivationQuantizationInsertionPoint",
                "qconfig": {"num_bits": 8, "mode": "asymmetric", "signedness_to_force": None, "per_channel": False},
                "directly_quantized_operator_node_names": ["/K_0", "/Q_0", "/V_0"],
            },
            8: {
                "qip": {"target_node_name": "/K_0"},
                "qip_class": "WeightQuantizationInsertionPoint",
                "qconfig": {"num_bits": 8, "mode": "symmetric", "signedness_to_force": True, "per_channel": True},
                "directly_quantized_operator_node_names": ["/K_0"],
            },
            9: {
                "qip": {"target_node_name": "/Q_0"},
                "qip_class": "WeightQuantizationInsertionPoint",
                "qconfig": {"num_bits": 8, "mode": "symmetric", "signedness_to_force": True, "per_channel": True},
                "directly_quantized_operator_node_names": ["/Q_0"],
            },
            10: {
                "qip": {"target_node_name": "/V_0"},
                "qip_class": "WeightQuantizationInsertionPoint",
                "qconfig": {"num_bits": 8, "mode": "symmetric", "signedness_to_force": True, "per_channel": True},
                "directly_quantized_operator_node_names": ["/V_0"],
            },
        },
        "unified_scale_groups": {},
        "shared_input_operation_set_groups": {0: [4, 5], 1: [8, 9, 10, 6]},
    }

    def test_model_type_transformer_quantization_config(self, transformer_nncf_graph):
        min_max_algo = MinMaxQuantization(model_type=ModelType.TRANSFORMER)
        min_max_algo._backend_entity = self.get_algo_backend()
        nncf_graph = transformer_nncf_graph.nncf_graph
        inference_nncf_graph = self._transform_to_inference_graph(nncf_graph, min_max_algo)
        hw_patterns = PatternsManager.get_full_hw_pattern_graph(
            backend=self.get_backend_type(), device=TargetDevice.ANY, model_type=ModelType.TRANSFORMER
        )
        ignored_patterns = PatternsManager.get_full_ignored_pattern_graph(
            backend=self.get_backend_type(), device=TargetDevice.ANY, model_type=ModelType.TRANSFORMER
        )
        q_setup = min_max_algo._get_quantizer_setup(
            nncf_graph, inference_nncf_graph, hw_patterns=hw_patterns, ignored_patterns=ignored_patterns
        )
        min_max_algo._apply_model_type_pass(ModelType.TRANSFORMER, q_setup, nncf_graph)

        state = q_setup.get_state()
        state["quantization_points"][6]["directly_quantized_operator_node_names"] = sorted(
            state["quantization_points"][6]["directly_quantized_operator_node_names"]
        )
        assert state == self.REF_TRANSFORMER_SETUP_STATE

    @pytest.mark.parametrize(
        "range_estimator_params", [RangeEstimatorParametersSet.MINMAX, RangeEstimatorParametersSet.MEAN_MINMAX]
    )
    @pytest.mark.parametrize("q_config_mode", [QuantizationMode.SYMMETRIC, QuantizationMode.ASYMMETRIC])
    @pytest.mark.parametrize("q_config_per_channel", [True, False])
    @pytest.mark.parametrize("num_samples", [5, 12])
    def test_get_stat_collector(
        self,
        range_estimator_params,
        q_config_mode,
        q_config_per_channel,
        num_samples,
        conv_sum_aggregation_nncf_graph,
        statistic_collector_parameters: GetStatisticsCollectorParameters,
    ):
        params = statistic_collector_parameters
        min_max_algo = MinMaxQuantization(
            subset_size=num_samples, activations_range_estimator_params=range_estimator_params
        )
        min_max_algo._backend_entity = self.get_algo_backend()
        min_max_algo._init_cache()
        q_config = QuantizerConfig(num_bits=8, mode=q_config_mode, per_channel=q_config_per_channel)

        if params.target_type in [TargetType.PRE_LAYER_OPERATION, TargetType.POST_LAYER_OPERATION]:
            port_id = None if TargetType.POST_LAYER_OPERATION else 0
            ip = ActivationQuantizationInsertionPoint(params.target_node_name, port_id)
            qp = SingleConfigQuantizationPoint(ip, q_config, [params.target_node_name])
            min_max_algo._add_activation_quantization_target_point(qp, conv_sum_aggregation_nncf_graph.nncf_graph)
        else:
            ip = WeightQuantizationInsertionPoint(params.target_node_name)
            qp = SingleConfigQuantizationPoint(ip, q_config, [params.target_node_name])
            min_max_algo._add_weight_quantization_target_point(qp, conv_sum_aggregation_nncf_graph.nncf_graph)

        target_point = list(min_max_algo._quantization_target_points_to_qconfig.keys())[0]
        tensor_collector = min_max_algo._get_stat_collector(
            conv_sum_aggregation_nncf_graph.nncf_graph, target_point, q_config, params.batchwise_statistics
        )

        is_weight_tp = target_point.is_weight_target_point()
        # tensor_collector type check
        if is_weight_tp:
            self.check_is_min_max_statistic_collector(tensor_collector)
        else:
            if range_estimator_params == RangeEstimatorParametersSet.MINMAX:
                self.check_is_min_max_statistic_collector(tensor_collector)
            elif range_estimator_params == RangeEstimatorParametersSet.MEAN_MINMAX:
                self.check_is_mean_min_max_statistic_collector(tensor_collector)

        reducers = tensor_collector.reducers
        assert len(reducers) == 2
        # use_abs_max check
        assert any(isinstance(r, MinReducer) for r in reducers)
        if q_config_mode == QuantizationMode.SYMMETRIC:
            assert any(isinstance(r, AbsMaxReducer) for r in reducers)
        elif q_config_mode == QuantizationMode.ASYMMETRIC:
            assert any(isinstance(r, MaxReducer) for r in reducers)

        for reducer in reducers:
            if q_config_per_channel:
                assert self.get_reduction_axes(reducer) == params.ref_per_ch_reduction_axes
            else:
                assert self.get_reduction_axes(reducer) == params.ref_per_tensor_reduction_axes
        if is_weight_tp:
            assert tensor_collector.num_samples == 1
        else:
            assert tensor_collector.num_samples == num_samples
