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
from copy import deepcopy
from typing import Dict

import pytest

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.experimental.common.tensor_statistics.collectors import MaxAggregator
from nncf.experimental.common.tensor_statistics.collectors import MeanAggregator
from nncf.experimental.common.tensor_statistics.collectors import MinAggregator
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.parameters import ModelType
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.passes import transform_to_inference_graph
from nncf.quantization.range_estimator import RangeEstimatorParametersSet
from nncf.scopes import IgnoredScope
from nncf.tensor import Tensor
from tests.common.quantization.metatypes import CatTestMetatype
from tests.common.quantization.metatypes import Conv2dTestMetatype
from tests.common.quantization.metatypes import IdentityTestMetatype
from tests.common.quantization.metatypes import LinearTestMetatype
from tests.common.quantization.metatypes import SoftmaxTestMetatype
from tests.common.quantization.metatypes import TestMetatype
from tests.common.quantization.mock_graphs import NodeWithType
from tests.common.quantization.mock_graphs import create_mock_graph
from tests.common.quantization.mock_graphs import get_nncf_graph_from_mock_nx_graph


class ModelToTestOverflowFix:
    #   Input_1       Input_2
    #      |             |
    #   Conv_1         FC_1
    #        \         / |
    #          \      /  |
    #            FC_2     \
    #             /        \
    #      Identity_1    Output_2
    #          |
    #       SoftMax
    #          |
    #       Output_1

    def __init__(self, metatypes: Dict[TestMetatype, OperatorMetatype]):
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Input_2", InputNoopMetatype),
            NodeWithType("Conv_1", metatypes[Conv2dTestMetatype]),
            NodeWithType("FC_1", metatypes[LinearTestMetatype]),
            NodeWithType("FC_2", metatypes[LinearTestMetatype]),
            NodeWithType("Identity_1", IdentityTestMetatype),
            NodeWithType("Output_2", OutputNoopMetatype),
            NodeWithType("SoftMax", metatypes[SoftmaxTestMetatype]),
            NodeWithType("Output_1", OutputNoopMetatype),
        ]
        node_edges = [
            ("Input_1", "Conv_1"),
            ("Conv_1", "FC_2"),
            ("FC_2", "Identity_1"),
            ("Identity_1", "SoftMax"),
            ("SoftMax", "Output_1"),
            ("Input_2", "FC_1"),
            ("FC_1", "FC_2"),
            ("FC_1", "Output_2"),
        ]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph)
        self.weight_quantization_target_point_names = []
        weigth_meatypes = [metatypes[Conv2dTestMetatype], metatypes[LinearTestMetatype]]
        for node in self.nncf_graph.get_nodes_by_metatypes(weigth_meatypes):
            self.weight_quantization_target_point_names.append(node.node_name)


class ModelWithUnifiedScales:
    #        Input_1
    #       /   |   \
    #  Conv_1 Conv_2 Conv_3
    #       \   |   /
    #         Cat_1
    #           |
    #        Output_1

    def __init__(self, metatypes: Dict[TestMetatype, OperatorMetatype], nncf_graph_cls=NNCFGraph):
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype),
            NodeWithType("Conv_1", metatypes[Conv2dTestMetatype]),
            NodeWithType("Conv_2", metatypes[Conv2dTestMetatype]),
            NodeWithType("Conv_3", metatypes[Conv2dTestMetatype]),
            NodeWithType("Cat_1", metatypes[CatTestMetatype]),
            NodeWithType("Output_1", OutputNoopMetatype),
        ]
        node_edges = [
            ("Input_1", "Conv_1"),
            ("Input_1", "Conv_2"),
            ("Input_1", "Conv_3"),
            ("Conv_1", "Cat_1"),
            ("Conv_2", "Cat_1"),
            ("Conv_3", "Cat_1"),
            ("Cat_1", "Output_1"),
        ]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph, nncf_graph_cls=nncf_graph_cls)


class DummyMinMaxTensorStatistic(MinMaxTensorStatistic):
    def tensor_eq(self):
        return True


class DummyMinMaxTensorCollector:
    def __init__(self, min_val, max_val):
        self._stat = DummyMinMaxTensorStatistic(min_values=min_val, max_values=max_val)

    def get_statistics(self):
        return self._stat


class TemplateTestPTQParams:
    @abstractmethod
    def get_algo_backend(self):
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

    @abstractmethod
    def check_quantize_outputs_fq_num(self, quantize_outputs, act_num_q, weight_num_q):
        pass

    @abstractmethod
    def check_unified_scale_layout(self, layout, unified_scales_group):
        """
        Checks that given transfromation layout and unified_scales_group target points
        are correspond to each other and to the test params
        """

    @abstractmethod
    @pytest.fixture(scope="session")
    def test_params(self):
        pass

    @abstractmethod
    @pytest.fixture
    def ignored_scopes_data(self, request):
        pass

    @abstractmethod
    def target_point(self, target_type: TargetType, target_node_name: str, port_id: int):
        pass

    @property
    @abstractmethod
    def metatypes_mapping(self):
        pass

    @property
    @abstractmethod
    def nncf_graph_cls(self):
        pass

    @abstractmethod
    def get_backend_tensor(self, value):
        pass

    @pytest.mark.parametrize(
        "range_estimator_params", [RangeEstimatorParametersSet.MINMAX, RangeEstimatorParametersSet.MEAN_MINMAX, None]
    )
    def test_range_estimator_per_tensor(self, test_params, range_estimator_params):
        min_max_algo = MinMaxQuantization(activations_range_estimator_params=range_estimator_params)
        min_max_algo._backend_entity = self.get_algo_backend()
        assert min_max_algo._range_estimator_params[QuantizerGroup.ACTIVATIONS] == range_estimator_params

        params = test_params["test_range_estimator_per_tensor"]
        stat_points = min_max_algo.get_statistic_points(params["model"], params["nncf_graph"])
        assert len(stat_points) == params["stat_points_num"]

        for _, stat_point in stat_points.items():
            for stat_point_ in stat_point:
                for tensor_collector in stat_point_.algorithm_to_tensor_collectors[min_max_algo._algorithm_key]:
                    if stat_point_.target_point.is_weight_target_point():
                        # default tensor_collector for weights
                        self.check_is_min_max_statistic_collector(tensor_collector)
                        continue
                    if range_estimator_params is None:
                        # default tensor_collector for per-tensor
                        self.check_is_mean_min_max_statistic_collector(tensor_collector)
                    if range_estimator_params == RangeEstimatorParametersSet.MINMAX:
                        self.check_is_min_max_statistic_collector(tensor_collector)
                    elif range_estimator_params == RangeEstimatorParametersSet.MEAN_MINMAX:
                        self.check_is_mean_min_max_statistic_collector(tensor_collector)

    @pytest.mark.parametrize("quantize_outputs", [False, True])
    def test_quantize_outputs(self, test_params, quantize_outputs):
        min_max_algo = MinMaxQuantization(quantize_outputs=quantize_outputs)
        min_max_algo._backend_entity = self.get_algo_backend()

        nncf_graph = test_params["test_quantize_outputs"]["nncf_graph"]

        assert min_max_algo._quantize_outputs == quantize_outputs
        hw_patterns = test_params["test_model_type_pass"]["hw_patterns"]
        ignored_patterns = test_params["test_model_type_pass"]["ignored_patterns"]
        inference_nncf_graph = transform_to_inference_graph(
            deepcopy(nncf_graph),
            min_max_algo._backend_entity.get_start_nodes_for_activation_path_tracing(nncf_graph),
            min_max_algo._backend_entity.shapeof_metatypes,
            min_max_algo._backend_entity.dropout_metatypes,
            min_max_algo._backend_entity.preserved_metatypes,
        )
        q_setup = min_max_algo._get_quantizer_setup(nncf_graph, inference_nncf_graph, hw_patterns, ignored_patterns)
        act_num_q, weight_num_q = 0, 0
        for quantization_point in q_setup.quantization_points.values():
            if quantization_point.is_activation_quantization_point():
                act_num_q += 1
            if quantization_point.is_weight_quantization_point():
                weight_num_q += 1

        self.check_quantize_outputs_fq_num(quantize_outputs, act_num_q, weight_num_q)

    def test_ignored_scopes(self, test_params, ignored_scopes_data):
        ignored_scope, act_num_ref, weight_num_ref = ignored_scopes_data
        min_max_algo = MinMaxQuantization(ignored_scope=ignored_scope)
        min_max_algo._backend_entity = self.get_algo_backend()
        assert min_max_algo._ignored_scope == ignored_scope

        nncf_graph = test_params["test_ignored_scopes"]["nncf_graph"]
        hw_patterns = test_params["test_model_type_pass"]["hw_patterns"]
        ignored_patterns = test_params["test_model_type_pass"]["ignored_patterns"]
        inference_nncf_graph = transform_to_inference_graph(
            deepcopy(nncf_graph),
            min_max_algo._backend_entity.get_start_nodes_for_activation_path_tracing(nncf_graph),
            min_max_algo._backend_entity.shapeof_metatypes,
            min_max_algo._backend_entity.dropout_metatypes,
            min_max_algo._backend_entity.preserved_metatypes,
        )
        q_setup = min_max_algo._get_quantizer_setup(nncf_graph, inference_nncf_graph, hw_patterns, ignored_patterns)
        act_num_q, weight_num_q = 0, 0
        for quantization_point in q_setup.quantization_points.values():
            if quantization_point.is_activation_quantization_point():
                act_num_q += 1
            if quantization_point.is_weight_quantization_point():
                weight_num_q += 1

        assert act_num_q == act_num_ref
        assert weight_num_q == weight_num_ref

    @pytest.mark.parametrize("model_type", [ModelType.TRANSFORMER])
    def test_model_type_pass(self, test_params, model_type):
        min_max_algo = MinMaxQuantization(preset=QuantizationPreset.MIXED, model_type=model_type)
        min_max_algo._backend_entity = self.get_algo_backend()

        nncf_graph = test_params["test_model_type_pass"]["nncf_graph"]
        hw_patterns = test_params["test_model_type_pass"]["hw_patterns"]
        ignored_patterns = test_params["test_model_type_pass"]["ignored_patterns"]
        inference_nncf_graph = transform_to_inference_graph(
            deepcopy(nncf_graph),
            min_max_algo._backend_entity.get_start_nodes_for_activation_path_tracing(nncf_graph),
            min_max_algo._backend_entity.shapeof_metatypes,
            min_max_algo._backend_entity.dropout_metatypes,
            min_max_algo._backend_entity.preserved_metatypes,
        )
        q_setup = min_max_algo._get_quantizer_setup(nncf_graph, inference_nncf_graph, hw_patterns, ignored_patterns)
        for quantization_point in q_setup.quantization_points.values():
            if quantization_point.is_activation_quantization_point():
                node_names = quantization_point.directly_quantized_operator_node_names
                for node_name in node_names:
                    if (
                        nncf_graph.get_node_by_name(node_name).metatype
                        == min_max_algo._backend_entity.mat_mul_metatypes
                    ):
                        assert quantization_point.qconfig.mode == QuantizationMode.ASYMMETRIC
        min_max_algo._apply_model_type_pass(model_type, q_setup, nncf_graph)
        for quantization_point in q_setup.quantization_points.values():
            if quantization_point.is_activation_quantization_point():
                node_names = quantization_point.directly_quantized_operator_node_names
                for node_name in node_names:
                    if (
                        nncf_graph.get_node_by_name(node_name).metatype
                        == min_max_algo._backend_entity.mat_mul_metatypes
                    ):
                        assert quantization_point.qconfig.mode == QuantizationMode.SYMMETRIC

    @pytest.mark.parametrize(
        "overflow_fix, affected_target_points, ignored_ops",
        [
            [OverflowFix.DISABLE, [], []],
            [OverflowFix.FIRST_LAYER, ["/Conv_1_0"], []],
            [OverflowFix.FIRST_LAYER, ["/Conv_1_0"], ["/FC_1_0"]],
            [OverflowFix.FIRST_LAYER, [], ["/FC_1_0", "/Conv_1_0"]],
            [OverflowFix.FIRST_LAYER, [], ["/FC_1_0", "/Conv_1_0", "/FC_2_0"]],
            [OverflowFix.ENABLE, ["/Conv_1_0", "/FC_1_0", "/FC_2_0"], []],
            [OverflowFix.ENABLE, ["/FC_1_0"], ["/Conv_1_0", "/FC_2_0"]],
        ],
    )
    def test_quantization_points_overflow_fix(self, overflow_fix, affected_target_points, ignored_ops):
        # Checks the return value of _get_quantization_points_overflow_fix based on
        # the overflow_fix and weight target points.
        model = ModelToTestOverflowFix(self.metatypes_mapping)
        nncf_graph = model.nncf_graph

        # Imitate _get_quantization_target_points
        weight_target_points = {}
        for target_point_name in model.weight_quantization_target_point_names:
            target_point = self.target_point(TargetType.OPERATION_WITH_WEIGHTS, target_point_name, 0)
            weight_target_points[target_point] = QuantizerConfig()

        # Remove ignored nodes from weight_target_points
        filtered_weight_target_points = {}
        for t_p, config in weight_target_points.items():
            if t_p.target_node_name not in ignored_ops:
                filtered_weight_target_points[t_p] = config

        algo = MinMaxQuantization()
        algo._backend_entity = self.get_algo_backend()
        target_points_overflow_fix = algo._get_quantization_points_overflow_fix(
            overflow_fix, filtered_weight_target_points, nncf_graph
        )
        assert Counter([t_p.target_node_name for t_p in target_points_overflow_fix]) == Counter(affected_target_points)

    def test_unified_scales_command_creation(self, mocker):
        model = ModelWithUnifiedScales(self.metatypes_mapping, self.nncf_graph_cls)
        algo = MinMaxQuantization()
        algo._backend_entity = self.get_algo_backend()
        # Imitating solver quantization setup building
        q_tp_vs_qcf = {}
        unified_scales_group = []
        for i in range(1, 4):
            tp = self.target_point(TargetType.POST_LAYER_OPERATION, f"/Conv_{i}_0", port_id=0)
            q_tp_vs_qcf[tp] = QuantizerConfig()
            unified_scales_group.append(tp)

        algo._quantization_target_points_to_qconfig = q_tp_vs_qcf
        algo._unified_scale_groups = [unified_scales_group]

        mock_transformer = mocker.MagicMock()
        mocker.patch(
            "nncf.quantization.algorithms.min_max.algorithm.ModelTransformerFactory.create",
            return_value=mock_transformer,
        )
        stats = StatisticPointsContainer()
        for idx, tp in enumerate(unified_scales_group):
            tc = DummyMinMaxTensorCollector(
                Tensor(self.get_backend_tensor(idx - 1)), Tensor(self.get_backend_tensor(idx + 2))
            )
            stats.add_statistic_point(StatisticPoint(tp, tc, algo._algorithm_key))
        algo.apply(model, model.nncf_graph, stats)
        mock_transformer.transform.assert_called_once()
        layout = mock_transformer.transform.call_args.args[0]
        self.check_unified_scale_layout(layout, unified_scales_group)

    @pytest.mark.parametrize("validate_scopes", (True, False))
    def test_validate_scope(self, test_params, validate_scopes):
        nncf_graph = test_params["test_model_type_pass"]["nncf_graph"]
        inference_nncf_graph = transform_to_inference_graph(
            deepcopy(nncf_graph),
            self.get_algo_backend().get_start_nodes_for_activation_path_tracing(nncf_graph),
            [],
            [],
            [],
        )
        ignored_patterns = test_params["test_model_type_pass"]["ignored_patterns"]
        algo = MinMaxQuantization(
            ignored_scope=IgnoredScope(names=["some_node"], validate=validate_scopes),
        )
        algo._backend_entity = self.get_algo_backend()
        if validate_scopes:
            with pytest.raises(nncf.ValidationError, match="Ignored nodes that matches names"):
                algo._get_ignored_names(nncf_graph, inference_nncf_graph, ignored_patterns)
        else:
            algo._get_ignored_names(nncf_graph, inference_nncf_graph, ignored_patterns)

    @pytest.mark.parametrize("mode", ["target_point", "unified_scales"])
    def test_empty_statistics(self, mode, mocker):
        algo = MinMaxQuantization()
        target_point = self.target_point(TargetType.PRE_LAYER_OPERATION, "A", 0)
        stat_points = StatisticPointsContainer()

        dummy_tp = {target_point: QuantizerConfig()}
        if mode == "target_point":
            dummy_tps = (dummy_tp, {})
        else:
            dummy_tps = ({}, ((target_point,),))
        stat_points.add_statistic_point(
            StatisticPoint(target_point, DummyMinMaxTensorCollector(None, None), algo._algorithm_key)
        )
        mocker.patch("nncf.common.factory.ModelTransformerFactory.create", return_value=mocker.MagicMock())
        mocker.patch(
            "nncf.quantization.algorithms.min_max.algorithm.MinMaxQuantization._get_quantization_target_points",
            return_value=dummy_tps,
        )
        mocker.patch(
            "nncf.quantization.algorithms.min_max.algorithm.MinMaxQuantization._get_quantization_points_overflow_fix",
            return_value=mocker.MagicMock(),
        )
        with pytest.raises(nncf.InternalError) as exc_info:
            algo.apply(None, None, stat_points)

        assert str(exc_info.value) == "Statistics were not collected for the node A"
