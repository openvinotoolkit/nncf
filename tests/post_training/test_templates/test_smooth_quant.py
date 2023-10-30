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
from typing import Callable, Dict, TypeVar

import pytest

from nncf.common.factory import NNCFGraphFactory
from nncf.common.factory import StatisticsAggregatorFactory
from nncf.common.graph.graph import NNCFNode
from nncf.experimental.common.tensor_statistics.collectors import AbsMaxReducer
from nncf.experimental.common.tensor_statistics.collectors import MaxAggregator
from nncf.parameters import ModelType
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import AdvancedSmoothQuantParameters
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.smooth_quant.algorithm import SmoothQuant
from nncf.quantization.algorithms.smooth_quant.backend import SmoothQuantAlgoBackend
from tests.post_training.test_templates.helpers import LinearMultiShapeModel
from tests.post_training.test_templates.helpers import NonZeroLinearModel
from tests.post_training.test_templates.helpers import get_static_dataset

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")


class TemplateTestSQAlgorithm:
    @staticmethod
    def fn_to_type(tensor) -> TTensor:
        return tensor

    @staticmethod
    @abstractmethod
    def get_transform_fn() -> Callable:
        """
        Get transformation function for dataset.
        """

    @staticmethod
    @abstractmethod
    def backend_specific_model(model: TModel, tmp_dir: str) -> TModel:
        """
        Return backend specific model.
        """

    @staticmethod
    @abstractmethod
    def check_scales(model: TModel, reference_values: Dict[str, TTensor]) -> None:
        """
        Checking scales from model with references.
        """

    @staticmethod
    @abstractmethod
    def get_backend() -> SmoothQuantAlgoBackend:
        """
        Returns backend-specific SmoothQuantAlgoBackend.
        """

    @staticmethod
    @abstractmethod
    def get_matmul_metatype():
        """
        Returns backend-specific MatMul metatype
        """

    @staticmethod
    def get_quantization_algorithm():
        return PostTrainingQuantization(
            subset_size=1,
            model_type=ModelType.TRANSFORMER,
            advanced_parameters=AdvancedQuantizationParameters(
                overflow_fix=OverflowFix.DISABLE,
                smooth_quant_params=AdvancedSmoothQuantParameters(matmul=0.95),
                inplace_statistics=False,
            ),
        )

    @pytest.mark.parametrize(
        "model_cls, reference_values",
        (
            (
                LinearMultiShapeModel,
                {
                    "/Add_0_0/nncf_smooth_quant": [[[[0.684977, 0.41752037, 0.51843774, 0.4748209]]]],
                    "/Add_2_0_0/nncf_smooth_quant": [
                        [[0.42803988, 0.5207079, 0.38387775, 0.49169466, 0.42942598, 0.2673041, 0.59089565, 0.5632654]]
                    ],
                    "/Add_3_0_0/nncf_smooth_quant": [[0.15430021]],
                    "/Add_4_0_0/nncf_smooth_quant": [
                        [0.15295707, 0.10677531, 0.4993426, 0.23435812, 0.37076628, 0.07112932, 0.21237421, 0.24852793]
                    ],
                    "/Add_5_0_0/nncf_smooth_quant": 0.13425705,
                    "/Add_5_0_1/nncf_smooth_quant": 0.13582686,
                },
            ),
        ),
    )
    def test_smooth_quant_algo(self, model_cls, reference_values, tmpdir):
        model = self.backend_specific_model(model_cls(), tmpdir)
        dataset = get_static_dataset(model_cls.INPUT_SIZE, self.get_transform_fn(), self.fn_to_type)

        quantization_algorithm = self.get_quantization_algorithm()
        graph = NNCFGraphFactory.create(model)
        quantized_model = quantization_algorithm.apply(model, graph, dataset=dataset)

        self.check_scales(quantized_model, reference_values)

    def test_get_abs_max_channel_collector(self):
        backend = self.get_backend()
        reduction_axes = (3, 2, 1)
        samples = 1

        for inplace_type in [False, True]:
            backend_tensor_collector = backend.get_abs_max_channel_collector(
                num_samples=samples,
                stats_reduction_axes=reduction_axes,
                inplace=inplace_type,
                branch_key="test_branch",
            )

            for aggregator in backend_tensor_collector.aggregators.values():
                assert isinstance(aggregator, MaxAggregator)

            for reducer in backend_tensor_collector.reducers:
                assert isinstance(reducer, AbsMaxReducer)
                assert reducer.inplace == inplace_type
                assert reducer._reduction_axes == reduction_axes

    @pytest.mark.parametrize(
        "model_cls, references",
        (
            (
                LinearMultiShapeModel,
                [
                    ("/MatMul_1", 0),
                    ("/MatMul", 0),
                    ("/MatMul_2", 0),
                    ("/MatMul_4", 1),
                    ("25", 1),
                    ("32", 0),
                    ("64", 1),
                    ("62", 0),
                ],
            ),
        ),
    )
    def test__get_nodes_to_smooth_data(self, model_cls, references, tmpdir):
        model = self.backend_specific_model(model_cls(), tmpdir)
        nncf_graph = NNCFGraphFactory.create(model)

        algo = SmoothQuant()
        algo._set_backend_entity(model)
        alpha_map = algo._get_alpha_map()
        smooth_data = algo._get_nodes_to_smooth_data(nncf_graph, alpha_map.keys())
        smooth_data = {d["node_to_smooth"].node_name: d["input_act_port"] for d in smooth_data}

        for ref_node_name, ref_port_id in references:
            assert ref_node_name in smooth_data
            assert smooth_data[ref_node_name] == ref_port_id

    def test_empty_stats(self, mocker, tmpdir):
        model_cls = NonZeroLinearModel
        model = self.backend_specific_model(model_cls(), tmpdir)
        dataset = get_static_dataset(model_cls.INPUT_SIZE, self.get_transform_fn(), self.fn_to_type)

        graph = NNCFGraphFactory.create(model)
        algo = SmoothQuant(subset_size=1, inplace_statistics=False)
        algo_statistic_points = algo.get_statistic_points(model, graph)
        statistics_aggregator = StatisticsAggregatorFactory.create(model, dataset)
        statistics_aggregator.register_statistic_points(algo_statistic_points)
        statistics_aggregator.collect_statistics(model, graph)

        mocked_transformer = mocker.MagicMock()
        mocker.patch("nncf.common.factory.ModelTransformerFactory.create", return_value=mocked_transformer)
        algo.apply(model, graph, algo_statistic_points)

        mocked_transformer.transform.assert_called_once()
        arg = mocked_transformer.transform.call_args.args[0]
        assert len(arg.transformations) == 2

        mm_metatype = self.get_matmul_metatype()
        matmuls = [node for node in graph.topological_sort() if node.metatype == mm_metatype]
        for transformation in arg.transformations:
            assert transformation.target_point.target_node_name != matmuls[0].node_name

    def test_get_activation_channel_axis(self, node_metatype, layer_attributes, port_id, reference_value):
        backend = self.get_backend()

        attributes = {
            NNCFNode.METATYPE_ATTR: node_metatype,
            NNCFNode.LAYER_ATTRIBUTES: layer_attributes,
            NNCFNode.NODE_NAME_ATTR: "test_node",
            NNCFNode.ID_NODE_ATTR: 0,
        }
        node = NNCFNode(attributes)

        try:
            activation_channel_axis = backend.get_activation_channel_axis(node, port_id)
        except RuntimeError as e:
            if isinstance(e, reference_value):
                pytest.xfail("Expected exception")

        assert activation_channel_axis == reference_value

    def test_get_weight_channel_axis(self, node_metatype, layer_attributes, port_id, reference_value):
        backend = self.get_backend()

        attributes = {
            NNCFNode.METATYPE_ATTR: node_metatype,
            NNCFNode.LAYER_ATTRIBUTES: layer_attributes,
            NNCFNode.NODE_NAME_ATTR: "test_node",
            NNCFNode.ID_NODE_ATTR: 0,
        }
        node = NNCFNode(attributes)

        try:
            activation_channel_axis = backend.get_weight_channel_axis(node, port_id)
        except RuntimeError as e:
            if isinstance(e, reference_value):
                pytest.xfail("Expected exception")

        assert activation_channel_axis == reference_value
