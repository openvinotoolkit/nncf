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
from typing import TypeVar

from nncf.common.factory import NNCFGraphFactory
from nncf.common.factory import StatisticsAggregatorFactory
from nncf.quantization.algorithms.layerwise.engine import LayerwiseEngine
from nncf.tensor import functions as fns
from tests.cross_fw.test_templates.helpers import ConvTestModel
from tests.cross_fw.test_templates.helpers import get_static_dataset

TModel = TypeVar("TModel")


class TemplateTestLayerwiseEngine:
    @staticmethod
    @abstractmethod
    def get_transform_fn():
        """
        Get transformation function for dataset.
        """

    @staticmethod
    @abstractmethod
    def backend_specific_model(model: TModel, tmp_dir: str):
        """
        Return backend specific model.
        """

    def iterate_trough_target_nodes_reference(self):
        return [
            (
                "__module.conv/aten::_convolution/Convolution",
                {
                    0: [
                        [
                            [
                                [0.54881352186203, 0.7151893377304077, 0.6027633547782898, 0.5448831915855408],
                                [0.42365479469299316, 0.6458941102027893, 0.4375872015953064, 0.891772985458374],
                                [0.9636627435684204, 0.3834415078163147, 0.7917250394821167, 0.5288949012756348],
                                [0.5680445432662964, 0.9255966544151306, 0.07103605568408966, 0.08712930232286453],
                            ]
                        ]
                    ]
                },
            ),
            (
                "__module.conv/aten::_convolution/Add",
                {
                    0: [
                        [
                            [
                                [-0.8872531056404114, -0.44435498118400574, -0.5027254819869995],
                                [-0.24741590023040771, -0.34797102212905884, -0.8951727151870728],
                                [-0.009912519715726376, -0.6124056577682495, -0.8988683223724365],
                            ],
                            [
                                [1.1261945962905884, 0.6749102473258972, 0.8016328811645508],
                                [0.4088350534439087, 0.6354947686195374, 1.08846914768219],
                                [0.38776442408561707, 0.703301191329956, 1.074639081954956],
                            ],
                        ]
                    ]
                },
            ),
        ]

    def test_iterate_trough_target_nodes(self, tmpdir):
        model = self.backend_specific_model(ConvTestModel(), tmpdir)
        dataset = get_static_dataset(ConvTestModel.INPUT_SIZE, self.get_transform_fn())

        graph = NNCFGraphFactory.create(model)
        output_nodes = graph.get_output_nodes()
        target_nodes = []
        for node in graph.get_all_nodes():
            if graph.get_previous_nodes(node) and node not in output_nodes:
                target_nodes.append(node)

        engine = LayerwiseEngine(subset_size=1)
        target_nodes_iterator = engine.create_iterator_through_target_nodes(
            model=model, graph=graph, target_nodes=target_nodes, dataset=dataset
        )

        reference = self.iterate_trough_target_nodes_reference()
        for target, reference in zip(target_nodes_iterator, self.iterate_trough_target_nodes_reference()):
            assert target[0].node_name == reference[0]
            for port, value in target[1].items():
                assert port in reference[1]
                assert fns.allclose(value[0], reference[1][port])

    def test_iterate_trough_target_nodes_with_statistics(self, tmpdir):
        model = self.backend_specific_model(ConvTestModel(), tmpdir)
        dataset = get_static_dataset(ConvTestModel.INPUT_SIZE, self.get_transform_fn())

        graph = NNCFGraphFactory.create(model)
        output_nodes = graph.get_output_nodes()
        target_nodes = []
        for node in graph.get_all_nodes():
            if graph.get_previous_nodes(node) and node not in output_nodes:
                target_nodes.append(node)

        engine = LayerwiseEngine(subset_size=1)

        statistic_points = engine.get_statistic_points(model=model, graph=graph, target_nodes=target_nodes)
        statistics_aggregator = StatisticsAggregatorFactory.create(model, dataset)
        statistics_aggregator.register_statistic_points(statistic_points)
        statistics_aggregator.collect_statistics(model, graph)

        target_nodes_iterator = engine.create_iterator_through_target_nodes(
            model=model, graph=graph, target_nodes=target_nodes, dataset=dataset, statistic_points=statistic_points
        )

        reference = self.iterate_trough_target_nodes_reference()
        for target, reference in zip(target_nodes_iterator, self.iterate_trough_target_nodes_reference()):
            assert target[0].node_name == reference[0]
            for port, value in target[1].items():
                assert port in reference[1]
                assert fns.allclose(value[0], reference[1][port])
