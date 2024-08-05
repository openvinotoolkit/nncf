# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Type, TypeVar, Optional

import numpy as np
import torch
import torch.nn as nn
from openvino.runtime import opset13 as opset
import openvino.runtime

import nncf
import nncf.tensor.functions as fns
from nncf.common.factory import StatisticsAggregatorFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import CONST_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer, StatisticPoint
from nncf.data import Dataset
from nncf.experimental.common.tensor_statistics.collectors import OnlineAggregatorBase, AggregationAxes, TensorCollector
from nncf.experimental.torch.sparsify_activations.sparsify_activations_impl import SparsifyActivationsAlgoBackend
from nncf.openvino.graph.model_transformer import OVModelTransformer
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.statistics.aggregator import OVStatisticsAggregator
from nncf.openvino.statistics.collectors import OVAbsQuantileReducer
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor.functions.torch_numeric import quantile
from nncf.openvino.graph.metatypes import openvino_metatypes as om
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.model_transformer import PTModelTransformer
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.utils import training_mode_switcher

ACTIVATIONS_SPARSIFIER_PREFIX = "activations_sparsifier"
STATISTIC_BRANCH_KEY = "abs_quantile"
ALGORITHM_KEY = "AS"
TModel = TypeVar("TModel")


class EMAAggregator(OnlineAggregatorBase):
    def __init__(
            self,
            alpha: float,
            num_samples: Optional[int] = None,
            window_size: Optional[int] = None,
    ):
        self._alpha = alpha
        super().__init__(aggregation_axes=(0,), num_samples=num_samples, window_size=window_size)

    def _aggregation_fn(self, stacked_value: Tensor, axis: AggregationAxes, keepdims: bool) -> Tensor:
        if self._collected_samples == 0:
            return stacked_value
        else:
            beta = 1.0 - self._alpha
            new_value = fns.expand_dims(stacked_value[0], 0)
            old_value = fns.expand_dims(stacked_value[1], 0)
            return (new_value * self._alpha + old_value * beta * (1 - beta ** self._collected_samples) /
                    (1 - beta ** (self._collected_samples + 1)))


class OVSparsifyActivationsAlgoBackend(SparsifyActivationsAlgoBackend):
    """
    OpenVINO backend for the activation sparsification algorithm.
    """

    SUPPORTED_METATYPES = [om.OVMatMulMetatype]

    @property
    def supported_metatypes(self) -> List[Type[OperatorMetatype]]:
        return OVSparsifyActivationsAlgoBackend.SUPPORTED_METATYPES

    def insert_sparsifiers(
        self,
        model: NNCFNetwork,
        graph: NNCFGraph,
        target_sparsity_by_node: Dict[NNCFNode, float],
    ) -> NNCFNetwork:
        return model

    def calibrate_sparsifiers(self, model: TModel, graph: NNCFGraph, dataset: Dataset) -> TModel:
        return None

    def do_sparsification(self, model, graph, target_sparsity_by_node, dataset: Dataset):
        statistic_points_container = StatisticPointsContainer()
        for node, sparsity in target_sparsity_by_node.items():
            stat_collector = TensorCollector()
            stat_collector.register_statistic_branch(
                container_key=STATISTIC_BRANCH_KEY,
                reducer=OVAbsQuantileReducer(quantile=[sparsity,]),
                aggregator=EMAAggregator(alpha=0.2)
            )
            activation_port_id = self._get_activation_port_id(node, graph)
            statistic_point = StatisticPoint(
                target_point=OVTargetPoint(TargetType.PRE_LAYER_OPERATION, node.node_name, port_id=activation_port_id),
                tensor_collector=stat_collector,
                algorithm=ALGORITHM_KEY,
            )
            statistic_points_container.add_statistic_point(statistic_point)

        statistics_aggregator = StatisticsAggregatorFactory.create(model, dataset)
        statistics_aggregator.register_statistic_points(statistic_points_container)
        statistics_aggregator.collect_statistics(model, graph)

        name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)
        for nncf_node in target_sparsity_by_node.keys():
            for tensor_collector in statistic_points_container.get_algo_statistics_for_node(
                nncf_node.node_name, lambda args: True, ALGORITHM_KEY
            ):
                activation_port_id = self._get_activation_port_id(nncf_node, graph)
                threshold = tensor_collector.get_statistics()[STATISTIC_BRANCH_KEY].data
                matmul_node = name_to_node_mapping[nncf_node.node_name]
                dense_activation = matmul_node.input(activation_port_id).get_source_output().get_node()

                dtype = dense_activation.get_element_type()
                threshold_const = opset.constant(threshold, dtype=dtype, name=f"{matmul_node.name}/sparsity_threshold")
                zero_const = opset.constant(0.0, dtype=dtype)

                less_mask = opset.less_equal(opset.abs(dense_activation), threshold_const)
                sparse_activation = opset.select(less_mask, zero_const, dense_activation, name=f"{matmul_node.name}/sparse_input")
                matmul_node.input(activation_port_id).replace_source_output(sparse_activation.output(0))

        return model


    @staticmethod
    def _get_activation_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> int:
        constant_ports = node.layer_attributes.get_const_port_ids()
        activation_ports = [
            e.input_port_id for e in nncf_graph.get_input_edges(node) if e.input_port_id not in constant_ports
        ]
        assert len(activation_ports) == 1
        return activation_ports[0]
