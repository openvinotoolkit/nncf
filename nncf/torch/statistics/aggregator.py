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

from copy import deepcopy
from typing import Dict

import numpy as np
import torch

from nncf.common.factory import TModel
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.tensor_statistics.aggregator import StatisticPointsContainer
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.tensor import PTNNCFTensor
from nncf.torch.tensor_statistics.algo import create_register_input_hook


class ModelView:
    def __init__(self, model: NNCFNetwork):
        self.model = model
        self.nncf_module_additions = self.model.nncf.save_nncf_module_additions()

    def __enter__(self):
        # Model ref removed to prevent copying
        self.model.nncf.update_model_ref(None)

        # nncf_replaced_models removed to prevent copying
        replaced_modules = self.model.nncf._nncf_replaced_modules
        self.model.nncf._nncf_replaced_modules = None

        self.nncf_interface = deepcopy(self.model.nncf)

        # Model ref is recovering
        self.model.nncf.update_model_ref(self.model)
        self.nncf_interface.update_model_ref(self.model)

        # nncf_replaced_models is recovering
        self.model.nncf._nncf_replaced_modules = replaced_modules
        self.nncf_interface._nncf_replaced_modules = replaced_modules
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model._nncf = self.nncf_interface
        self.model.nncf.reset_nncf_modules()
        self.model.nncf.load_nncf_module_additions(self.nncf_module_additions)


class PTStatisticsAggregator(StatisticsAggregator):
    HOOKS_GROUP_NAME = "statistics_hooks"

    def collect_statistics(self, model: NNCFNetwork, graph: NNCFGraph) -> None:
        with torch.no_grad():
            super().collect_statistics(model, graph)
        model.nncf.remove_hooks_group(self.HOOKS_GROUP_NAME)

    def _register_statistics(
        self, outputs: Dict[str, PTNNCFTensor], statistic_points: StatisticPointsContainer
    ) -> None:
        return

    def _get_transformation_layout_extra_outputs(
        self, statistic_points: StatisticPointsContainer
    ) -> TransformationLayout:
        transformation_layout = TransformationLayout()
        transformation_commands = []

        for _statistic_points in statistic_points.values():
            for _statistic_point in _statistic_points:
                for collectors in _statistic_point.algorithm_to_tensor_collectors.values():
                    for collector in collectors:
                        transformation_commands.append(
                            PTInsertionCommand(
                                _statistic_point.target_point,
                                create_register_input_hook(collector=collector),
                                TransformationPriority.FP32_TENSOR_STATISTICS_OBSERVATION,
                                hooks_group_name=self.HOOKS_GROUP_NAME,
                            )
                        )

        for transformation_command in transformation_commands:
            transformation_layout.register(transformation_command)

        return transformation_layout

    @staticmethod
    def _get_merged_statistic_points(
        statistic_points: StatisticPointsContainer, model: TModel, graph: NNCFGraph
    ) -> StatisticPointsContainer:
        # TODO: mirgate to experimental statistic collector and use common merging algorithm
        return statistic_points

    @staticmethod
    def _process_outputs(outputs: Dict[str, np.ndarray]) -> Dict[str, PTNNCFTensor]:
        return outputs
