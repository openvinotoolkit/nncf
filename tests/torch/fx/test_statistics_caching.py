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
import torch

from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.torch.fx.statistics.aggregator import FXStatisticsAggregator
from nncf.tensor import Tensor
from nncf.torch.graph.transformations.commands import PTTargetPoint
from tests.cross_fw.test_templates.test_statistics_caching import TemplateTestStatisticsCaching


class TestStatisticsCaching(TemplateTestStatisticsCaching):
    def create_target_point(self, target_point_type: TargetType, name: str, port_id: int) -> PTTargetPoint:
        return PTTargetPoint(target_type=target_point_type, target_node_name=name, input_port_id=port_id)

    def get_statistics_aggregator(self):
        return FXStatisticsAggregator(None)

    def _create_dummy_min_max_tensor(self) -> Tensor:
        return Tensor(torch.zeros((3))), Tensor(torch.ones((3)))
