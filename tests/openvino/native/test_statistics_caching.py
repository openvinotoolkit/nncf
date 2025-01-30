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
import numpy as np

from nncf.common.graph.transformations.commands import TargetType
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.statistics.aggregator import OVStatisticsAggregator
from nncf.tensor import Tensor
from tests.cross_fw.test_templates.test_statistics_caching import TemplateTestStatisticsCaching


class TestStatisticsCaching(TemplateTestStatisticsCaching):
    def create_target_point(self, target_point_type: TargetType, name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_point_type, name, port_id)

    def get_statistics_aggregator(self):
        return OVStatisticsAggregator(None)

    def _create_dummy_min_max_tensor(self) -> Tensor:
        return Tensor(np.zeros((3))), Tensor(np.ones((3)))
