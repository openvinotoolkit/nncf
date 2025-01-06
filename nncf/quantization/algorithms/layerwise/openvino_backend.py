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

from typing import Dict, List, Optional

import openvino.runtime as ov

from nncf.common.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.data.dataset import Dataset
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.statistics.collectors import get_raw_stat_collector
from nncf.quantization.algorithms.layerwise.backend import LayerwiseEngineBackend
from nncf.quantization.algorithms.layerwise.openvino_iterator import OVLayerwiseIterator
from nncf.quantization.algorithms.layerwise.scheduler import LayerwiseStep
from nncf.quantization.algorithms.layerwise.scheduler import NodeOutputPort
from nncf.tensor import Tensor


class OVLayerwiseEngineBackend(LayerwiseEngineBackend):
    @staticmethod
    def create_layerwise_iterator(
        model: ov.Model,
        graph: NNCFGraph,
        schedule: List[LayerwiseStep],
        dataset: Dataset,
        subset_size: int = 100,
        cache: Optional[Dict[NodeOutputPort, List[Tensor]]] = None,
    ) -> OVLayerwiseIterator:
        return OVLayerwiseIterator(model, graph, schedule, dataset, subset_size, cache)

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def raw_statistic_collector(num_samples: Optional[int] = None) -> TensorStatisticCollectorBase:
        return get_raw_stat_collector(num_samples)
