"""
 Copyright (c) 2019-2020 Intel Corporation
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
from typing import Set, List, Dict

from nncf.config import NNCFConfig
from nncf.compression_method_api import CompressionAlgorithmBuilder, CompressionAlgorithmController, CompressionLevel
from nncf.module_operations import UpdateWeight
from nncf.nncf_network import InsertionPoint, NNCFNetwork, InsertionCommand, OperationPriority, InsertionType
from nncf.tensor_statistics.collectors import TensorStatisticCollectorBase, MinMaxStatisticCollector, ReductionShape


class TensorStatisticObservationPoint:
    def __init__(self, insertion_point: InsertionPoint, reduction_shapes: Set[ReductionShape] = None):
        self.insertion_point = insertion_point
        self.reduction_shapes = reduction_shapes

    def __hash__(self):
        return hash(self.insertion_point)

    def __eq__(self, other: 'TensorStatisticObservationPoint'):
        return self.insertion_point == other.insertion_point


class TensorStatisticsCollectionBuilder(CompressionAlgorithmBuilder):
    def __init__(self, config: NNCFConfig, observation_points: Set[TensorStatisticObservationPoint]):
        super().__init__(config)
        self._observation_points = observation_points
        self._ip_vs_collector_dict = {} # type: Dict[InsertionPoint, TensorStatisticCollectorBase]

    def _apply_to(self, target_model: NNCFNetwork) -> List[InsertionCommand]:
        # Will it really suffice to use a single collector for all threads? After all, each of the threads
        # receives its own data, and should we use a thread-local collector, there would have to be a
        # separate thread reduction step involved. Still, is there a better option here than to rely on GIL?
        retval = []  # type: List[InsertionCommand]
        for op in self._observation_points:
            # TODO: factory
            collector = MinMaxStatisticCollector(reduction_shapes=op.reduction_shapes)
            self._ip_vs_collector_dict[op.insertion_point] = collector
            hook_obj = collector.register_input
            if op.insertion_point.insertion_type in [InsertionType.NNCF_MODULE_PRE_OP, InsertionType.NNCF_MODULE_POST_OP]:
                hook_obj = UpdateWeight(hook_obj)
            command = InsertionCommand(op.insertion_point, hook_obj,
                                       OperationPriority.FP32_TENSOR_STATISTICS_OBSERVATION)
            retval.append(command)
        return retval

    def build_controller(self, target_model: NNCFNetwork) -> 'TensorStatisticsCollectionController':
        return TensorStatisticsCollectionController(target_model, self._ip_vs_collector_dict)


class TensorStatisticsCollectionController(CompressionAlgorithmController):
    def __init__(self, target_model: NNCFNetwork,
                 ip_vs_collector_dict: Dict[InsertionPoint, TensorStatisticCollectorBase]):
        super().__init__(target_model)
        self.ip_vs_collector_dict = ip_vs_collector_dict

    def start_collection(self):
        for collector in self.ip_vs_collector_dict.values():
            collector.enable()

    def stop_collection(self):
        for collector in self.ip_vs_collector_dict.values():
            collector.disable()

    def compression_level(self) -> CompressionLevel:
        return CompressionLevel.FULL
