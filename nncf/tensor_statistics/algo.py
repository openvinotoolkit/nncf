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
from typing import Dict
from typing import List
from typing import Set

from nncf.compression_method_api import CompressionAlgorithmBuilder
from nncf.compression_method_api import CompressionAlgorithmController
from nncf.compression_method_api import CompressionLevel
from nncf.config import NNCFConfig
from nncf.module_operations import UpdateWeight
from nncf.nncf_network import InsertionCommand
from nncf.nncf_network import InsertionPoint
from nncf.nncf_network import InsertionType
from nncf.nncf_network import NNCFNetwork
from nncf.nncf_network import OperationPriority
from nncf.tensor_statistics.collectors import ReductionShape
from nncf.tensor_statistics.collectors import TensorStatisticCollectorBase


class TensorStatisticObservationPoint:
    def __init__(self, insertion_point: InsertionPoint,
                 reduction_shapes: Set[ReductionShape] = None):
        self.insertion_point = insertion_point
        self.reduction_shapes = reduction_shapes

    def __hash__(self):
        return hash(self.insertion_point)

    def __eq__(self, other: 'TensorStatisticObservationPoint'):
        return self.insertion_point == other.insertion_point


class TensorStatisticsCollectionBuilder(CompressionAlgorithmBuilder):
    def __init__(self, config: NNCFConfig,
                 observation_points_vs_collectors: Dict[TensorStatisticObservationPoint,
                                                        TensorStatisticCollectorBase]):
        super().__init__(config)
        self._observation_points_vs_collectors = observation_points_vs_collectors

    def _apply_to(self, target_model: NNCFNetwork) -> List[InsertionCommand]:
        # Will it really suffice to use a single collector for all threads? After all, each of the threads
        # receives its own data, and should we use a thread-local collector, there would have to be a
        # separate thread reduction step involved. Still, is there a better option here than to rely on GIL?
        retval = []  # type: List[InsertionCommand]
        for op, collector in self._observation_points_vs_collectors.items():
            hook_obj = collector.register_input
            is_weights = op.insertion_point.insertion_type in [InsertionType.NNCF_MODULE_PRE_OP,
                                                               InsertionType.NNCF_MODULE_POST_OP]
            if is_weights:
                hook_obj = UpdateWeight(hook_obj)
            command = InsertionCommand(op.insertion_point, hook_obj,
                                       OperationPriority.FP32_TENSOR_STATISTICS_OBSERVATION)
            retval.append(command)
        return retval

    def build_controller(self, target_model: NNCFNetwork) -> 'TensorStatisticsCollectionController':
        return TensorStatisticsCollectionController(target_model,
                                                    {k.insertion_point: v
                                                     for k, v in self._observation_points_vs_collectors.items()})

    def _handle_frozen_layers(self):
        pass


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
