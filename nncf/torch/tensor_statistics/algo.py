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
from typing import Set

from nncf.torch.algo_selector import ZeroCompressionLoss
from nncf.api.compression import CompressionStage
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.common.schedulers import StubCompressionScheduler
from nncf.common.statistics import NNCFStatistics
from nncf.config import NNCFConfig
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.graph.transformations.commands import TransformationPriority
from nncf.torch.tensor_statistics.collectors import ReductionShape
from nncf.torch.tensor_statistics.collectors import TensorStatisticCollectorBase


class TensorStatisticObservationPoint:
    def __init__(self, insertion_point: PTTargetPoint,
                 reduction_shapes: Set[ReductionShape] = None):
        self.insertion_point = insertion_point
        self.reduction_shapes = reduction_shapes

    def __hash__(self):
        return hash(self.insertion_point)

    def __eq__(self, other: 'TensorStatisticObservationPoint'):
        return self.insertion_point == other.insertion_point


class TensorStatisticsCollectionBuilder(PTCompressionAlgorithmBuilder):
    def __init__(self, config: NNCFConfig,
                 observation_points_vs_collectors: Dict[TensorStatisticObservationPoint,
                                                        TensorStatisticCollectorBase]):
        super().__init__(config)
        self._observation_points_vs_collectors = observation_points_vs_collectors

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        # Will it really suffice to use a single collector for all threads? After all, each of the threads
        # receives its own data, and should we use a thread-local collector, there would have to be a
        # separate thread reduction step involved. Still, is there a better option here than to rely on GIL?
        layout = PTTransformationLayout()
        for op, collector in self._observation_points_vs_collectors.items():
            hook_obj = collector.register_input
            command = PTInsertionCommand(op.insertion_point, hook_obj,
                                         TransformationPriority.FP32_TENSOR_STATISTICS_OBSERVATION)
            layout.register(command)
        return layout

    def build_controller(self, target_model: NNCFNetwork) -> 'TensorStatisticsCollectionController':
        return TensorStatisticsCollectionController(target_model,
                                                    {k.insertion_point: v
                                                     for k, v in self._observation_points_vs_collectors.items()})

    def _handle_frozen_layers(self, target_model: NNCFNetwork):
        pass

    def initialize(self, model: NNCFNetwork) -> None:
        pass


class TensorStatisticsCollectionController(PTCompressionAlgorithmController):
    def __init__(self, target_model: NNCFNetwork,
                 ip_vs_collector_dict: Dict[PTTargetPoint, TensorStatisticCollectorBase]):
        super().__init__(target_model)
        self.ip_vs_collector_dict = ip_vs_collector_dict
        self._scheduler = StubCompressionScheduler()
        self._loss = ZeroCompressionLoss('cpu')

    @property
    def loss(self) -> ZeroCompressionLoss:
        return self._loss

    @property
    def scheduler(self) -> StubCompressionScheduler:
        return self._scheduler

    def start_collection(self):
        for collector in self.ip_vs_collector_dict.values():
            collector.enable()

    def stop_collection(self):
        for collector in self.ip_vs_collector_dict.values():
            collector.disable()

    def compression_stage(self) -> CompressionStage:
        return CompressionStage.FULLY_COMPRESSED

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        return NNCFStatistics()
