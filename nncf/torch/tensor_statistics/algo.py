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
from typing import Callable, Dict, Set, Union

import torch

from nncf.api.compression import CompressionStage
from nncf.common.schedulers import StubCompressionScheduler
from nncf.common.statistics import NNCFStatistics
from nncf.common.tensor_statistics.collectors import ReductionAxes
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.config import NNCFConfig
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.tensor import Tensor
from nncf.torch.algo_selector import ZeroCompressionLoss
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.dynamic_graph.context import no_nncf_trace
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.commands import TransformationPriority
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.return_types import maybe_get_values_from_torch_return_type


class TensorStatisticObservationPoint:
    def __init__(self, target_point: PTTargetPoint, reduction_shapes: Set[ReductionAxes] = None):
        self.target_point = target_point
        self.reduction_shapes = reduction_shapes

    def __hash__(self):
        return hash(self.target_point)

    def __eq__(self, other: "TensorStatisticObservationPoint"):
        return self.target_point == other.target_point


def create_register_input_hook(collector: TensorCollector) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Function to create register inputs hook function.

    :param collector: Collector to use in resulting hook.
    :return: Register inputs hook function.
    """

    def register_inputs_hook(x: Union[torch.Tensor, tuple]) -> torch.Tensor:
        """
        Register inputs hook function.

        :parameter x: tensor to register in hook.
        :return: tensor to register in hook.
        """
        with no_nncf_trace():
            x_unwrapped = maybe_get_values_from_torch_return_type(x)
            collector.register_input_for_all_reducers(Tensor(x_unwrapped))
        return x

    return register_inputs_hook


class TensorStatisticsCollectionBuilder(PTCompressionAlgorithmBuilder):
    def __init__(
        self,
        config: NNCFConfig,
        observation_points_vs_collectors: Dict[TensorStatisticObservationPoint, TensorStatisticCollectorBase],
    ):
        super().__init__(config)
        self._observation_points_vs_collectors = observation_points_vs_collectors

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        # Will it really suffice to use a single collector for all threads? After all, each of the threads
        # receives its own data, and should we use a thread-local collector, there would have to be a
        # separate thread reduction step involved. Still, is there a better option here than to rely on GIL?
        layout = PTTransformationLayout()
        for op, rs_vs_collector in self._observation_points_vs_collectors.items():
            for collector in rs_vs_collector.values():
                command = PTInsertionCommand(
                    op.target_point,
                    create_register_input_hook(collector=collector),
                    TransformationPriority.FP32_TENSOR_STATISTICS_OBSERVATION,
                )
                layout.register(command)
        return layout

    def _build_controller(self, model: NNCFNetwork) -> "TensorStatisticsCollectionController":
        return TensorStatisticsCollectionController(
            model, {k.target_point: v for k, v in self._observation_points_vs_collectors.items()}
        )

    def _handle_frozen_layers(self, target_model: NNCFNetwork):
        pass

    def initialize(self, model: NNCFNetwork) -> None:
        pass

    def _get_algo_specific_config_section(self) -> Dict:
        return {}


class TensorStatisticsCollectionController(PTCompressionAlgorithmController):
    def __init__(
        self, target_model: NNCFNetwork, ip_vs_collector_dict: Dict[PTTargetPoint, TensorStatisticCollectorBase]
    ):
        super().__init__(target_model)
        self.ip_vs_collector_dict = ip_vs_collector_dict
        self._scheduler = StubCompressionScheduler()
        self._loss = ZeroCompressionLoss("cpu")

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
