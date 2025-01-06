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
from typing import Dict

import torch

from nncf.api.compression import CompressionScheduler
from nncf.api.compression import CompressionStage
from nncf.common.compression import NO_COMPRESSION_ALGORITHM_NAME
from nncf.common.schedulers import StubCompressionScheduler
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.backend import copy_model
from nncf.common.utils.registry import Registry
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.compression_method_api import PTCompressionLoss
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.utils import get_model_device

PT_COMPRESSION_ALGORITHMS = Registry("compression algorithm", add_name_as_attr=True)


class ZeroCompressionLoss(PTCompressionLoss):
    def __init__(self, device: torch.device):
        super().__init__()
        self._device = device

    def calculate(self) -> torch.Tensor:
        return torch.zeros([], device=self._device)


@PT_COMPRESSION_ALGORITHMS.register(NO_COMPRESSION_ALGORITHM_NAME)
class NoCompressionAlgorithmBuilder(PTCompressionAlgorithmBuilder):
    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        return PTTransformationLayout()

    def _get_algo_specific_config_section(self) -> Dict:
        return {}

    def _build_controller(self, model: NNCFNetwork) -> PTCompressionAlgorithmController:
        return NoCompressionAlgorithmController(model)

    def initialize(self, model: NNCFNetwork) -> None:
        pass

    def _handle_frozen_layers(self, target_model: NNCFNetwork):
        pass


class NoCompressionAlgorithmController(PTCompressionAlgorithmController):
    def __init__(self, target_model):
        super().__init__(target_model)

        self._loss = ZeroCompressionLoss(get_model_device(target_model))
        self._scheduler = StubCompressionScheduler()

    def compression_stage(self) -> CompressionStage:
        """
        Returns level of compression. Should be used on saving best checkpoints to distinguish between
        uncompressed, partially compressed and fully compressed models.
        """
        return CompressionStage.UNCOMPRESSED

    @property
    def loss(self) -> ZeroCompressionLoss:
        return self._loss

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        return NNCFStatistics()

    def strip(self, do_copy: bool = True) -> NNCFNetwork:
        model = self.model
        if do_copy:
            model = copy_model(self.model)
        return model
