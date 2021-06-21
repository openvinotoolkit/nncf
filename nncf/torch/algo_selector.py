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

# pylint:disable=relative-beyond-top-level
import torch

from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.nncf_network import NNCFNetwork

from nncf.api.compression import CompressionStage
from nncf.api.compression import CompressionScheduler
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmController

from nncf.torch.compression_method_api import PTCompressionLoss
from nncf.common.schedulers import StubCompressionScheduler
from nncf.common.utils.registry import Registry
from nncf.common.statistics import NNCFStatistics


COMPRESSION_ALGORITHMS = Registry('compression algorithm', add_name_as_attr=True)


class ZeroCompressionLoss(PTCompressionLoss):
    def __init__(self, device: str):
        super().__init__()
        self._device = device

    def calculate(self) -> torch.Tensor:
        return torch.zeros([], device=self._device)


@COMPRESSION_ALGORITHMS.register('NoCompressionAlgorithmBuilder')
class NoCompressionAlgorithmBuilder(PTCompressionAlgorithmBuilder):
    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        return PTTransformationLayout()

    def build_controller(self, target_model: NNCFNetwork) -> PTCompressionAlgorithmController:
        return NoCompressionAlgorithmController(target_model)

    def initialize(self, model: NNCFNetwork) -> None:
        pass


# pylint:disable=abstract-method
class NoCompressionAlgorithmController(PTCompressionAlgorithmController):
    def __init__(self, target_model):
        super().__init__(target_model)
        self._loss = ZeroCompressionLoss(next(target_model.parameters()).device)
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
