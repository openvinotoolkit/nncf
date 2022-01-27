"""
 Copyright (c) 2022 Intel Corporation
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

from copy import deepcopy

from torch import nn

from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.statistics import NNCFStatistics
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf import NNCFConfig
from nncf.torch.knowledge_distillation.knowledge_distillation_loss import KnowledgeDistillationLoss
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.api.compression import CompressionLoss, CompressionScheduler, CompressionStage
from nncf.torch.algo_selector import PT_COMPRESSION_ALGORITHMS


@PT_COMPRESSION_ALGORITHMS.register('knowledge_distillation')
class KnowledgeDistillationBuilder(PTCompressionAlgorithmBuilder):
    def __init__(self, config: NNCFConfig, should_init: bool = True):
        super().__init__(config, should_init)
        self.kd_type = self._algo_config.get('type', None)
        self.scale = self._algo_config.get('scale', 1)
        self.temperature = self._algo_config.get('temperature', 1)
        if 'temperature' in self._algo_config.keys() and self.kd_type == 'mse':
            raise ValueError("Temperature shouldn't be stated for MSE Loss (softmax only feature)")

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        self.original_model = deepcopy(target_model.nncf_module)
        for param in self.original_model.parameters():
            param.requires_grad = False
        return PTTransformationLayout()

    def _build_controller(self, model):
        return KnowledgeDistillationController(model, self.original_model, self.kd_type, self.scale, self.temperature)

    def initialize(self, model: NNCFNetwork) -> None:
        pass


class KnowledgeDistillationController(PTCompressionAlgorithmController):
    def __init__(self, target_model: NNCFNetwork, original_model: nn.Module, kd_type: str, scale: float,
                 temperature: float):
        super().__init__(target_model)
        original_model.train()
        self._scheduler = BaseCompressionScheduler()
        self._loss = KnowledgeDistillationLoss(target_model=target_model,
                                               original_model=original_model,
                                               kd_type=kd_type,
                                               scale=scale,
                                               temperature=temperature)

    def compression_stage(self) -> CompressionStage:
        """
        Returns level of compression. Should be used on saving best checkpoints to distinguish between
        uncompressed, partially compressed and fully compressed models.
        """
        return CompressionStage.FULLY_COMPRESSED

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        return NNCFStatistics()

    def distributed(self):
        pass
