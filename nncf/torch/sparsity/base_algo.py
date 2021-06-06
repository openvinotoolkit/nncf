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

from collections import namedtuple
from typing import List

from nncf.torch.algo_selector import ZeroCompressionLoss
from nncf.api.compression import CompressionStage
from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.sparsity.controller import SparsityController
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.common.utils.logger import logger as nncf_logger
from nncf.torch.graph.transformations.commands import TransformationPriority
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.common.schedulers import BaseCompressionScheduler
from nncf.torch.nncf_network import NNCFNetwork

SparseModuleInfo = namedtuple('SparseModuleInfo', ['module_name', 'module', 'operand'])


class BaseSparsityAlgoBuilder(PTCompressionAlgorithmBuilder):
    def __init__(self, config, should_init: bool = True):
        super().__init__(config, should_init)
        self._sparsified_module_info = []

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        layout = PTTransformationLayout()
        commands = self._sparsify_weights(target_model)
        for command in commands:
            layout.register(command)
        return layout

    def _sparsify_weights(self, target_model: NNCFNetwork) -> List[PTInsertionCommand]:
        device = next(target_model.parameters()).device
        sparsified_modules = target_model.get_nncf_modules_by_module_names(self.compressed_nncf_module_names)
        insertion_commands = []
        for module_scope, module in sparsified_modules.items():
            scope_str = str(module_scope)

            if not self._should_consider_scope(scope_str):
                nncf_logger.info("Ignored adding Weight Sparsifier in scope: {}".format(scope_str))
                continue

            nncf_logger.info("Adding Weight Sparsifier in scope: {}".format(scope_str))
            compression_lr_multiplier = self.config.get("compression_lr_multiplier", None)
            operation = self.create_weight_sparsifying_operation(module, compression_lr_multiplier)
            hook = operation.to(device)
            insertion_commands.append(PTInsertionCommand(PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS,
                                                                       module_scope=module_scope),
                                                         hook, TransformationPriority.SPARSIFICATION_PRIORITY))
            self._sparsified_module_info.append(
                SparseModuleInfo(scope_str, module, hook))

        return insertion_commands

    def create_weight_sparsifying_operation(self, target_module, compression_lr_multiplier):
        raise NotImplementedError


class BaseSparsityAlgoController(PTCompressionAlgorithmController, SparsityController):
    def __init__(self, target_model: NNCFNetwork,
                 sparsified_module_info: List[SparseModuleInfo]):
        super().__init__(target_model)
        self._loss = ZeroCompressionLoss(next(target_model.parameters()).device)
        self._scheduler = BaseCompressionScheduler()
        self.sparsified_module_info = sparsified_module_info

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    def compression_stage(self) -> CompressionStage:
        return CompressionStage.FULLY_COMPRESSED
