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

from typing import List

from nncf.torch.algo_selector import ZeroCompressionLoss
import torch
from nncf.api.compression import CompressionStage
from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
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


class SparseModuleInfo:
    def __init__(self, module_node_name: NNCFNodeName, module: torch.nn.Module,
                 operand):
        self.module_node_name = module_node_name
        self.module = module
        self.operand = operand


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
        sparsified_module_nodes = target_model.get_weighted_original_graph_nodes(
            nncf_module_names=self.compressed_nncf_module_names)
        insertion_commands = []
        for module_node in sparsified_module_nodes:
            node_name = module_node.node_name

            if not self._should_consider_scope(node_name):
                nncf_logger.info("Ignored adding Weight Sparsifier in scope: {}".format(node_name))
                continue

            nncf_logger.info("Adding Weight Sparsifier in scope: {}".format(node_name))
            compression_lr_multiplier = self.config.get("compression_lr_multiplier", None)
            operation = self.create_weight_sparsifying_operation(module_node, compression_lr_multiplier)
            hook = operation.to(device)
            insertion_commands.append(PTInsertionCommand(PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS,
                                                                       target_node_name=node_name),
                                                         hook, TransformationPriority.SPARSIFICATION_PRIORITY))
            sparsified_module = target_model.get_containing_module(node_name)
            self._sparsified_module_info.append(
                SparseModuleInfo(node_name, sparsified_module, hook))

        return insertion_commands

    def create_weight_sparsifying_operation(self, target_module_node: NNCFNode, compression_lr_multiplier: float):
        raise NotImplementedError

    def initialize(self, model: NNCFNetwork) -> None:
        pass


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
