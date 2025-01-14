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
"""
Base classes for NNCF PyTorch sparsity algorithm builder and controller objects.
"""
from typing import List

import torch

from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionStage
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.logging import nncf_logger
from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.schedulers import StubCompressionScheduler
from nncf.common.sparsity.controller import SparsityController
from nncf.common.sparsity.schedulers import SparsityScheduler
from nncf.common.utils.api_marker import api
from nncf.common.utils.backend import copy_model
from nncf.torch.algo_selector import ZeroCompressionLoss
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.commands import TransformationPriority
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.sparsity.layers import BinaryMask
from nncf.torch.utils import get_model_device


class SparseModuleInfo:
    def __init__(self, module_node_name: NNCFNodeName, module: torch.nn.Module, operand):
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
        device = get_model_device(target_model)
        sparsified_module_nodes = target_model.nncf.get_weighted_original_graph_nodes(
            nncf_module_names=self.compressed_nncf_module_names
        )
        insertion_commands = []
        for module_node in sparsified_module_nodes:
            node_name = module_node.node_name

            if not self._should_consider_scope(node_name):
                nncf_logger.info(f"Ignored adding weight sparsifier for operation: {node_name}")
                continue

            compression_lr_multiplier = self.config.get_redefinable_global_param_value_for_algo(
                "compression_lr_multiplier", self.name
            )
            operation = self.create_weight_sparsifying_operation(module_node, compression_lr_multiplier)
            hook = operation.to(device)
            insertion_commands.append(
                PTInsertionCommand(
                    PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS, target_node_name=node_name),
                    hook,
                    TransformationPriority.SPARSIFICATION_PRIORITY,
                )
            )
            sparsified_module = target_model.nncf.get_containing_module(node_name)
            self._sparsified_module_info.append(SparseModuleInfo(node_name, sparsified_module, hook))

        return insertion_commands

    def create_weight_sparsifying_operation(self, target_module_node: NNCFNode, compression_lr_multiplier: float):
        raise NotImplementedError

    def initialize(self, model: NNCFNetwork) -> None:
        pass


@api()
class BaseSparsityAlgoController(PTCompressionAlgorithmController, SparsityController):
    """
    Base class for sparsity algorithm controllers in PT.
    """

    def __init__(self, target_model: NNCFNetwork, sparsified_module_info: List[SparseModuleInfo]):
        super().__init__(target_model)
        self._loss = ZeroCompressionLoss(get_model_device(target_model))
        self._scheduler = BaseCompressionScheduler()
        self.sparsified_module_info = sparsified_module_info

    @property
    def current_sparsity_level(self) -> float:
        return self._scheduler.current_sparsity_level

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    @property
    def scheduler(self) -> SparsityScheduler:
        return self._scheduler

    def disable_scheduler(self):
        self._scheduler = StubCompressionScheduler()
        self._scheduler.target_level = 0.0
        self._scheduler.current_sparsity_level = 0.0

    def compression_stage(self) -> CompressionStage:
        return CompressionStage.FULLY_COMPRESSED

    def strip_model(self, model: NNCFNetwork, do_copy: bool = False) -> NNCFNetwork:
        if do_copy:
            model = copy_model(model)

        for node in model.nncf.get_original_graph().get_all_nodes():
            if node.node_type in ["nncf_model_input", "nncf_model_output"]:
                continue
            nncf_module = model.nncf.get_containing_module(node.node_name)
            if hasattr(nncf_module, "pre_ops"):
                for key in list(nncf_module.pre_ops.keys()):
                    op = nncf_module.get_pre_op(key)
                    if isinstance(op.operand, BinaryMask):
                        nncf_module.weight.data = op.operand.apply_binary_mask(nncf_module.weight.data)
                        nncf_module.remove_pre_forward_operation(key)

        return model
