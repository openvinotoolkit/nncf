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

from texttable import Texttable

from nncf.algo_selector import ZeroCompressionLoss
from nncf.api.compression import CompressionLevel
from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.sparsity.controller import SparsityController
from nncf.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.compression_method_api import PTCompressionAlgorithmController
from nncf.graph.transformations.layout import PTTransformationLayout
from nncf.layer_utils import COMPRESSION_MODULES
from nncf.common.utils.logger import logger as nncf_logger
from nncf.graph.transformations.commands import TransformationPriority
from nncf.graph.transformations.commands import PTTargetPoint
from nncf.graph.transformations.commands import PTInsertionCommand
from nncf.common.schedulers import BaseCompressionScheduler
from nncf.nncf_network import NNCFNetwork

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
            operation = self.create_weight_sparsifying_operation(module)
            hook = operation.to(device)
            insertion_commands.append(PTInsertionCommand(PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS,
                                                                       module_scope=module_scope),
                                                         hook, TransformationPriority.SPARSIFICATION_PRIORITY))
            self._sparsified_module_info.append(
                SparseModuleInfo(scope_str, module, hook))

        return insertion_commands

    def create_weight_sparsifying_operation(self, target_module):
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

    @property
    def sparsified_weights_count(self):
        count = 0
        for minfo in self.sparsified_module_info:
            count = count + minfo.module.weight.view(-1).size(0)
        return max(count, 1)

    def sparsity_rate_for_sparsified_modules(self, target_sparsified_module_info=None):
        if target_sparsified_module_info is None:
            target_sparsified_module_info = self.sparsified_module_info
        else:
            target_sparsified_module_info = [target_sparsified_module_info]

        nonzero = 0
        count = 0
        for minfo in target_sparsified_module_info:
            mask = minfo.operand.apply_binary_mask(minfo.module.weight)
            nonzero = nonzero + mask.nonzero().size(0)
            count = count + mask.view(-1).size(0)

        return 1 - nonzero / max(count, 1)

    @property
    def sparsity_rate_for_model(self):
        nonzero = 0
        count = 0

        for m in self._model.modules():
            if isinstance(m, tuple(COMPRESSION_MODULES.registry_dict.values())):
                continue

            sparsified_module = False
            for minfo in self.sparsified_module_info:
                if minfo.module == m:
                    mask = minfo.operand.apply_binary_mask(m.weight)
                    nonzero = nonzero + mask.nonzero().size(0)
                    count = count + mask.numel()

                    if not m.bias is None:
                        nonzero = nonzero + m.bias.nonzero().size(0)
                        count = count + m.bias.numel()

                    sparsified_module = True

            if not sparsified_module:
                for param in m.parameters(recurse=False):
                    nonzero = nonzero + param.nonzero().size(0)
                    count = count + param.numel()

        return 1 - nonzero / max(count, 1)

    def statistics(self, quickly_collected_only=False):
        stats = super().statistics(quickly_collected_only)
        table = Texttable()
        header = ["Name", "Weight's Shape", "SR", "% weights"]
        data = [header]

        sparsified_weights_count = self.sparsified_weights_count

        for minfo in self.sparsified_module_info:
            drow = {h: 0 for h in header}
            drow["Name"] = minfo.module_name
            drow["Weight's Shape"] = list(minfo.module.weight.size())
            mask = minfo.operand.apply_binary_mask(minfo.module.weight)
            nonzero = mask.nonzero().size(0)
            drow["SR"] = 1.0 - nonzero / max(mask.view(-1).size(0), 1)
            drow["% weights"] = (mask.view(-1).size(0) / sparsified_weights_count) * 100
            row = [drow[h] for h in header]
            data.append(row)
        table.add_rows(data)

        stats["sparsity_statistic_by_module"] = table
        stats["sparsity_rate_for_sparsified_modules"] = self.sparsity_rate_for_sparsified_modules()
        stats["sparsity_rate_for_model"] = self.sparsity_rate_for_model

        return stats

    def compression_level(self) -> CompressionLevel:
        return CompressionLevel.FULL
