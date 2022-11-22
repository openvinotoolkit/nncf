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
from copy import deepcopy
from typing import DefaultDict, List, OrderedDict, Optional
from collections import OrderedDict
import torch
import torch.distributed as dist

from nncf import NNCFConfig
from nncf.config.extractors import extract_algo_specific_config
from nncf.torch.algo_selector import PT_COMPRESSION_ALGORITHMS
from nncf.api.compression import CompressionStage
from nncf.common.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.utils.logger import logger as nncf_logger
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.sparsity.base_algo import BaseSparsityAlgoBuilder, BaseSparsityAlgoController, SparseModuleInfo
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.commands import TransformationPriority
from nncf.experimental.torch.sparsity.movement.layers import MovementSparsifier, SparseConfig, SparseStructure
from nncf.experimental.torch.sparsity.movement.layers import SparseConfigByScope
from nncf.experimental.torch.sparsity.movement.loss import ImportanceLoss
from nncf.experimental.torch.sparsity.movement.scheduler import MovementPolynomialThresholdScheduler
from nncf.experimental.torch.sparsity.movement.structured_mask_handler import StructuredMaskHandler, SparsifiedModuleInfoGroup
from nncf.torch.module_operations import UpdateWeightAndBias
from nncf.torch.utils import get_world_size, get_model_device
from nncf.common.utils.helpers import matches_any
from nncf.common.accuracy_aware_training.training_loop import ADAPTIVE_COMPRESSION_CONTROLLERS
from nncf.torch.sparsity.collector import PTSparseModelStatisticsCollector
from nncf.common.sparsity.schedulers import SPARSITY_SCHEDULERS
from nncf.common.schedulers import StubCompressionScheduler
from nncf.common.sparsity.statistics import MovementSparsityStatistics
from nncf.common.statistics import NNCFStatistics
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlock, get_building_blocks, BuildingBlockType, BlockFilteringStrategy
from collections import defaultdict, namedtuple
from nncf.torch.dynamic_graph.operation_address import OperationAddress
import networkx as nx
from nncf.torch.layers import NNCF_MODULES_OP_NAMES, NNCFLinear
import os
import numpy as np
import pandas as pd
from nncf.experimental.torch.sparsity.movement.structured_mask_strategy import STRUCTURED_MASK_STRATEGY, detect_supported_model_family

SUPPORTED_NNCF_MODULES = [NNCFLinear]


@PT_COMPRESSION_ALGORITHMS.register('movement_sparsity')
class MovementSparsityBuilder(BaseSparsityAlgoBuilder):
    def __init__(self, config, should_init: bool = True):
        super().__init__(config, should_init)
        configs = self._algo_config.get('sparse_structure_by_scopes', [])
        self._sparse_configs_by_scopes = [SparseConfigByScope.from_config(c) for c in configs]

    def _sparsify_weights(self, target_model: NNCFNetwork) -> List[PTInsertionCommand]:
        device = get_model_device(target_model)
        sparsified_module_nodes = target_model.get_weighted_original_graph_nodes(
            nncf_module_names=[m.__name__ for m in SUPPORTED_NNCF_MODULES])
        insertion_commands = []
        for module_node in sparsified_module_nodes:
            node_name = module_node.node_name

            if not self._should_consider_scope(node_name):
                nncf_logger.info("Ignored adding Weight Sparsifier in scope: {}".format(node_name))
                continue

            nncf_logger.info("Adding Weight Sparsifier in scope: {}".format(node_name))
            compression_lr_multiplier = \
                self.config.get_redefinable_global_param_value_for_algo('compression_lr_multiplier',
                                                                        self.name)
            sparsifying_operation = self.create_weight_sparsifying_operation(module_node, compression_lr_multiplier)
            hook = UpdateWeightAndBias(sparsifying_operation).to(device)
            insertion_commands.append(
                PTInsertionCommand(
                    PTTargetPoint(TargetType.PRE_LAYER_OPERATION, target_node_name=node_name),
                    hook,
                    TransformationPriority.SPARSIFICATION_PRIORITY
                )
            )
            sparsified_module = target_model.get_containing_module(node_name)
            self._sparsified_module_info.append(
                SparseModuleInfo(node_name, sparsified_module, sparsifying_operation))

        return insertion_commands

    def create_weight_sparsifying_operation(self, target_module_node: NNCFNode, compression_lr_multiplier: float):
        sparse_cfg = SparseConfig(SparseStructure.FINE)
        node_name = target_module_node.node_name
        for configs_per_scopes in self._sparse_configs_by_scopes:
            target_scopes = configs_per_scopes.target_scopes
            if matches_any(node_name, target_scopes):
                sparse_cfg = configs_per_scopes.sparse_config
                break

        return MovementSparsifier(target_module_node, sparse_cfg=sparse_cfg, frozen=False,
                                  compression_lr_multiplier=compression_lr_multiplier)

    def _build_controller(self, model: NNCFNetwork) -> PTCompressionAlgorithmController:
        return MovementSparsityController(model, self._sparsified_module_info, self.config)


@ADAPTIVE_COMPRESSION_CONTROLLERS.register('pt_movement_sparsity')
class MovementSparsityController(BaseSparsityAlgoController):
    def __init__(self, target_model: NNCFNetwork, sparsified_module_info: List[SparseModuleInfo],
                 config: NNCFConfig):
        super().__init__(target_model, sparsified_module_info)
        algo_config = extract_algo_specific_config(config, 'movement_sparsity')
        self._distributed = False
        sparsify_operations = [m.operand for m in self.sparsified_module_info]
        params = deepcopy(algo_config.get('params', {}))
        self._scheduler = MovementPolynomialThresholdScheduler(self, params)
        self._loss = ImportanceLoss(sparsify_operations, self.scheduler)

        # TODO: review - perhaps not the right place
        self.config = config
        if self._scheduler.enable_structured_masking:
            model_family = detect_supported_model_family(self.model)
            if model_family not in STRUCTURED_MASK_STRATEGY.registry_dict:
                raise RuntimeError("You set `enable_structured_masking=True`, but no supported model is detected. "
                                   "Supported model families: {}".format(list(STRUCTURED_MASK_STRATEGY.keys())))
            strategy_cls = STRUCTURED_MASK_STRATEGY.get(model_family)
            strategy = strategy_cls.from_compressed_model(self.model)
            self._structured_mask_handler = StructuredMaskHandler(self.model,
                                                                  self.sparsified_module_info,
                                                                  strategy)

    def compression_stage(self) -> CompressionStage:
        # if self._mode == 'local':
        #     return CompressionStage.FULLY_COMPRESSED
        if self.scheduler.current_epoch < self.scheduler.warmup_start_epoch:
            return CompressionStage.UNCOMPRESSED
        if self.scheduler.current_sparsity_level >= self.scheduler.warmup_end_epoch:
            return CompressionStage.FULLY_COMPRESSED
        return CompressionStage.PARTIALLY_COMPRESSED

    def distributed(self):
        if not dist.is_initialized():
            raise KeyError('Could not set distributed mode for the compression algorithm '
                           'because the default process group has not been initialized.')

        if next(self._model.parameters()).is_cuda:
            state = torch.cuda.get_rng_state()
            if dist.get_backend() == dist.Backend.NCCL:
                state = state.cuda()
            torch.distributed.broadcast(state, src=0)
            torch.cuda.set_rng_state(state.cpu())
        else:
            state = torch.get_rng_state()
            torch.distributed.broadcast(state, src=0)
            torch.set_rng_state(state)

        self._distributed = True

    def _check_distributed_masks(self):
        if not self._distributed or get_world_size() == 1:
            return 1

        nvalues = 0
        ncor_values = 0
        eps = 1e-4
        for minfo in self.sparsified_module_info:
            mask = minfo.operand.mask

            mask_list = [torch.empty_like(mask) for _ in range(get_world_size())]
            # nccl does not support gather, send, recv operations
            dist.all_gather(mask_list, mask)

            for i in range(1, len(mask_list)):
                rel_error = (mask_list[0] - mask_list[i]) / mask_list[0]
                ncor_values = ncor_values + (rel_error.abs() < eps).sum(dtype=mask.dtype)
                nvalues = nvalues + mask_list[i].numel()

        return ncor_values / nvalues

    def statistics(self, quickly_collected_only=False) -> NNCFStatistics:
        collector = PTSparseModelStatisticsCollector(self.model, self.sparsified_module_info,
                                                     supports_sparse_bias=True)
        model_statistics = collector.collect()

        stats = MovementSparsityStatistics(model_statistics,
                                           self.scheduler.current_importance_threshold,
                                           self.scheduler.current_importance_lambda)

        nncf_stats = NNCFStatistics()
        nncf_stats.register('movement_sparsity', stats)
        return nncf_stats

    def reset_independent_structured_mask(self):
        assert self._scheduler.enable_structured_masking is True
        self._structured_mask_handler.update_independent_structured_mask()

    def resolve_structured_mask(self):
        assert self._scheduler.enable_structured_masking is True
        self._structured_mask_handler.resolve_dependent_structured_mask()

    def populate_structured_mask(self):
        assert self._scheduler.enable_structured_masking is True
        self._structured_mask_handler.populate_dependent_structured_mask_to_operand()
        self._structured_mask_handler.report_structured_sparsity(self.config.get('log_dir', '.'))

    @property
    def compression_rate(self):
        return self.statistics().movement_sparsity.model_statistics.sparsity_level

    def prepare_for_export(self):
        """
        Applies pruning masks to layer weights before exporting the model to ONNX.
        """
        self._propagate_masks()

    def _propagate_masks(self):
        # TODO(yujie): change the O(mn) complexity
        sparse_state_dict = OrderedDict()
        with torch.no_grad():
            for minfo in self.sparsified_module_info:
                for name, module in self.model.named_modules():
                    if module == minfo.module:
                        sparse_state_dict[name + '.weight'] = \
                            minfo.operand.apply_binary_mask(module.weight)
                        if hasattr(module, 'bias') and module.bias is not None:
                            sparse_state_dict[name + '.bias'] = \
                                minfo.operand.apply_binary_mask(module.bias, is_bias=True)

        model_state_dict = self.model.state_dict()
        for key, value in sparse_state_dict.items():
            assert key in model_state_dict, f'sparse parameter <{key}> is not found in model state dict.'
            model_state_dict[key] = value
        self.model.load_state_dict(model_state_dict)
