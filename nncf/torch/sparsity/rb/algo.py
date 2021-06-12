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

import torch
import torch.distributed as dist

from nncf.torch.algo_selector import COMPRESSION_ALGORITHMS
from nncf.api.compression import CompressionStage
from nncf.common.graph import NNCFNode
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.sparsity.base_algo import BaseSparsityAlgoBuilder, BaseSparsityAlgoController, SparseModuleInfo
from nncf.torch.sparsity.rb.layers import RBSparsifyingWeight
from nncf.torch.sparsity.rb.loss import SparseLoss, SparseLossForPerLayerSparsity
from nncf.torch.sparsity.collector import PTSparseModelStatisticsCollector
from nncf.common.sparsity.schedulers import SPARSITY_SCHEDULERS
from nncf.common.schedulers import StubCompressionScheduler
from nncf.common.sparsity.statistics import RBSparsityStatistics
from nncf.torch.utils import get_world_size
from nncf.common.statistics import NNCFStatistics


@COMPRESSION_ALGORITHMS.register('rb_sparsity')
class RBSparsityBuilder(BaseSparsityAlgoBuilder):
    def apply_to(self, target_model: NNCFNetwork) -> NNCFNetwork:
        target_model = super().apply_to(target_model)
        return target_model

    def create_weight_sparsifying_operation(self, target_module_node: NNCFNode, compression_lr_multiplier: float):
        return RBSparsifyingWeight(target_module_node.layer_attributes.get_weight_shape(), frozen=False,
                                   compression_lr_multiplier=compression_lr_multiplier)

    def build_controller(self, target_model: NNCFNetwork) -> PTCompressionAlgorithmController:
        return RBSparsityController(target_model, self._sparsified_module_info, self.config)


class RBSparsityController(BaseSparsityAlgoController):
    def __init__(self, target_model: NNCFNetwork, sparsified_module_info: List[SparseModuleInfo], config):
        super().__init__(target_model, sparsified_module_info)
        params = config.get('params', {})

        self._distributed = False
        self._mode = params.get('sparsity_level_setting_mode', 'global')
        self._check_sparsity_masks = params.get('check_sparsity_masks', False)

        sparsify_operations = [m.operand for m in self.sparsified_module_info]
        if self._mode == 'local':
            self._loss = SparseLossForPerLayerSparsity(sparsify_operations)
            self._scheduler = StubCompressionScheduler()
        else:
            self._loss = SparseLoss(sparsify_operations)

            sparsity_init = config.get('sparsity_init', 0)
            params['sparsity_init'] = sparsity_init
            scheduler_cls = SPARSITY_SCHEDULERS.get(params.get('schedule', 'exponential'))
            self._scheduler = scheduler_cls(self, params)
            self.set_sparsity_level(sparsity_init)

    def set_sparsity_level(self, sparsity_level, target_sparsified_module_info: SparseModuleInfo = None):
        if target_sparsified_module_info is None:
            #pylint:disable=no-value-for-parameter
            self._loss.set_target_sparsity_loss(sparsity_level)
        else:
            sparse_op = target_sparsified_module_info.operand
            self._loss.set_target_sparsity_loss(sparsity_level, sparse_op)

    def compression_stage(self) -> CompressionStage:
        if self._mode == 'local':
            return CompressionStage.FULLY_COMPRESSED

        if self.scheduler.current_sparsity_level == 0:
            return CompressionStage.UNCOMPRESSED
        if self.scheduler.current_sparsity_level >= self.scheduler.target_level:
            return CompressionStage.FULLY_COMPRESSED
        return CompressionStage.PARTIALLY_COMPRESSED

    def freeze(self):
        self._loss.disable()

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
        collector = PTSparseModelStatisticsCollector(self.model, self.sparsified_module_info)
        model_statistics = collector.collect()

        target_sparsity_level = self.scheduler.current_sparsity_level if self._mode == 'global' else None

        mean_sparse_prob = 1.0 - self.loss.mean_sparse_prob

        stats = RBSparsityStatistics(model_statistics, target_sparsity_level, mean_sparse_prob)

        nncf_stats = NNCFStatistics()
        nncf_stats.register('rb_sparsity', stats)
        return nncf_stats
