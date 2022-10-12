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
from typing import DefaultDict, List, OrderedDict

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
from nncf.torch.sparsity.movement.layers import MovementSparsifier, SparseConfig, SparseStructure
from nncf.torch.sparsity.movement.loss import ImportanceLoss, SparseLossForPerLayerSparsity
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
from nncf.torch.layers import NNCF_MODULES_OP_NAMES
import os
import numpy as np
import pandas as pd


@PT_COMPRESSION_ALGORITHMS.register('movement_sparsity')
class MovementSparsityBuilder(BaseSparsityAlgoBuilder):
    def _sparsify_weights(self, target_model: NNCFNetwork) -> List[PTInsertionCommand]:
        device = get_model_device(target_model)
        sparsified_module_nodes = target_model.get_weighted_original_graph_nodes(
            nncf_module_names=self.compressed_nncf_module_names)
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
            operation = self.create_weight_sparsifying_operation(module_node, compression_lr_multiplier)
            hook = operation.to(device)
            # TODO: hardcoded to OPERATION_WITH_WEIGHT_WT_BIAS
            insertion_commands.append(PTInsertionCommand(PTTargetPoint(TargetType.OPERATION_WITH_WEIGHT_WT_BIAS,
                                                                       target_node_name=node_name),
                                                         hook, TransformationPriority.SPARSIFICATION_PRIORITY))
            sparsified_module = target_model.get_containing_module(node_name)
            self._sparsified_module_info.append(
                SparseModuleInfo(node_name, sparsified_module, hook))

        return insertion_commands

    def create_weight_sparsifying_operation(self, target_module_node: NNCFNode, compression_lr_multiplier: float):
        sparse_cfg=None
        if 'sparse_structure_by_scopes' in self._algo_config:
            for sparse_mode, sparse_args, regex in self._algo_config['sparse_structure_by_scopes']:
                if matches_any(target_module_node.node_name, regex):
                    sparse_cfg = SparseConfig(sparse_mode, sparse_args)
                    break

        if sparse_cfg is None:
            sparse_cfg = SparseConfig()

        return MovementSparsifier(
                    target_module_node, 
                    frozen=False,
                    compression_lr_multiplier=compression_lr_multiplier,
                    eps=1e-6, 
                    sparse_cfg=sparse_cfg)

    def _build_controller(self, model: NNCFNetwork) -> PTCompressionAlgorithmController:
        return MovementSparsityController(model, self._sparsified_module_info, self.config)

class StructuredMask:
    def __init__(self, 
                 target_module_node, 
                 sparsifying_node_name, 
                 grid_size,
                 dependent_group_id,
                 sparse_module_info):

        self.target_module_node=target_module_node
        self.sparsifying_node_name=sparsifying_node_name
        self.grid_size=grid_size
        self.dependent_group_id=dependent_group_id
        self.sparse_module_info=sparse_module_info

    @property
    def independent_structured_mask(self):
        return self._independent_structured_mask
    
    @independent_structured_mask.setter
    def independent_structured_mask(self, tensor):
        with torch.no_grad():
            self._independent_structured_mask = tensor
            # self._independent_structured_mask.set_(tensor)

    @property
    def dependent_structured_mask(self):
        return self._dependent_structured_mask
    
    @dependent_structured_mask.setter
    def dependent_structured_mask(self, tensor):
        # TODO: check dim
        with torch.no_grad():
            self._dependent_structured_mask = tensor
            # self._dependent_structured_mask.set_(tensor)

@ADAPTIVE_COMPRESSION_CONTROLLERS.register('pt_movement_sparsity')
class MovementSparsityController(BaseSparsityAlgoController):
    def __init__(self, target_model: NNCFNetwork, sparsified_module_info: List[SparseModuleInfo],
                 config: NNCFConfig):
        super().__init__(target_model, sparsified_module_info)
        algo_config = extract_algo_specific_config(config, 'movement_sparsity')
        params = deepcopy(algo_config.get('params', {}))

        self._distributed = False
        self._mode = params.get('sparsity_level_setting_mode', 'global')
        self._check_sparsity_masks = params.get('check_sparsity_masks', False)

        sparsify_operations = [m.operand for m in self.sparsified_module_info]
        if self._mode == 'local':
            # TODO: make sure we test this loop out
            self._loss = SparseLossForPerLayerSparsity(sparsify_operations)
            self._scheduler = StubCompressionScheduler()
        else:
            scheduler_cls = SPARSITY_SCHEDULERS.get(params.get('schedule', 'exponential')) #TODO: can we actually map to other scheduler in current implementation
            self._scheduler = scheduler_cls(self, params)
            self._loss = ImportanceLoss(sparsify_operations, self.scheduler)

        #TODO: review - perhaps not the right place
        self.config = config
        self.prunableops_per_group = self._get_group_of_prunable_ops()
        # self.visualize_groups_of_prunables()
        self.create_structured_sparsity_context()

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

        stats = MovementSparsityStatistics(model_statistics,
                                           self.scheduler.current_importance_threshold,
                                           self.scheduler.current_importance_lambda)

        nncf_stats = NNCFStatistics()
        nncf_stats.register('movement_sparsity', stats)
        return nncf_stats

    def create_structured_sparsity_context(self):
        DEBUG=False
        # Structured_mask per tensor -------------------
        node_name_sparse_mod_info_map = {sparse_info.module_node_name: sparse_info for sparse_info in self.sparsified_module_info}
        self.node_name_sparse_mod_info_map = node_name_sparse_mod_info_map

        self.structured_ctx_by_group = defaultdict(list)
        masks_per_group = dict()
        op2namedmodule = dict()

        for group_id, op_list in self.prunableops_per_group.items():
            masks_per_group[group_id]=dict()
            for op in op_list:
                sparsifying_node_name = str(op.op_addr)

                # find op's torch module name
                for n, m in self.model.named_modules():
                    if m == op.op_mod:
                        op2namedmodule[sparsifying_node_name] = n
                        break
                
                sparse_module_info = node_name_sparse_mod_info_map[sparsifying_node_name]

                if any(map(sparsifying_node_name.__contains__, ['query','key','value'])):
                    # these matrices must be pruned by group(s) of cols
                    nrow_per_head = self.model.nncf_module.bert.config.hidden_size//self.model.nncf_module.bert.config.num_attention_heads
                    ncol_per_head = self.model.nncf_module.bert.config.hidden_size
                    grid_size = (nrow_per_head, ncol_per_head)
                    
                    if DEBUG is True:
                        masks_per_group[group_id]['qkv_grain'] = grid_size
                        mask = sparse_module_info.operand.get_structured_mask(grid_size)
                        if 'qkv' not in masks_per_group[group_id]:
                            masks_per_group[group_id]['qkv'] = [mask]
                            masks_per_group[group_id]['qkv_nodes'] = [sparsifying_node_name]
                        else:
                            masks_per_group[group_id]['qkv'].append(mask)
                            masks_per_group[group_id]['qkv_nodes'].append(sparsifying_node_name)
                        print("{:15} | {:20} | {}".format('group_of_rows', str(mask.shape), sparsifying_node_name))

                    structured_mask_ctx = StructuredMask(
                                                sparse_module_info.module_node_name,
                                                sparsifying_node_name,
                                                grid_size,
                                                group_id,
                                                sparse_module_info)

                    structured_mask_ctx.independent_structured_mask = sparse_module_info.operand.get_structured_mask(grid_size)
                    sparse_module_info.operand.structured_mask_ctx = structured_mask_ctx
                    self.structured_ctx_by_group[group_id].append(sparse_module_info.operand.structured_mask_ctx)

                    if DEBUG is True:
                        assert ((mask==structured_mask_ctx.independent_structured_mask).sum() == mask.numel()).item(), "qkv: Logical Bug, pls debug"
                    
                elif 'BertSelfOutput' in sparsifying_node_name:
                    # this matrix must be pruned by group(s) of cols
                    ncol_per_head = self.model.nncf_module.bert.config.hidden_size//self.model.nncf_module.bert.config.num_attention_heads
                    nrow_per_head = self.model.nncf_module.bert.config.hidden_size
                    grid_size = (nrow_per_head, ncol_per_head)

                    if DEBUG is True:
                        masks_per_group[group_id]['concat_grain'] = grid_size
                        mask = sparse_module_info.operand.get_structured_mask(grid_size)
                        masks_per_group[group_id]['concat'] = mask
                        masks_per_group[group_id]['concat_node'] = sparsifying_node_name
                        print("{:15} | {:20} | {}".format('group_of_cols', str(mask.shape), sparsifying_node_name))

                    structured_mask_ctx = StructuredMask(
                                                sparse_module_info.module_node_name,
                                                sparsifying_node_name,
                                                grid_size,
                                                group_id,
                                                sparse_module_info)

                    structured_mask_ctx.independent_structured_mask = sparse_module_info.operand.get_structured_mask(grid_size)
                    sparse_module_info.operand.structured_mask_ctx = structured_mask_ctx
                    self.structured_ctx_by_group[group_id].append(sparse_module_info.operand.structured_mask_ctx)

                    if DEBUG is True:
                        assert ((mask==structured_mask_ctx.independent_structured_mask).sum() == mask.numel()).item(), "BertSelfOutput: Logical Bug, pls debug"

                elif any(map(sparsifying_node_name.__contains__, ['BertIntermediate','BertOutput'])):
                    mask = sparse_module_info.operand.get_structured_mask()
                    grid_size = sparse_module_info.operand.sparse_cfg.sparse_factors

                    if DEBUG is True:
                        if 'BertIntermediate' in sparsifying_node_name:
                            masks_per_group[group_id]['ffnn_w1_grain'] = grid_size
                            masks_per_group[group_id]['ffnn_w1'] = mask
                            masks_per_group[group_id]['ffnn_w1_node'] = sparsifying_node_name
                        elif 'BertOutput' in sparsifying_node_name:
                            masks_per_group[group_id]['ffnn_w2_grain'] = grid_size
                            masks_per_group[group_id]['ffnn_w2'] = mask
                            masks_per_group[group_id]['ffnn_w2_node'] = sparsifying_node_name
                        print("{:15} | {:20} | {}".format('per_dim', str(mask.shape), sparsifying_node_name))

                    structured_mask_ctx = StructuredMask(
                                                sparse_module_info.module_node_name,
                                                sparsifying_node_name,
                                                grid_size,
                                                group_id,
                                                sparse_module_info)

                    structured_mask_ctx.independent_structured_mask = sparse_module_info.operand.get_structured_mask(grid_size)
                    sparse_module_info.operand.structured_mask_ctx = structured_mask_ctx
                    self.structured_ctx_by_group[group_id].append(sparse_module_info.operand.structured_mask_ctx)

                    if DEBUG is True:
                        assert ((mask==structured_mask_ctx.independent_structured_mask).sum() == mask.numel()).item(), "ffnn: Logical Bug, pls debug"
                else:
                    raise ValueError("Invalid entry, pls debug")
        
        self.op2namedmodule = op2namedmodule

        # This Structure can be improved but good for now: TODO: revision of structure
        # masks_per_group[group_id][sparsifying_node_name] = mask

    def reset_independent_structured_mask(self):
        for group_id, ctxes in self.structured_ctx_by_group.items():
            for ctx in ctxes:
                ctx.independent_structured_mask = ctx.sparse_module_info.operand.get_structured_mask(ctx.grid_size)

    def populate_structured_mask(self):
        def inflate_structured_mask(mask, grid_size):
            assert len(mask.shape) == len(grid_size), "Unmatching dimension"
            inflated_mask = mask.detach().clone()
            for axis, repeat in enumerate(grid_size):
                inflated_mask = inflated_mask.repeat_interleave(repeat, dim=axis)
            return inflated_mask

        for group_id, ctxes in self.structured_ctx_by_group.items():
            for ctx in ctxes:
                ctx.sparse_module_info.operand.set_structured_mask(
                    inflate_structured_mask(ctx.dependent_structured_mask, ctx.grid_size)
                )

    def resolve_structured_mask(self):
        for group_id, ctxes in self.structured_ctx_by_group.items():
            allnodenames = list(map(lambda x: x.target_module_node, ctxes))
            
            if any(map(ctxes[0].target_module_node.__contains__, ['query','key','value','BertSelfOutput'])):
                qid = list(map(lambda x: x.__contains__('query'), allnodenames)).index(True)
                kid = list(map(lambda x: x.__contains__('key'), allnodenames)).index(True)
                vid = list(map(lambda x: x.__contains__('value'), allnodenames)).index(True)
                oid = list(map(lambda x: x.__contains__('BertSelfOutput'), allnodenames)).index(True)

                coarse_mask = ctxes[qid].independent_structured_mask.logical_or(
                                ctxes[kid].independent_structured_mask).logical_or(
                                    ctxes[vid].independent_structured_mask).logical_or(
                                        ctxes[oid].independent_structured_mask.transpose(0, 1)
                                    ).to(torch.float32)
                ctxes[qid].dependent_structured_mask = coarse_mask
                ctxes[kid].dependent_structured_mask = coarse_mask
                ctxes[vid].dependent_structured_mask = coarse_mask
                ctxes[oid].dependent_structured_mask = coarse_mask.transpose(0, 1)
            elif any(map(ctxes[0].target_module_node.__contains__, ['BertIntermediate','BertOutput'])):
                w1_id = list(map(lambda x: x.__contains__('BertIntermediate'), allnodenames)).index(True)
                w2_id = list(map(lambda x: x.__contains__('BertOutput'), allnodenames)).index(True)
                coarse_mask = ctxes[w1_id].independent_structured_mask.logical_or(
                                ctxes[w2_id].independent_structured_mask.transpose(0, 1)
                              ).to(torch.float32)

                ctxes[w1_id].dependent_structured_mask = coarse_mask
                ctxes[w2_id].dependent_structured_mask = coarse_mask.transpose(0, 1)
            else:
                raise ValueError("logical bug, pls debug")

        # # Structured_mask alignment by group -------------------

        # for group_id, mask_dict in masks_per_group.items():
        #     if 'qkv' in mask_dict:
        #         final_mask = torch.zeros_like(mask_dict['qkv'][0]).to(torch.bool)
        #         for each_mask in mask_dict['qkv']:
        #             final_mask = final_mask.logical_or(each_mask)
        #         final_mask = final_mask.logical_or(mask_dict['concat'].transpose(0, 1))
        #         final_mask = final_mask.to(torch.float32)

        #         masks_per_group[group_id]['final_structured_mask'] = dict(
        #             qkv=final_mask,
        #             concat=final_mask.transpose(0, 1)
        #         )

        #     elif 'ffnn_w1' in mask_dict:
        #         final_mask = mask_dict['ffnn_w1'].logical_or(mask_dict['ffnn_w2'].transpose(0, 1))
        #         final_mask = final_mask.to(torch.float32)

        #         masks_per_group[group_id]['final_structured_mask'] = dict(
        #             ffnn_w1=final_mask,
        #             ffnn_w2=final_mask.transpose(0, 1)
        #         )
        #     else:
        #         raise ValueError("Invalid entry, pls debug")

    def report_structured_sparsity(self, dirname):
        listofentry=[]
        for group_id, ctxes in self.structured_ctx_by_group.items():
            for ctx in ctxes:
                nncf_graph_node_name = ctx.sparsifying_node_name
                named_mod = self.op2namedmodule[nncf_graph_node_name]
                block_id = group_id
                orig_wshape = tuple(list(ctx.sparse_module_info.module.weight.shape))
                if hasattr(ctx.sparse_module_info.module, 'bias'):
                    orig_bshape = tuple(list(ctx.sparse_module_info.module.bias.shape))

                if any(map(nncf_graph_node_name.__contains__, ['BertIntermediate','BertOutput'])):
                    head_id_to_keep = 'skip reporting'
                    if nncf_graph_node_name.__contains__('BertIntermediate'):
                        final_wshape = (ctx.sparse_module_info.operand.weight_ctx.binary_mask.amax(dim=1).count_nonzero().item(), orig_wshape[1])
                        final_bshape = (ctx.sparse_module_info.operand.bias_ctx.binary_mask.count_nonzero().item(),)
                    else:
                        final_wshape = (orig_wshape[0], ctx.sparse_module_info.operand.weight_ctx.binary_mask.amax(dim=0).count_nonzero().item())
                        final_bshape = (ctx.sparse_module_info.operand.bias_ctx.binary_mask.count_nonzero().item(),)
                else:
                    ndiv = ctx.dependent_structured_mask.reshape(-1).shape[0]
                    head_id_to_keep = torch.masked_select(torch.range(0, ndiv-1, dtype=int), 
                                        ctx.dependent_structured_mask.reshape(-1).cpu().to(bool)).tolist()

                    if any(map(nncf_graph_node_name.__contains__, ['query','key','value'])):
                        # prune by row
                        final_wshape = (ctx.sparse_module_info.operand.weight_ctx.binary_mask.amax(dim=1).count_nonzero().item(), orig_wshape[1])
                        final_bshape = (ctx.sparse_module_info.operand.bias_ctx.binary_mask.count_nonzero().item(),)
                    else:
                        # prune by col
                        final_wshape = (orig_wshape[0], ctx.sparse_module_info.operand.weight_ctx.binary_mask.amax(dim=0).count_nonzero().item())
                        final_bshape = (ctx.sparse_module_info.operand.bias_ctx.binary_mask.count_nonzero().item(),)

                listofentry.append(
                    OrderedDict(
                        pt_module_name=named_mod,
                        block_id=block_id,
                        weight_shape=orig_wshape,
                        prune_w_shape=final_wshape,
                        bias_shape=orig_bshape,
                        prune_b_shape=final_bshape,
                        head_id_to_keep=head_id_to_keep,
                        nncf_graph_node=nncf_graph_node_name
                    )
                )
        df = pd.DataFrame.from_dict(listofentry)
        df.to_csv(os.path.join(dirname, 'structured_sparsity.csv'))
        with open(os.path.join(dirname, 'structured_sparsity.md'), 'w') as f:
            df.to_markdown(f)


    @property
    def compression_rate(self):
        return self.statistics().movement_sparsity.model_statistics.sparsity_level

    def prepare_for_export(self):
        """
        Applies pruning masks to layer weights before exporting the model to ONNX.
        """
        self._propagate_masks()

    def _propagate_masks(self):
        def calc_sparsity(tensor):
            return 1-tensor.count_nonzero()/tensor.numel()
        # nncf_logger.debug("MVMT - Propagating pruning masks")
        # 1. Propagate masks for all modules
        from collections import OrderedDict
        sparse_sd = OrderedDict()
        with torch.no_grad():    
            for sparse_info in self.sparsified_module_info:
                for n, m in self.model.named_modules():
                    if m == sparse_info.module:
                        # print("- SparseModule: {} -".format(n))
                        # print("\tw_mask sparsity: {:.3f}".format(calc_sparsity(sparse_info.operand.weight_ctx.binary_mask)))
                        # print("\tw_sd   sparsity: {:.3f}".format(calc_sparsity(m.weight)))
                        sparse_sd[n+'.weight'] = sparse_info.operand.apply_binary_mask(m.weight)
                        # print("\t*w_sd  sparsity: {:.3f}".format(calc_sparsity(sparse_sd[n+'.weight'])))

                        if hasattr(m, 'bias'):
                            # print("\tb_mask sparsity: {:.3f}".format(calc_sparsity(sparse_info.operand.bias_ctx.binary_mask)))
                            # print("\tb_sd   sparsity: {:.3f}".format(calc_sparsity(m.bias)))
                            sparse_sd[n+'.bias'] = sparse_info.operand.apply_binary_mask(m.bias, isbias=True)
                            # print("\t*w_sd  sparsity: {:.3f}".format(calc_sparsity(sparse_sd[n+'.bias'])))

        model_sd = self.model.state_dict()
        for k, v in sparse_sd.items():
            assert k in model_sd, "key not exists!"
            model_sd[k] = sparse_sd[k]
        self.model.load_state_dict(model_sd)

    def print_prunableops_per_group(self):
        for group, op_list in self.prunableops_per_group.items():
            print("= Group {} ======".format(group))
            print('\n'.join(list(map(lambda x: '{:12} | {}'.format(str(list(x.op_mod.weight.shape)), str(x.op_addr)), op_list))))
  
    def _get_group_of_prunable_ops(self):
        PrunableOp = namedtuple("PrunableOp", "op_addr op_mod")

        building_blocks, _ = get_building_blocks(self.model,
                                target_block_types=[BuildingBlockType.MSHA, BuildingBlockType.FF],
                                block_filter_strategy=BlockFilteringStrategy.KEEP_SMALL,
                                hw_fused_ops=True)

        all_node_op_addr_in_blocks = self._get_all_node_op_addresses_in_block(self.model, building_blocks)

        prunableops_per_group = {}
        for group_id, nodes_per_block in all_node_op_addr_in_blocks.items():
            prunableops_per_group[group_id] = []

            for str_op_addr in nodes_per_block:
                op_address = OperationAddress.from_str(str_op_addr)
                if op_address.operator_name in NNCF_MODULES_OP_NAMES:

                    prunableops_per_group[group_id].append(
                        PrunableOp(
                            op_address,
                            self.model.get_module_by_scope(op_address.scope_in_model)
                        )
                    )
        return prunableops_per_group

    def _get_all_node_op_addresses_in_block(self, nncf_network, blocks):
        graph = nncf_network.get_original_graph()
        all_nodes_per_skipped_block_idxs = {}
        for idx, block in enumerate(blocks):
            start_node, end_node = block.start_node_name, block.end_node_name
            start_node_key, end_node_key = None, None
            for node in graph._nx_graph._node.values():
                if start_node == str(node['node_name']):
                    start_node_key = node['key']
                if end_node == str(node['node_name']):
                    end_node_key = node['key']
            simple_paths = nx.all_simple_paths(graph._nx_graph, start_node_key, end_node_key)
            all_nodes_in_block = set()
            for node_keys_in_path in simple_paths:
                for node_key in node_keys_in_path:
                    all_nodes_in_block.add(str(graph._nx_graph._node[node_key]['node_name']))
            start_op_address = str(graph._nx_graph._node[start_node_key]['node_name'])
            all_nodes_in_block.remove(start_op_address)
            all_nodes_per_skipped_block_idxs[idx] = list(all_nodes_in_block)
        return all_nodes_per_skipped_block_idxs

    def visualize_groups_of_prunables(self, path=None):
        import networkx as nx
        from nncf.torch.graph.graph import PTNNCFGraph
        from networkx.drawing.nx_agraph import to_agraph
        import matplotlib._color_data as mcd
        import matplotlib.pyplot as plt
        import numpy as np
        palette = np.array(list(mcd.CSS4_COLORS.keys())).reshape(-1, 4).transpose().reshape(-1).tolist()

        from matplotlib.colors import to_hex
        palette = np.array([to_hex(c) for c in plt.get_cmap("tab20b").colors]).reshape(-1, 5).transpose().reshape(-1).tolist()
        
        learnable_node_color_map = dict()
        opbook = dict()

        for group_id, op_list in self.prunableops_per_group.items():
            color = palette[group_id % len(palette)]
            for op in op_list:
                learnable_node_color_map[str(op.op_addr)] = color
                opbook[str(op.op_addr)] = op

        building_blocks  = get_building_blocks(self.model, allow_nested_blocks=False)
        node_op_address_per_block = self._get_all_node_op_addresses_in_block(self.model, building_blocks)
        node_color_map = dict()
        for group_id, op_list in node_op_address_per_block.items():
            color = palette[group_id % len(palette)]
            for op in op_list:
                node_color_map[op] = color

        g = self.model.get_graph()

        out_graph = nx.DiGraph()
        for node_name, node in g._nx_graph.nodes.items():
            # ia_op_exec_context = node[PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR]

            attrs_node = {}
            label = node['key']
            # label = str(node[PTNNCFGraph.ID_NODE_ATTR]) + ' ' + str(ia_op_exec_context)
            # if 'conv2d' in label.lower():
            #     label = "*prunable*\n" + label
            tokens=label.split("/")
            new_tokens=[]
            for i, token in enumerate(tokens):
                if (i+1)%2==0:
                    token += "\n"
                new_tokens.append(token)
            attrs_node['label'] = '/'.join(new_tokens)

            if node['node_name'] in node_color_map:
                # cluster_id = self.df.cluster_id[self.df.node_name == node_name].values[0]
                # attrs_node['label'] += "\n(cluster {})".format(cluster_id)
                # mcd.CSS4_COLORS
                # attrs_node['color'] = mcd.CSS4_COLORS[node_color_map[node['node_name']]]

                
                attrs_node['color'] = node_color_map[node['node_name']]
                if node['node_name'] in learnable_node_color_map:
                    attrs_node['label'] += "\n{}\n".format(str(tuple(opbook[node['node_name']].op_mod.weight.shape)))
                    attrs_node['style'] = 'filled'
                else:
                    attrs_node['style'] = 'diagonals'
                    # At present, there are 8 style values recognized: filled , invisible , diagonals , rounded . dashed , dotted , solid and bold

            out_graph.add_node(node_name, **attrs_node)

        for u, v in g._nx_graph.edges:
            out_graph.add_edge(u, v, label=g._nx_graph.edges[u, v][PTNNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR])

        mapping = {k: v["label"] for k, v in out_graph.nodes.items()}
        out_graph = nx.relabel_nodes(out_graph, mapping)
        for node in out_graph.nodes.values():
            node.pop("label")

        if path is None:
            path = 'mvmt_prunableops_group_viz.dot'
        path = os.path.join(self.config.get("log_dir", "."), path)
        
        nx.drawing.nx_pydot.write_dot(out_graph, path)

        try:
            A = to_agraph(out_graph)
            A.layout('dot')
            png_path = os.path.splitext(path)[0]+'.png'
            A.draw(png_path)
        except ImportError:
            print("Graphviz is not installed - only the .dot model visualization format will be used. "
                                "Install pygraphviz into your Python environment and graphviz system-wide to enable "
                                "PNG rendering.")