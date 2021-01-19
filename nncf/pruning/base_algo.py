"""
 Copyright (c) 2020 Intel Corporation
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
from typing import Dict, List

from functools import partial, update_wrapper
from texttable import Texttable
from torch import nn

from nncf.compression_method_api import CompressionAlgorithmBuilder, \
    CompressionAlgorithmController
from nncf.dynamic_graph.context import Scope
from nncf.dynamic_graph.graph import NNCFNode
from nncf.module_operations import UpdateWeight
from nncf.nncf_logger import logger as nncf_logger
from nncf.nncf_network import NNCFNetwork, InsertionPoint, InsertionCommand, InsertionType, OperationPriority
from nncf.pruning.export_helpers import IdentityMaskForwardOps
from nncf.pruning.filter_pruning.layers import apply_filter_binary_mask
from nncf.pruning.model_analysis import cluster_special_ops, NodesCluster, Clusterization, ModelAnalyzer
from nncf.pruning.utils import get_bn_for_module_scope, \
    get_first_pruned_modules, get_last_pruned_modules, is_conv_with_downsampling, is_grouped_conv, is_depthwise_conv, \
    get_previous_conv, get_sources_of_node


class BatchNormInfo:
    def __init__(self, module_scope: Scope, module: nn.Module, nncf_id: int):
        self.module = module
        self.module_scope = module_scope
        self.nncf_node_id = nncf_id

class PrunedModuleInfo:
    BN_MODULE_NAME = 'bn_module'

    def __init__(self, module_scope: Scope, module: nn.Module, operand, related_modules: Dict, node_id: int):
        self.module_scope = module_scope
        self.module = module
        self.operand = operand
        self.related_modules = related_modules
        self.nncf_node_id = node_id


class NodeInfo:
    def __init__(self, nncf_node: NNCFNode, module: nn.Module, module_scope: Scope):
        self.node = nncf_node
        self.id = nncf_node.node_id
        self.module = module
        self.module_scope = module_scope


class BasePruningAlgoBuilder(CompressionAlgorithmBuilder):
    def __init__(self, config, should_init: bool = True):
        super().__init__(config, should_init)
        params = config.get('params', {})
        self._params = params

        self.ignore_frozen_layers = True
        self.prune_first = params.get('prune_first_conv', False)
        self.prune_last = params.get('prune_last_conv', False)
        self.prune_batch_norms = params.get('prune_batch_norms', True)
        self.prune_downsample_convs = params.get('prune_downsample_convs', False)

        self.pruned_module_groups_info = []

    def _apply_to(self, target_model: NNCFNetwork) -> List[InsertionCommand]:
        return self._prune_weights(target_model)

    def _check_pruning_groups(self, target_model: NNCFNetwork, pruned_nodes_clusterization: Clusterization,
                              can_prune: dict):
        """
        Check whether all nodes in group can be pruned based on user-defined constraints and
         connections inside the network. Otherwise the whole group cannot be pruned.
        :param target_model: model to work with
        :param pruned_nodes_clusterization: clusterization with pruning nodes groups
        :param can_prune: dict with bool value: can this node be pruned or not
        """
        for cluster in list(pruned_nodes_clusterization.get_all_clusters()):
            cluster_nodes_names = [str(n.module_scope) for n in cluster.nodes]
            # Check whether this node should be pruned according to the user-defined algorithm constraints
            should_prune_modules = [self._is_module_prunable(target_model, node_info.module,
                                                             node_info.module_scope) for node_info in
                                    cluster.nodes]

            # Check whether this node can be potentially pruned from architecture point of view
            can_prune_nodes = [can_prune[node_info.id] for node_info in cluster.nodes]
            if not all([can_prune[0] for can_prune in should_prune_modules]):
                shouldnt_prune_msgs = [should_prune[1] for should_prune in should_prune_modules if not should_prune[0]]
                nncf_logger.info("Group of nodes [{}] can't be pruned, because some nodes should't be pruned, "
                                 "error messages for this nodes: {}".format(", ".join(cluster_nodes_names),
                                                                            ", ".join(shouldnt_prune_msgs)))
                pruned_nodes_clusterization.delete_cluster(cluster.id)
            elif not all(can_prune_nodes):
                cant_prune_nodes_names = [str(node_info.module_scope) for node_info in cluster.nodes
                                          if not can_prune[node_info.id]]
                nncf_logger.info("Group of nodes [{}] can't be pruned, because {} nodes can't be pruned "
                                 "according to model analysis"
                                 .format(", ".join(cluster_nodes_names), ", ".join(cant_prune_nodes_names)))
                pruned_nodes_clusterization.delete_cluster(cluster.id)
            else:
                nncf_logger.info("Group of nodes [{}] will be pruned together.".format(", ".join(cluster_nodes_names)))

    def _create_pruning_groups(self, target_model: NNCFNetwork):
        """
        This function groups ALL modules with pruning types to groups that should be pruned together.
        1. Create clusters for special ops (eltwises) that should be pruned together
        2. Create groups of nodes that should be pruned together (taking into account clusters of special ops)
        3. Add remaining single nodes
        4. Unite clusters for Conv + Depthwise conv (should be pruned together too)
        5. Checks for groups (all nodes in group can prune or all group can't be pruned)
        Return groups of modules that should be pruned together.
        :param target_model: model to work with
        :return: clusterisation of pruned nodes
        """
        graph = target_model.get_original_graph()
        pruned_types = self.get_op_types_of_pruned_modules()
        all_modules_to_prune = target_model.get_nncf_modules_by_module_names(self.compressed_nncf_module_names)
        all_nodes_to_prune = graph.get_nodes_by_types(pruned_types)  # NNCFNodes here
        assert len(all_nodes_to_prune) <= len(all_modules_to_prune)

        # 1. Clusters for special ops
        special_ops_types = self.get_types_of_grouping_ops()
        identity_like_types = IdentityMaskForwardOps.get_all_op_aliases()
        special_ops_clusterization = cluster_special_ops(target_model, special_ops_types,
                                                         identity_like_types)

        pruned_nodes_clusterization = Clusterization("id")

        # 2. Clusters for nodes that should be pruned together (taking into account clusters for special ops)
        for i, cluster in enumerate(special_ops_clusterization.get_all_clusters()):
            all_pruned_inputs = []
            pruned_inputs_idxs = set()

            for node in cluster.nodes:
                sources = get_sources_of_node(node, graph, pruned_types)
                for source_node in sources:
                    source_scope = source_node.op_exec_context.scope_in_model
                    source_module = target_model.get_module_by_scope(source_scope)
                    source_node_info = NodeInfo(source_node, source_module, source_scope)

                    if source_node.node_id not in pruned_inputs_idxs:
                        all_pruned_inputs.append(source_node_info)
                        pruned_inputs_idxs.add(source_node.node_id)
            if all_pruned_inputs:
                cluster = NodesCluster(i, list(all_pruned_inputs), [n.id for n in all_pruned_inputs])
                pruned_nodes_clusterization.add_cluster(cluster)

        last_cluster_idx = len(special_ops_clusterization.get_all_clusters())

        # 3. Add remaining single nodes as separate clusters
        for node in all_nodes_to_prune:
            if not pruned_nodes_clusterization.is_node_in_clusterization(node.node_id):
                scope = node.op_exec_context.scope_in_model
                module = target_model.get_module_by_scope(scope)
                node_info = NodeInfo(node, module, scope)

                cluster = NodesCluster(last_cluster_idx, [node_info], [node.node_id])
                pruned_nodes_clusterization.add_cluster(cluster)

                last_cluster_idx += 1

        # 4. Merge clusters for Conv + Depthwise conv (should be pruned together too)
        for node in all_nodes_to_prune:
            scope = node.op_exec_context.scope_in_model
            module = target_model.get_module_by_scope(scope)
            cluster_id = pruned_nodes_clusterization.get_cluster_by_node_id(node.node_id).id

            if is_depthwise_conv(module):
                previous_conv = get_previous_conv(target_model, module, scope)
                if previous_conv:
                    previous_conv_cluster_id = pruned_nodes_clusterization.get_cluster_by_node_id(
                        previous_conv.node_id).id
                    pruned_nodes_clusterization.merge_clusters(cluster_id, previous_conv_cluster_id)

        # 5. Checks for groups (all nodes in group can be pruned or all group can't be pruned).
        model_analyser = ModelAnalyzer(target_model)
        can_prune_analysis = model_analyser.analyse_model_before_pruning()
        self._check_pruning_groups(target_model, pruned_nodes_clusterization, can_prune_analysis)
        return pruned_nodes_clusterization

    def _is_module_prunable(self, target_model: NNCFNetwork, module, module_scope: Scope):
        """
        Check whether we should prune module according to algorithm parameters.
        :param target_model: model to work with
        :param module: module to check
        :param module_scope: scope for module
        :return: (prune: bool, msg: str)
        prune: Whether we should prune module
        msg: additional information why we should/shouldn't prune
        """
        prune = True
        msg = None

        pruned_types = self.get_op_types_of_pruned_modules()
        input_non_pruned_modules = get_first_pruned_modules(target_model, pruned_types + ['linear'])
        output_non_pruned_modules = get_last_pruned_modules(target_model, pruned_types + ['linear'])
        module_scope_str = str(module_scope)

        if self.ignore_frozen_layers and not module.weight.requires_grad:
            msg = "Ignored adding Weight Pruner in scope: {} because"\
                    " the layer appears to be frozen (requires_grad=False)".format(module_scope_str)
            prune = False
        elif not self._should_consider_scope(module_scope_str):
            msg = "Ignored adding Weight Pruner in scope: {}".format(module_scope_str)
            prune = False
        elif not self.prune_first and module in input_non_pruned_modules:
            msg = "Ignored adding Weight Pruner in scope: {} because"\
                             " this scope is one of the first convolutions".format(module_scope_str)
            prune = False
        elif not self.prune_last and module in output_non_pruned_modules:
            msg = "Ignored adding Weight Pruner in scope: {} because"\
                             " this scope is one of the last convolutions".format(module_scope_str)
            prune = False
        elif is_grouped_conv(module):
            if not is_depthwise_conv(module):
                msg = "Ignored adding Weight Pruner in scope: {} because" \
                      " this scope is grouped convolution".format(module_scope_str)
                prune = False
        elif not self.prune_downsample_convs and is_conv_with_downsampling(module):
            msg = "Ignored adding Weight Pruner in scope: {} because"\
                             " this scope is convolution with downsample".format(module_scope_str)
            prune = False
        return prune, msg

    def _prune_weights(self, target_model: NNCFNetwork):
        grops_of_modules_to_prune = self._create_pruning_groups(target_model)

        device = next(target_model.parameters()).device
        insertion_commands = []
        self.pruned_module_groups_info = Clusterization('module_scope')

        for i, group in enumerate(grops_of_modules_to_prune.get_all_clusters()):
            group_minfos = []
            for node in group.nodes:
                module_scope, module = node.module_scope, node.module
                # Check that we need to prune weights in this op
                assert self._is_pruned_module(module)

                module_scope_str = str(module_scope)

                nncf_logger.info("Adding Weight Pruner in scope: {}".format(module_scope_str))
                operation = self.create_weight_pruning_operation(module)
                hook = UpdateWeight(operation).to(device)
                insertion_commands.append(
                    InsertionCommand(
                        InsertionPoint(InsertionType.NNCF_MODULE_PRE_OP,
                                       module_scope=module_scope),
                        hook,
                        OperationPriority.PRUNING_PRIORITY
                    )
                )

                related_modules = {}
                if self.prune_batch_norms:
                    bn_module, bn_scope = get_bn_for_module_scope(target_model, module_scope)
                    related_modules[PrunedModuleInfo.BN_MODULE_NAME] = BatchNormInfo(module_scope,
                                                                                     bn_module, bn_scope)

                minfo = PrunedModuleInfo(module_scope, module, hook.operand, related_modules, node.id)
                group_minfos.append(minfo)
            cluster = NodesCluster(i, group_minfos, [n.id for n in group.nodes])
            self.pruned_module_groups_info.add_cluster(cluster)
        return insertion_commands

    def build_controller(self, target_model: NNCFNetwork) -> CompressionAlgorithmController:
        return BasePruningAlgoController(target_model, self.pruned_module_groups_info, self._params)

    def create_weight_pruning_operation(self, module):
        raise NotImplementedError

    def _is_pruned_module(self, module: nn.Module):
        """
        Return whether this module should be pruned or not.
        """
        raise NotImplementedError

    def get_op_types_of_pruned_modules(self):
        """
        Returns list of operation types that should be pruned.
        """
        raise NotImplementedError

    def get_types_of_grouping_ops(self):
        raise NotImplementedError


class BasePruningAlgoController(CompressionAlgorithmController):
    def __init__(self, target_model: NNCFNetwork,
                 pruned_module_groups_info: Clusterization,
                 config):
        super().__init__(target_model)
        self.config = config
        params = self.config.get("params", {})
        self.pruned_module_groups_info = pruned_module_groups_info
        self.prune_batch_norms = params.get('prune_batch_norms', True)
        self.prune_first = params.get('prune_first_conv', False)
        self.prune_last = params.get('prune_last_conv', False)
        self.zero_grad = params.get('zero_grad', True)
        self.prune_flops = False
        self.check_pruning_rate(params)
        self._hooks = []

    def freeze(self):
        raise NotImplementedError

    def set_pruning_rate(self, pruning_rate):
        raise NotImplementedError

    def step(self, next_step):
        raise NotImplementedError

    def zero_grads_for_pruned_modules(self):
        """
        This function registers a hook that will set the
        gradients for pruned filters to zero.
        """
        self._clean_hooks()

        def hook(grad, mask, dim=0):
            mask = mask.to(grad.device)
            return apply_filter_binary_mask(mask, grad, dim=dim)

        for minfo in self.pruned_module_groups_info.get_all_nodes():
            mask = minfo.operand.binary_filter_pruning_mask
            weight = minfo.module.weight
            dim = minfo.module.target_weight_dim_for_compression
            partial_hook = update_wrapper(partial(hook, mask=mask, dim=dim), hook)
            self._hooks.append(weight.register_hook(partial_hook))
            if minfo.module.bias is not None:
                bias = minfo.module.bias
                partial_hook = update_wrapper(partial(hook, mask=mask), hook)
                self._hooks.append(bias.register_hook(partial_hook))

    def check_pruning_rate(self, params):
        """
        Check that set only one of pruning target params
        """
        pruning_target = params.get('pruning_target', None)
        pruning_flops_target = params.get('pruning_flops_target', None)
        if pruning_target and pruning_flops_target:
            raise ValueError('Only one parameter from \'pruning_target\' and \'pruning_flops_target\' can be set.')
        if pruning_flops_target:
            self.prune_flops = True

    def _clean_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _get_mask(self, minfo: PrunedModuleInfo):
        """
        Returns pruning mask for minfo.module.
        """
        raise NotImplementedError

    @staticmethod
    def pruning_rate_for_weight(minfo: PrunedModuleInfo):
        """
        Calculates sparsity rate for all weight elements.
        """
        weight = minfo.module.weight
        pruning_rate = 1 - weight.nonzero().size(0) / weight.view(-1).size(0)
        return pruning_rate

    @staticmethod
    def pruning_rate_for_filters(minfo: PrunedModuleInfo):
        """
        Calculates sparsity rate for weight filter-wise.
        """
        dim = minfo.module.target_weight_dim_for_compression
        weight = minfo.module.weight.transpose(0, dim).contiguous()
        filters_sum = weight.view(weight.size(0), -1).sum(axis=1)
        pruning_rate = 1 - len(filters_sum.nonzero()) / filters_sum.size(0)
        return pruning_rate

    def pruning_rate_for_mask(self, minfo: PrunedModuleInfo):
        mask = self._get_mask(minfo)
        pruning_rate = 1 - mask.nonzero().size(0) / max(mask.view(-1).size(0), 1)
        return pruning_rate

    def mask_shape(self, minfo: PrunedModuleInfo):
        mask = self._get_mask(minfo)
        return mask.shape

    def statistics(self, quickly_collected_only=False):
        stats = super().statistics()
        table = Texttable()
        header = ["Name", "Weight's Shape", "Mask Shape", "Mask zero %", "PR", "Filter PR"]
        data = [header]

        for minfo in self.pruned_module_groups_info.get_all_nodes():
            drow = {h: 0 for h in header}
            drow["Name"] = str(minfo.module_scope)
            drow["Weight's Shape"] = list(minfo.module.weight.size())

            drow["Mask Shape"] = list(self.mask_shape(minfo))

            drow["Mask zero %"] = self.pruning_rate_for_mask(minfo) * 100

            drow["PR"] = self.pruning_rate_for_weight(minfo)

            drow["Filter PR"] = self.pruning_rate_for_filters(minfo)

            row = [drow[h] for h in header]
            data.append(row)
        table.add_rows(data)

        stats["pruning_statistic_by_module"] = table
        return self.add_algo_specific_stats(stats)


    def add_algo_specific_stats(self, stats):
        return stats

    def get_stats_for_pruned_modules(self):
        """
        Return dict with information about pruned modules. Keys in dict is module names, values is dicts with next keys:
         'w_shape': shape of module weight,
         'b_shape': shape of module bias,
         'params_count': total number of params in module
         'mask_pr': proportion of zero elements in filter pruning mask.
        """
        stats = {}
        for minfo in self.pruned_module_groups_info.get_all_nodes():
            layer_info = {}
            layer_info["w_shape"] = list(minfo.module.weight.size())
            layer_info["b_shape"] = list(minfo.module.bias.size()) if minfo.module.bias is not None else []
            layer_info["params_count"] = sum(p.numel() for p in minfo.module.parameters() if p.requires_grad)

            layer_info["mask_pr"] = self.pruning_rate_for_mask(minfo)

            if PrunedModuleInfo.BN_MODULE_NAME in minfo.related_modules and \
                    minfo.related_modules[PrunedModuleInfo.BN_MODULE_NAME].module is not None:
                bn_info = {}
                bn_module = minfo.related_modules[PrunedModuleInfo.BN_MODULE_NAME].module
                bn_info['w_shape'] = bn_module.weight.size()

                bn_info["b_shape"] = bn_module.bias.size() if bn_module.bias is not None else []
                bn_info['params_count'] = sum(p.numel() for p in bn_module.parameters() if p.requires_grad)
                bn_info["mask_pr"] = self.pruning_rate_for_mask(minfo)
                stats[str(minfo.module_scope) + '/BatchNorm'] = bn_info

            stats[str(minfo.module_scope)] = layer_info

        return stats
